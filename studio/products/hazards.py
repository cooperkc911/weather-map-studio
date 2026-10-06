"""National hazard scan: is a big winter storm / cold blast / heat wave / wind event / flood underway?

Counting uses the zone lists already in the NWS alert feed (cheap). Only when a post is due do we
download zone shapes, every METAR in the country, and WPC's fronts, and draw the analysis map.
"""
import math
import re
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import config, geo, net

# group -> NWS event names, VTEC codes for shapes, zone-count threshold, map color, obs to plot
GROUPS = {
    "blizzard": {"events": {"Blizzard Warning"}, "vtec": {"BZ.W"}, "threshold": 12,
                 "label": "Blizzard Warnings", "color": "#ff4500", "obs": "winter"},
    "ice": {"events": {"Ice Storm Warning"}, "vtec": {"IS.W"}, "threshold": 10,
            "label": "Ice Storm Warnings", "color": "#8b008b", "obs": "winter"},
    "winter": {"events": {"Winter Storm Warning", "Lake Effect Snow Warning"}, "vtec": {"WS.W", "LE.W"},
               "threshold": 120, "label": "Winter Storm Warnings", "color": "#ff69b4", "obs": "winter"},
    "cold": {"events": {"Extreme Cold Warning", "Wind Chill Warning"}, "vtec": {"EC.W", "WC.W"},
             "threshold": 120, "label": "Extreme Cold Warnings", "color": "#3a6bff", "obs": "cold"},
    "heat": {"events": {"Extreme Heat Warning", "Excessive Heat Warning"}, "vtec": {"XH.W", "EH.W"},
             "threshold": 150, "label": "Extreme Heat Warnings", "color": "#c71585", "obs": "heat"},
    "wind": {"events": {"High Wind Warning", "Extreme Wind Warning"}, "vtec": {"HW.W", "EW.W"},
             "threshold": 100, "label": "High Wind Warnings", "color": "#daa520", "obs": "wind"},
    "fire": {"events": {"Red Flag Warning"}, "vtec": {"FW.W"}, "threshold": 250,
             "label": "Red Flag Warnings", "color": "#ff1493", "obs": "fire"},
    "flood": {"events": {"Flood Warning", "Flash Flood Warning"}, "vtec": {"FA.W", "FL.W", "FF.W"},
              "threshold": 150, "label": "Flood Warnings", "color": "#00d060", "obs": None},
    "dust": {"events": {"Dust Storm Warning"}, "vtec": {"DS.W"}, "threshold": 4,
             "label": "Dust Storm Warnings", "color": "#e0c68a", "obs": "wind"},
}
for _g, _v in GROUPS.items():
    _v["threshold"] = int(__import__("os").getenv(f"HAZARD_{_g.upper()}_ZONES") or _v["threshold"])

STATE_CODES = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
               "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
               "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"}


# --- Counting (pure) -----------------------------------------------------------
def count(alerts_json):
    """Unique zones/counties under each hazard group, plus the states involved."""
    zones = {g: set() for g in GROUPS}
    for f in alerts_json.get("features", []):
        pr = f.get("properties") or {}
        ev = pr.get("event")
        ugcs = ((pr.get("geocode") or {}).get("UGC")) or []
        for g, spec in GROUPS.items():
            if ev in spec["events"]:
                zones[g].update(ugcs)
    out = {}
    for g, z in zones.items():
        st = {}
        for u in z:
            if u[:2] in STATE_CODES:
                st[u[:2]] = st.get(u[:2], 0) + 1
        out[g] = {"zones": len(z), "states": [s for s, _ in sorted(st.items(), key=lambda kv: -kv[1])]}
    return out


def decide(counts, state, now, max_posts=2):
    """Onset when a group crosses its threshold; follow-ups only if it grows a lot or a day passes."""
    hs = state.setdefault("hazards", {})
    due = []
    for g, c in counts.items():
        thr = GROUPS[g]["threshold"]
        rec = hs.get(g)
        n = c["zones"]
        if rec and n < thr * 0.5:
            hs.pop(g)                                      # event over (quietly)
            continue
        if n < thr:
            continue
        if not rec:
            hs[g] = {"since": now.isoformat(), "last_post": now.isoformat(), "last_count": n, "peak": n}
            due.append((n / thr, {"type": "hazard", "group": g, "phase": "onset", "zones": n,
                                  "states": c["states"], "day": 1}))
            continue
        rec["peak"] = max(rec["peak"], n)
        since_post = now - datetime.fromisoformat(rec["last_post"])
        grew = n >= rec["last_count"] * 1.5 and since_post >= timedelta(hours=6)
        daily = since_post >= timedelta(hours=24)
        if grew or daily:
            day = (now - datetime.fromisoformat(rec["since"])).days + 1
            rec.update(last_post=now.isoformat(), last_count=n)
            due.append((n / thr, {"type": "hazard", "group": g, "phase": "expanding" if grew else "ongoing",
                                  "zones": n, "states": c["states"], "day": day}))
    due.sort(key=lambda x: -x[0])
    for _, a in due[max_posts:]:                           # not posted now -> try again next run
        rec = hs[a["group"]]
        if a["phase"] == "onset":
            hs.pop(a["group"])
        else:
            rec["last_post"] = (now - timedelta(hours=24)).isoformat()
    return [a for _, a in due[:max_posts]], state


# --- Observations (METARs) -----------------------------------------------------
METAR_API = "https://aviationweather.gov/api/data/metar?bbox={s},{w},{n},{e}&format=json"
TILES = [(24, -125, 37, -103), (37, -125, 50, -103), (24, -103, 37, -88), (37, -103, 50, -88),
         (24, -88, 37, -66), (37, -88, 48, -66)]


def c_to_f(c):
    return c * 9 / 5 + 32


def wind_chill_f(t_f, mph):
    if t_f > 50 or mph < 3:
        return t_f
    return 35.74 + 0.6215 * t_f - 35.75 * mph ** 0.16 + 0.4275 * t_f * mph ** 0.16


def heat_index_f(t_f, rh):
    if t_f < 80:
        return t_f
    hi = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh - .22475541 * t_f * rh - .00683783 * t_f ** 2
          - .05481717 * rh ** 2 + .00122874 * t_f ** 2 * rh + .00085282 * t_f * rh ** 2
          - .00000199 * t_f ** 2 * rh ** 2)
    return hi


def rh_pct(t_c, td_c):
    a, b = 17.625, 243.04
    return 100 * math.exp(a * td_c / (b + td_c)) / math.exp(a * t_c / (b + t_c))


def _num(x):
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_metars(rows):
    out = {}
    for r in rows:
        try:
            t, td = r.get("temp"), r.get("dewp")
            if t is None or r.get("lat") is None:
                continue
            spd = _num(r.get("wspd")) * 1.15078
            gst = _num(r.get("wgst")) * 1.15078
            tf = c_to_f(float(t))
            rh = rh_pct(float(t), float(td)) if td is not None else None
            name = (r.get("name") or r.get("icaoId", "")).replace(", US", "")
            out[r["icaoId"]] = {
                "id": r["icaoId"], "name": name, "lat": float(r["lat"]), "lon": float(r["lon"]),
                "temp_f": round(tf), "wind_mph": round(spd), "gust_mph": round(gst),
                "wind_chill_f": round(wind_chill_f(tf, spd)),
                "heat_index_f": round(heat_index_f(tf, rh)) if rh is not None else round(tf),
                "rh": round(rh) if rh is not None else None, "wx": r.get("wxString") or ""}
        except (TypeError, ValueError, KeyError):
            continue
    return list(out.values())


def fetch_metars():
    rows = []
    for s, w, n, e in TILES:
        try:
            rows += net.get_json(METAR_API.format(s=s, w=w, n=n, e=e))
        except Exception as ex:  # noqa: BLE001
            print(f"[hazards] METAR tile failed: {ex}")
    return parse_metars(rows)


def _spread(rows, key, n, reverse, min_sep=2.5):
    """Top-n extremes, skipping stations too close to one already chosen (cleaner map)."""
    rows = sorted([r for r in rows if r.get(key) is not None], key=lambda r: r[key], reverse=reverse)
    out = []
    for r in rows:
        if all(abs(r["lat"] - o["lat"]) > min_sep * 0.6 or abs(r["lon"] - o["lon"]) > min_sep for o in out):
            out.append(r)
        if len(out) >= n:
            break
    return out


WX_LABELS = [("+SN", "heavy snow"), ("BLSN", "blowing snow"), ("FZRA", "freezing rain"), ("FZDZ", "freezing drizzle"),
             ("PL", "sleet"), ("SN", "snow"), ("DS", "dust storm"), ("SS", "sandstorm"), ("+RA", "heavy rain"),
             ("TS", "thunderstorm")]


def wx_label(wx):
    for code, lab in WX_LABELS:
        if code in wx:
            return lab
    return ""


def extremes(obs, kind, zone_bbox=None):
    """Stations + one-line facts for the hazard type."""
    if zone_bbox:
        w, e, s, n = zone_bbox
        local = [o for o in obs if w <= o["lon"] <= e and s <= o["lat"] <= n] or obs
    else:
        local = obs
    if kind == "cold":
        pts = _spread(local, "wind_chill_f", 10, False)
        lab = [(p, f"{p['wind_chill_f']}°") for p in pts]
        facts = {"lowest_wind_chills": [f"{p['wind_chill_f']}F {p['name']}" for p in pts[:5]],
                 "coldest_temps": [f"{p['temp_f']}F {p['name']}" for p in _spread(local, 'temp_f', 3, False)]}
    elif kind == "heat":
        pts = _spread(local, "heat_index_f", 10, True)
        lab = [(p, f"{p['heat_index_f']}°") for p in pts]
        facts = {"highest_heat_index": [f"{p['heat_index_f']}F {p['name']}" for p in pts[:5]],
                 "hottest_temps": [f"{p['temp_f']}F {p['name']}" for p in _spread(local, 'temp_f', 3, True)]}
    elif kind in ("wind", "fire"):
        pts = _spread(local, "gust_mph", 10, True)
        lab = [(p, f"G{p['gust_mph']}") for p in pts if p["gust_mph"]]
        facts = {"peak_gusts": [f"{p['gust_mph']} mph {p['name']}" for p in pts[:5] if p["gust_mph"]]}
        if kind == "fire":
            dry = _spread([o for o in local if o["rh"] is not None], "rh", 4, False)
            facts["lowest_humidity"] = [f"{p['rh']}% {p['name']}" for p in dry]
    elif kind == "winter":
        wx = [o for o in local if wx_label(o["wx"]) in ("heavy snow", "blowing snow", "freezing rain",
                                                         "freezing drizzle", "sleet", "snow")]
        pts = _spread(wx, "temp_f", 12, False, min_sep=1.8)
        lab = [(p, f"{p['temp_f']}° {wx_label(p['wx'])}") for p in pts]
        facts = {"stations_reporting_snow_or_ice": len(wx),
                 "notable_obs": [f"{p['name']}: {wx_label(p['wx'])}, {p['temp_f']}F, gust {p['gust_mph']} mph"
                                 for p in pts[:5]]}
    else:
        lab, facts = [], {}
    return lab, facts


# --- Fronts (WPC coded surface bulletin) -------------------------------------
CODSUS_URL = "https://tgftp.nws.noaa.gov/data/raw/as/asus02.kwbc.cod.sus.txt"
_VALID = re.compile(r"VALID\s+(\d{2})(\d{2})(\d{2})Z")


def _pt(tok):
    if len(tok) == 7:
        return -int(tok[3:]) / 10.0, int(tok[:3]) / 10.0
    if len(tok) == 5:                                    # low-res bulletin
        return -float(tok[2:]), float(tok[:2])
    return None


def parse_codsus(text, now=None):
    now = now or datetime.now(timezone.utc)
    m = _VALID.search(text)
    valid = None
    if m:
        mo, dd, hh = (int(x) for x in m.groups())
        year = now.year - (1 if mo > now.month + 1 else 0)
        valid = datetime(year, mo, dd, hh, tzinfo=timezone.utc)
    body = text[m.end():] if m else text
    tokens = body.replace("\n", " ").split()
    feats, cur = [], None
    for tok in tokens:
        if tok in ("HIGHS", "LOWS", "COLD", "WARM", "STNRY", "OCFNT", "TROF"):
            cur = {"kind": tok, "items": []}
            feats.append(cur)
        elif tok == "$$":
            break
        elif cur is not None:
            cur["items"].append(tok)
    centers, fronts = [], []
    for f in feats:
        if f["kind"] in ("HIGHS", "LOWS"):
            it = f["items"]
            for i in range(0, len(it) - 1, 2):
                p = _pt(it[i + 1])
                if p and it[i].isdigit():
                    centers.append({"kind": "H" if f["kind"] == "HIGHS" else "L", "mb": int(it[i]),
                                    "lon": p[0], "lat": p[1]})
        else:
            pts = [p for p in (_pt(t) for t in f["items"] if t.isdigit()) if p]
            if len(pts) >= 2:
                fronts.append({"kind": f["kind"], "pts": pts})
    return {"valid": valid, "centers": centers, "fronts": fronts}


def fetch_fronts(now=None):
    try:
        d = parse_codsus(net.get(CODSUS_URL).text, now)
        if d["valid"] and (now or datetime.now(timezone.utc)) - d["valid"] <= timedelta(hours=9):
            return d
        print("[hazards] surface bulletin is stale; skipping fronts")
    except Exception as e:  # noqa: BLE001
        print(f"[hazards] fronts unavailable: {e}")
    return None


def fronts_summary(fr):
    if not fr:
        return None
    lows = sorted([c for c in fr["centers"] if c["kind"] == "L" and 24 <= c["lat"] <= 52 and -126 <= c["lon"] <= -66],
                  key=lambda c: c["mb"])[:2]
    out = []
    for c in lows:
        place, dist = geo.nearest_place(c["lat"], c["lon"], min_pop=400000)
        out.append(f"{c['mb']} mb low {'near' if dist and dist < 80 else f'{dist} mi from'} {place}")
    return out


# --- Zone shapes (IEM) -----------------------------------------------------------
ZONES_URL = "https://mesonet.agron.iastate.edu/api/1/vtec/county_zone.geojson"


def fetch_zone_shapes(vtec_codes):
    data = net.get_json(ZONES_URL)
    rings = []
    for f in data.get("features", []):
        if (f.get("properties") or {}).get("ph_sig") in vtec_codes:
            g = f.get("geometry") or {}
            polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", [])
            rings += [p[0] for p in polys]
    return rings


# --- Map --------------------------------------------------------------------------
FRONT_STYLE = {"COLD": "#2f6bff", "WARM": "#ff3030", "STNRY": "#b05cff", "OCFNT": "#c040c0", "TROF": "#d8a050"}


def render_map(out_path, group, rings, labels, fronts, now, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    from ..render import local_time_label
    spec = GROUPS[group]
    proj = geo.Lambert(lon0=-96, lat0=38, lat1=33, lat2=45)
    fig = plt.figure(figsize=(16, 10), dpi=100, facecolor="#0e141e")
    ax = fig.add_axes([0, 0.04, 1, 0.885])
    ax.set_facecolor("#121a26")
    ax.axis("off")
    xs, ys = proj([-121, -71, -125, -66, -96, -96], [24, 24, 47, 47, 50, 23])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    want = 16 / (10 * 0.885)
    if (x1 - x0) / (y1 - y0) < want:
        c = (x0 + x1) / 2; w = (y1 - y0) * want; x0, x1 = c - w / 2, c + w / 2   # noqa: E702
    else:
        c = (y0 + y1) / 2; h = (x1 - x0) / want; y0, y1 = c - h / 2, c + h / 2   # noqa: E702
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)   # noqa: E702
    polys = []
    for r in rings:
        a = np.array(r)
        x, y = proj(a[:, 0], a[:, 1])
        polys.append(np.column_stack([x, y]))
    if polys:
        ax.add_collection(PolyCollection(polys, facecolors=spec["color"], edgecolors=spec["color"],
                                         alpha=0.55, linewidths=0.3, zorder=2))
    geo.draw_basemap(ax, proj, color="#9aa8ba", lw=0.8)
    pe = [patheffects.withStroke(linewidth=3, foreground="black")]
    if fronts:
        for f in fronts["fronts"]:
            a = np.array(f["pts"])
            x, y = proj(a[:, 0], a[:, 1])
            ax.plot(x, y, color=FRONT_STYLE[f["kind"]], lw=2.6 if f["kind"] != "TROF" else 1.6,
                    ls="--" if f["kind"] == "TROF" else "-", zorder=6)
        for c in fronts["centers"]:
            if -128 <= c["lon"] <= -62 and 22 <= c["lat"] <= 52:
                x, y = proj(c["lon"], c["lat"])
                col = "#3a8bff" if c["kind"] == "H" else "#ff3b3b"
                ax.text(x, y, c["kind"], color=col, fontsize=26, weight="bold", ha="center", va="center",
                        zorder=8, path_effects=pe)
                ax.text(x, y - (y1 - y0) * 0.03, str(c["mb"]), color="white", fontsize=10, ha="center",
                        va="top", zorder=8, path_effects=pe)
    for st, text in labels:
        x, y = proj(st["lon"], st["lat"])
        ax.scatter([x], [y], s=16, c="white", edgecolors="black", zorder=9)
        ax.text(x + (x1 - x0) * 0.004, y, text, color="#ffe9a8", fontsize=12, weight="bold", zorder=9,
                va="bottom", path_effects=pe)
    fig.text(0.015, 0.965, f"{spec['label']}{title_extra}", color="white", fontsize=24, weight="bold", va="center")
    fig.text(0.985, 0.965, local_time_label(now), color="#ffd25a", fontsize=16, weight="bold", va="center", ha="right")
    fig.text(0.015, 0.02, "Warnings: NWS · Obs: METARs (aviationweather.gov data) · Fronts: WPC surface analysis",
             color="#aab4c3", fontsize=11, va="center")
    fig.text(0.985, 0.02, config.HANDLE or config.BRAND, color="#aab4c3", fontsize=11, va="center", ha="right")
    handles = [Line2D([], [], color=spec["color"], lw=8, alpha=0.7, label=spec["label"])]
    if fronts:
        handles += [Line2D([], [], color=FRONT_STYLE[k], lw=2.5, label=n) for k, n in
                    (("COLD", "Cold front"), ("WARM", "Warm front"), ("STNRY", "Stationary"), ("OCFNT", "Occluded"))]
    leg = ax.legend(handles=handles, loc="lower right", ncol=len(handles), fontsize=11, frameon=True)
    leg.get_frame().set_facecolor("#0e141e"); leg.get_frame().set_edgecolor("none")   # noqa: E702
    for t in leg.get_texts():
        t.set_color("white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def zone_bbox(rings):
    if not rings:
        return None
    pts = np.array([p for r in rings for p in r])
    return [pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()]


def build(out_path, action, now=None, rings=None, obs=None, fronts="fetch"):
    now = now or datetime.now(timezone.utc)
    g = action["group"]
    spec = GROUPS[g]
    rings = rings if rings is not None else fetch_zone_shapes(spec["vtec"])
    obs = obs if obs is not None else (fetch_metars() if spec["obs"] else [])
    fronts = fetch_fronts(now) if fronts == "fetch" else fronts
    labels, ofacts = extremes(obs, spec["obs"], zone_bbox(rings)) if spec["obs"] else ([], {})
    extra = f" · Day {action['day']}" if action.get("day", 1) > 1 else ""
    render_map(out_path, g, rings, labels, fronts, now, extra)
    return {"hazard": spec["label"], "phase": action["phase"], "day": action.get("day", 1),
            "zones_under_warning": action["zones"], "states": action["states"][:10], **ofacts,
            "surface_lows": fronts_summary(fronts),
            "hashtags": geo.state_hashtags(action["states"]) + {"heat": ["#heatwave"], "cold": ["#arctic"],
                                                                 "fire": ["#firewx"]}.get(g, [])}
