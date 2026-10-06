"""Severe weather: live warnings, clustering, radar loops with warning polygons, storm reports."""
import re
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import config, geo, net, radar, render

PHEN = {"Tornado Warning": "TO", "Severe Thunderstorm Warning": "SV", "Flash Flood Warning": "FF"}
COLORS = {"TO": "#ff2a2a", "SV": "#ffd000", "FF": "#2bff77", "HIGH": "#ff3df5"}
LSR_STYLE = {  # IEM LSR type code -> (label, marker, color)
    "T": ("Tornado", "v", "#ff2a2a"), "H": ("Hail", "o", "#3ddc5a"),
    "G": ("Wind", "s", "#3aa0ff"), "D": ("Wind", "s", "#3aa0ff"), "M": ("Wind", "s", "#3aa0ff"),
}
_STATE_RE = re.compile(r"\[([A-Z]{2})\]|,\s*([A-Z]{2})\b")


def _iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _rings(geom):
    if not geom:
        return []
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [p[0] for p in geom["coordinates"]]
    return []


def _states(text):
    return sorted({a or b for a, b in _STATE_RE.findall(text or "")})


def _vtec_key(props):
    v = (props.get("parameters") or {}).get("VTEC") or []
    if v:
        parts = v[0].strip("/").split(".")
        if len(parts) >= 6:
            return f"{parts[2]}.{parts[3]}.{parts[5]}"     # office.phenomena.event-number
    return props.get("id", "")


def is_high_end(w):
    p = w["params"]
    return (p.get("tornadoDamageThreat") in ("CONSIDERABLE", "CATASTROPHIC") or
            p.get("flashFloodDamageThreat") == "CATASTROPHIC" or
            p.get("thunderstormDamageThreat") == "DESTRUCTIVE")


def high_end_label(w):
    p = w["params"]
    if p.get("tornadoDamageThreat") == "CATASTROPHIC":
        return "TORNADO EMERGENCY"
    if p.get("flashFloodDamageThreat") == "CATASTROPHIC":
        return "FLASH FLOOD EMERGENCY"
    if p.get("tornadoDamageThreat") == "CONSIDERABLE":
        return "PDS TORNADO WARNING"
    if p.get("thunderstormDamageThreat") == "DESTRUCTIVE":
        return "DESTRUCTIVE SEVERE T-STORM WARNING"
    return w["event"].upper()


# --- Live warnings (NWS API) ------------------------------------------------
def parse_live(alerts_json):
    out = []
    for f in alerts_json.get("features", []):
        pr = f.get("properties") or {}
        ph = PHEN.get(pr.get("event"))
        rings = _rings(f.get("geometry"))
        if not ph or not rings:
            continue
        pts = [pt for r in rings for pt in r]
        params = {k: (v[0] if isinstance(v, list) and v else v) for k, v in (pr.get("parameters") or {}).items()}
        out.append({
            "key": _vtec_key(pr), "event": pr["event"], "ph": ph, "rings": rings,
            "center": (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))),
            "issued": pr.get("sent") or pr.get("effective"), "expires": pr.get("expires") or pr.get("ends"),
            "area": pr.get("areaDesc", ""), "states": _states(pr.get("areaDesc", "")),
            "headline": (pr.get("parameters") or {}).get("NWSheadline", [pr.get("headline", "")])[0],
            "description": (pr.get("description") or "")[:900], "params": params,
            "sender": pr.get("senderName", ""),
        })
    return out


def fetch_live():
    return parse_live(net.get_json(config.NWS_ALERTS))


def clusters(warnings, link_deg=None):
    """Single-linkage clustering of warning centers (union-find)."""
    link = link_deg or config.CLUSTER_LINK_DEG
    n = len(warnings)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            (x1, y1), (x2, y2) = warnings[i]["center"], warnings[j]["center"]
            if abs(x1 - x2) <= link * 1.3 and abs(y1 - y2) <= link:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(warnings[i])
    out = []
    for members in groups.values():
        pts = [pt for w in members for r in w["rings"] for pt in r]
        counts = {ph: sum(1 for w in members if w["ph"] == ph) for ph in ("TO", "SV", "FF")}
        states = sorted({s for w in members for s in w["states"]})
        out.append({"members": members, "bbox": geo.bbox_for(pts), "counts": counts,
                    "total": len(members), "states": states,
                    "keys": sorted(w["key"] for w in members)})
    return sorted(out, key=lambda c: (-c["counts"]["TO"], -c["total"]))


def meets_event_threshold(c):
    return c["counts"]["TO"] >= config.EVENT_TOR_COUNT or c["total"] >= config.EVENT_ANY_COUNT


# --- Historical polygons + storm reports (IEM) ------------------------------
def fetch_sbw(start, end):
    data = net.get_json(f"{config.IEM_SBW}?begints={start:%Y-%m-%dT%H:%MZ}&endts={end:%Y-%m-%dT%H:%MZ}"
                        "&ph=TO&ph=SV&ph=FF&only_new=false")
    out = []
    for f in data.get("features", []):
        p = f["properties"]
        if p.get("significance") != "W":
            continue
        out.append({"ph": p.get("phenomena"), "rings": _rings(f.get("geometry")),
                    "begin": _iso(p.get("utc_polygon_begin") or p.get("utc_issue")),
                    "end": _iso(p.get("utc_polygon_end") or p.get("utc_expire")),
                    "status": p.get("status"), "key": f"{p.get('wfo')}.{p.get('phenomena')}.{p.get('eventid')}"})
    return out


def fetch_lsr(start, end, bbox):
    data = net.get_json(f"{config.IEM_LSR}?sts={start:%Y%m%d%H%M}&ets={end:%Y%m%d%H%M}")
    w, e, s, n = bbox
    out = []
    for f in data.get("features", []):
        p = f["properties"]
        lon, lat = float(p.get("lon")), float(p.get("lat"))
        if p.get("type") in LSR_STYLE and w <= lon <= e and s <= lat <= n:
            out.append({"type": p["type"], "lat": lat, "lon": lon, "time": _iso(p.get("valid")),
                        "mag": p.get("magnitude"), "city": p.get("city"), "state": p.get("state"),
                        "remark": (p.get("remark") or "")[:200]})
    return out


def in_bbox(w, bbox):
    x, y = w.get("center") or (np.mean([p[0] for r in w["rings"] for p in r]),
                               np.mean([p[1] for r in w["rings"] for p in r]))
    return bbox[0] <= x <= bbox[1] and bbox[2] <= y <= bbox[3]


def report_stats(lsrs):
    tor = sum(1 for r in lsrs if r["type"] == "T")
    hail = [r for r in lsrs if r["type"] == "H"]
    wind = [r for r in lsrs if r["type"] in ("G", "D", "M")]

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0
    big_hail = max(hail, key=lambda r: num(r["mag"]), default=None)
    top_gust = max((r for r in wind if r["type"] in ("G", "M")), key=lambda r: num(r["mag"]), default=None)
    return {"tornado_reports": tor, "hail_reports": len(hail), "wind_reports": len(wind),
            "largest_hail": f'{big_hail["mag"]}" at {big_hail["city"]}, {big_hail["state"]}' if big_hail and num(big_hail["mag"]) else None,
            "peak_gust": f'{top_gust["mag"]} mph at {top_gust["city"]}, {top_gust["state"]}' if top_gust and num(top_gust["mag"]) else None}


# --- Rendering --------------------------------------------------------------
def render_frame(t, bbox, radar_rgba, radar_ext, polys, lsrs, title, counters, places_n=10):
    """One 1280x720 frame. polys: list of (ph_or_HIGH, rings). lsrs: reports to show."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from PIL import Image

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0b1018")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#121a26")
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])   # noqa: E702
    ax.axis("off")
    if radar_rgba is not None:
        ax.imshow(radar_rgba, extent=radar_ext, origin="upper", interpolation="nearest", zorder=2,
                  aspect="auto")
    geo.draw_basemap(ax, geo.PlateCarree(), color="#a9b6c6", lw=0.9)
    for ph, rings in polys:
        for r in rings:
            a = np.array(r)
            ax.plot(a[:, 0], a[:, 1], color="black", lw=4.2 if ph == "HIGH" else 3.0, zorder=7)
            ax.plot(a[:, 0], a[:, 1], color=COLORS.get(ph, "#fff"), lw=2.6 if ph == "HIGH" else 1.7, zorder=8)
    for r in lsrs:
        _, mk, col = LSR_STYLE[r["type"]]
        ax.scatter([r["lon"]], [r["lat"]], marker=mk, s=70 if r["type"] == "T" else 32, c=col,
                   edgecolors="black", linewidths=0.8, zorder=9)
    w, e, s, n = bbox
    shown, used = 0, {}
    pe = [patheffects.withStroke(linewidth=2.5, foreground="black")]
    for name, lat, lon, pop, adm in geo.places(75000):
        if shown >= places_n:
            break
        crowded = any(abs(lon - ux) < (e - w) * 0.07 and abs(lat - uy) < (n - s) * 0.05 for ux, uy in used.values())
        if name not in used and not crowded and w + 0.2 < lon < e - 0.6 and s + (n - s) * 0.1 < lat < n - (n - s) * 0.14:
            ax.scatter([lon], [lat], s=9, c="white", zorder=10)
            ax.text(lon + (e - w) * 0.004, lat, name, color="white", fontsize=9, zorder=10, va="bottom",
                    path_effects=pe)
            used[name] = (lon, lat)
            shown += 1
    # overlay (header, footer, counters, legend) drawn above the map
    ov = fig.add_axes([0, 0, 1, 1], zorder=50)
    ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off"); ov.patch.set_alpha(0)   # noqa: E702
    ov.add_patch(plt.Rectangle((0, 0.925), 1, 0.075, color="#0b1018", alpha=0.88))
    ov.add_patch(plt.Rectangle((0, 0), 1, 0.04, color="#0b1018", alpha=0.88))
    ov.text(0.012, 0.962, title, color="white", fontsize=17, weight="bold", va="center")
    ov.text(0.988, 0.962, render.local_time_label(t), color="#ffd25a", fontsize=13, weight="bold",
            va="center", ha="right")
    ov.text(0.012, 0.02, "Radar: NEXRAD composite via Iowa Environmental Mesonet · Warnings & reports: NWS",
            color="#aab4c3", fontsize=8.5, va="center")
    ov.text(0.988, 0.02, config.HANDLE or config.BRAND, color="#aab4c3", fontsize=8.5, va="center", ha="right")
    if counters:
        ov.text(0.012, 0.895, "   ".join(counters), color="white", fontsize=11, weight="bold", va="center",
                bbox=dict(facecolor="#0b1018", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.35"))
    # legend (lower right, above the footer)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS["TO"], lw=3, label="Tornado warning"),
               Line2D([], [], color=COLORS["SV"], lw=3, label="Severe T-storm"),
               Line2D([], [], color=COLORS["FF"], lw=3, label="Flash flood")]
    if any(ph == "HIGH" for ph, _ in polys):
        handles.insert(0, Line2D([], [], color=COLORS["HIGH"], lw=3, label="PDS / Emergency"))
    if lsrs:
        handles += [Line2D([], [], marker=m, ls="", color=c, mec="black", ms=7, label=l)
                    for l, m, c in [("Tornado rpt", "v", "#ff2a2a"), ("Hail rpt", "o", "#3ddc5a"),
                                    ("Wind rpt", "s", "#3aa0ff")]]
    leg = ov.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.995, 0.048), ncol=len(handles),
                    fontsize=8.5, frameon=True, handlelength=1.6, columnspacing=1.1, borderpad=0.5)
    leg.get_frame().set_facecolor("#0b1018"); leg.get_frame().set_alpha(0.85); leg.get_frame().set_edgecolor("none")  # noqa: E702
    for txt in leg.get_texts():
        txt.set_color("white")
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img


def build_live_loop(out_path, warnings, bbox, title, minutes=None, now=None, highlight_keys=(),
                    radar_fetch=None):
    """Loop of the last N minutes of radar with the CURRENT warning polygons on every frame."""
    now = now or datetime.now(timezone.utc)
    radar_fetch = radar_fetch or radar.fetch
    minutes = minutes or config.LOOP_MINUTES
    times = radar.frame_times(now - timedelta(minutes=minutes), now, count=minutes // 5 + 1)
    inside = [w for w in warnings if in_bbox(w, bbox)]
    polys = [("HIGH" if w["key"] in highlight_keys else w["ph"], w["rings"]) for w in inside]
    counts = {ph: sum(1 for w in inside if w["ph"] == ph) for ph in ("TO", "SV", "FF")}
    counters = [f"{counts['TO']} Tornado", f"{counts['SV']} Severe T-storm", f"{counts['FF']} Flash Flood",
                "warnings active"]
    frames = []
    for t in times:
        tt, arr, ext = radar_fetch(t, bbox)
        if arr is None:
            continue
        frames.append(render_frame(tt, bbox, arr, ext, polys, [], title, counters))
    if len(frames) < 3:
        raise RuntimeError("not enough radar frames for a loop")
    render.frames_to_mp4(frames, out_path, fps=7, hold_last_sec=1.2, loops=2)
    return {"warnings_in_view": counts, "loop_minutes": minutes}


def build_summary(out_path, start, end, bbox, title, sbw=None, lsrs=None, radar_fetch=None):
    """End-of-event time lapse: radar + warnings active at each moment + reports accumulating."""
    radar_fetch = radar_fetch or radar.fetch
    sbw = sbw if sbw is not None else fetch_sbw(start - timedelta(hours=1), end)
    lsrs = lsrs if lsrs is not None else fetch_lsr(start, end, bbox)
    sbw = [w for w in sbw if w["rings"] and in_bbox(w, bbox)]
    times = radar.frame_times(start, end, config.SUMMARY_FRAMES)
    frames = []
    seen = {"TO": set(), "SV": set(), "FF": set()}
    for t in times:
        active = [w for w in sbw if w["begin"] and w["end"] and w["begin"] <= t < w["end"]]
        for w in active:
            seen.setdefault(w["ph"], set()).add(w["key"])
        shown = [r for r in lsrs if r["time"] and r["time"] <= t]
        st = report_stats(shown)
        counters = [f"Tornado warnings: {len(seen['TO'])}",
                    f"Reports — Tornado {st['tornado_reports']} · Hail {st['hail_reports']} · Wind {st['wind_reports']}"]
        tt, arr, ext = radar_fetch(t, bbox, back_steps=1)
        frames.append(render_frame(tt, bbox, arr, ext, [(w["ph"], w["rings"]) for w in active], shown,
                                   title, counters))
    if len(frames) < 6:
        raise RuntimeError("not enough frames for a summary")
    # final frame: every polygon of the event + all reports
    allp = [(w["ph"], w["rings"]) for w in sbw if w["status"] in (None, "NEW")]
    st = report_stats(lsrs)
    counters = [f"Tornado warnings: {len({w['key'] for w in sbw if w['ph'] == 'TO'})}",
                f"Reports — Tornado {st['tornado_reports']} · Hail {st['hail_reports']} · Wind {st['wind_reports']}"]
    frames.append(render_frame(end, bbox, None, None, allp, lsrs, title + " · Full event", counters))
    render.frames_to_mp4(frames, out_path, fps=10, hold_last_sec=3.0, loops=1)
    keys = {ph: len({w["key"] for w in sbw if w["ph"] == ph}) for ph in ("TO", "SV", "FF")}
    return {"window": f"{render.local_time_label(start)} to {render.local_time_label(end)}",
            "warnings_issued": {"tornado": keys["TO"], "severe_tstorm": keys["SV"], "flash_flood": keys["FF"]},
            **st}
