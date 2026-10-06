"""Tropical cyclones: NHC active storms, trigger logic, GOES satellite loops, NHC cone graphic."""
import io
import re
from datetime import datetime, timedelta, timezone

from PIL import Image

from .. import config, geo, net, render

CLASS_NAMES = {"TD": "Tropical Depression", "STD": "Subtropical Depression", "TS": "Tropical Storm",
               "STS": "Subtropical Storm", "HU": "Hurricane", "TY": "Typhoon", "PTC": "Potential Tropical Cyclone",
               "PC": "Post-Tropical Cyclone"}
_FRAME = re.compile(r"(\d{4})(\d{3})(\d{2})(\d{2})_GOES(\d\d)-ABI-([A-Za-z0-9-]+)-GEOCOLOR-(\d+)x(\d+)\.jpg")


def category(kt):
    """0 = tropical depression, 0.5 = tropical storm, 1-5 = Saffir-Simpson."""
    kt = int(kt or 0)
    for cat, lo in ((5, 137), (4, 113), (3, 96), (2, 83), (1, 64)):
        if kt >= lo:
            return cat
    return 0.5 if kt >= 34 else 0


def label(storm):
    cat = category(storm["kt"])
    if cat >= 1:
        return f"{'Major ' if cat >= 3 else ''}Hurricane {storm['name']} (Category {cat})"
    return f"{CLASS_NAMES.get(storm['cls'], storm['cls'])} {storm['name']}"


def parse(nhc_json):
    out = []
    for s in nhc_json.get("activeStorms", []):
        out.append({"id": s["id"].upper(), "bin": s.get("binNumber", ""), "name": s.get("name", "?"),
                    "cls": s.get("classification", ""), "kt": int(s.get("intensity") or 0),
                    "mb": int(s.get("pressure") or 0), "lat": float(s.get("latitudeNumeric") or 0),
                    "lon": float(s.get("longitudeNumeric") or 0), "move_dir": s.get("movementDir"),
                    "move_mph": s.get("movementSpeed"), "updated": s.get("lastUpdate"),
                    "adv": (s.get("publicAdvisory") or {}).get("advNum")})
    return out


def fetch():
    return parse(net.get_json(config.NHC_STORMS))


def near_us(storm, miles=350):
    us = [p for p in geo.places(60000) if p[4] in geo.US_STATES]
    d = min(geo.haversine_mi(storm["lat"], storm["lon"], p[1], p[2]) for p in us)
    return d <= miles, round(d)


def storm_facts(storm):
    near, dist = near_us(storm)
    place, pdist = geo.nearest_place(storm["lat"], storm["lon"], min_pop=60000)
    return {"storm": label(storm), "max_wind_mph": round(storm["kt"] * 1.15078 / 5) * 5,
            "max_wind_kt": storm["kt"], "pressure_mb": storm["mb"],
            "position": f"{abs(storm['lat']):.1f}{'N' if storm['lat'] >= 0 else 'S'} "
                        f"{abs(storm['lon']):.1f}{'W' if storm['lon'] < 0 else 'E'}",
            "nearest_city": f"{pdist} mi from {place}" if place else None,
            "movement": f"{storm['move_dir']} deg at {storm['move_mph']} mph" if storm["move_mph"] else None,
            "miles_to_nearest_us_city": dist, "advisory": storm["adv"],
            "hashtags": [f"#{storm['name']}", "#tropics" if category(storm["kt"]) < 1 else "#hurricane"]}


# --- Trigger logic (pure; unit-tested) --------------------------------------
def decide(storms, state, now):
    """Returns (actions, new_state). state['storms'][id] keeps history + what was posted."""
    st = state.setdefault("storms", {})
    actions = []
    min_cat = {"TD": 0, "TS": 0.5, "HU": 1}.get(config.TROPICAL_NEW_MIN, 0.5)
    for s in storms:
        rec = st.setdefault(s["id"], {"history": [], "posted_new": False, "posted_cat": 0,
                                      "ri_last": None, "near_us_adv": None})
        rec["history"] = [h for h in rec["history"]
                          if now - datetime.fromisoformat(h[0]) <= timedelta(hours=30)]
        if not rec["history"] or rec["history"][-1][1] != s["kt"] or rec["history"][-1][2] != s["adv"]:
            rec["history"].append([now.isoformat(), s["kt"], s["adv"]])
        rec["name"] = s["name"]
        rec["max_kt"] = max(rec.get("max_kt", 0), s["kt"])
        rec["last_seen"] = now.isoformat()
        cat = category(s["kt"])
        reasons = []
        if not rec["posted_new"] and cat >= min_cat:
            reasons.append("new_storm")
            rec["posted_new"] = True
            rec["posted_cat"] = max(rec["posted_cat"], cat)
        if cat >= 1 and cat > rec["posted_cat"]:
            reasons.append("category_up")
            rec["posted_cat"] = cat
        window = [h[1] for h in rec["history"] if now - datetime.fromisoformat(h[0]) <= timedelta(hours=24, minutes=30)]
        gain = s["kt"] - min(window) if window else 0
        ri_ok = not rec["ri_last"] or now - datetime.fromisoformat(rec["ri_last"]) >= timedelta(hours=18)
        if gain >= config.RI_KT_24H and ri_ok:
            reasons.append("rapid_intensification")
            rec["ri_last"] = now.isoformat()
        near, _ = near_us(s)
        if near and cat >= 3 and rec["near_us_adv"] != s["adv"] and _adv_is_6h(s["adv"]):
            reasons.append("major_hurricane_near_us")
            rec["near_us_adv"] = s["adv"]
        if reasons:
            actions.append({"type": "tropical", "storm": s, "reasons": reasons, "kt_gain_24h": gain})
    active = {s["id"] for s in storms}
    for sid in list(st):
        if sid not in active:
            rec = st.pop(sid)
            # storm is gone (dissipated / post-tropical): post its life story if it was a big one
            if rec.get("max_kt", 0) >= config.RECAP_MIN_KT:
                actions.append({"type": "storm_recap", "id": sid, "name": rec.get("name", ""),
                                "max_kt": rec["max_kt"]})
    return actions, state


def _adv_is_6h(adv):
    """Only full advisories (1, 2, 3...), not intermediate '2A' ones, to space updates ~6h apart."""
    return bool(adv) and str(adv).isdigit()


# --- Media -------------------------------------------------------------------
def _sector_candidates(storm):
    lat, lon, basin = storm["lat"], storm["lon"], storm["id"][:2]
    if basin == "AL":
        out = []
        if 17 <= lat <= 31.5 and -98.5 <= lon <= -79:
            out.append(("GOES19", "gm"))
        if 9 <= lat <= 23 and -89 <= lon <= -59:
            out.append(("GOES19", "car"))
        if 22 <= lat <= 42 and -85 <= lon <= -66:
            out.append(("GOES19", "se"))
        return out + [("GOES19", "taw")]
    return [("GOES18", "mex"), ("GOES18", "tpw")] if lon > -120 else [("GOES18", "tpw")]


def list_frames(html, count, target_w=1000):
    found = {}
    for m in _FRAME.finditer(html):
        found.setdefault((int(m.group(7)), int(m.group(8))), set()).add(m.group(0))
    if not found:
        return []
    size = min(found, key=lambda wh: abs(wh[0] - target_w))
    return sorted(found[size])[-count:]


def frame_time(name):
    m = _FRAME.search(name)
    y, doy, hh, mm = (int(m.group(i)) for i in range(1, 5))
    return datetime(y, 1, 1, hh, mm, tzinfo=timezone.utc) + timedelta(days=doy - 1)


def satellite_frames(storm, count=None):
    count = count or config.TROPICAL_SATELLITE_FRAMES
    urls = [config.GOES_FLOATER.format(ID=storm["id"])] + [
        config.GOES_SECTOR.format(sat=sat, sector=sec) for sat, sec in _sector_candidates(storm)]
    for base in urls:
        try:
            names = list_frames(net.get(base).text, count)
        except Exception:  # noqa: BLE001
            continue
        if len(names) < 8:
            continue
        frames = []
        for n in names:
            try:
                frames.append((frame_time(n), Image.open(io.BytesIO(net.get(base + n).content))))
            except Exception:  # noqa: BLE001
                pass
        if len(frames) >= 8:
            return frames, base
    return [], None


def build_loop(out_path, storm, frames=None):
    if frames is None:
        frames, _ = satellite_frames(storm)
    if not frames:
        return None
    title = label(storm)
    branded = [render.brand(img, title, render.local_time_label(t), max_w=1200,
                            credit="Imagery: NOAA / NESDIS / STAR · GOES GeoColor") for t, img in frames]
    render.frames_to_mp4(branded, out_path, fps=10, hold_last_sec=1.5, loops=2)
    return out_path


def fetch_cone(storm, out_path):
    try:
        sid = storm["id"]                       # e.g. AL092026 -> folder AT09
        folder = ("AT" if sid.startswith("AL") else sid[:2]) + sid[2:4]
        url = config.NHC_CONE.format(bin=folder, ID=sid)
        img = Image.open(io.BytesIO(net.get(url).content))
        img.convert("RGB").save(out_path, "PNG")
        return out_path
    except Exception as e:  # noqa: BLE001
        print(f"[tropical] cone graphic unavailable: {e}")
        return None
