"""Command line entry point.

    python -m studio watch               # check everything; post only if something big is happening
    python -m studio watch --plan-only   # just report what would be posted (no state change)
    python -m studio upper-air           # 500/250/850 mb analysis + anomalies (12Z always, 00Z if notable)
    python -m studio replay --start 2025-05-16T17:00 --end 2025-05-17T04:00 [--bbox W,E,S,N] [--title ...]
                                      # build an event-summary time lapse for any past event
    python -m studio hazard --group cold # draw/post a national hazard analysis map right now
    python -m studio storm-anim --name KATRINA --year 2005   # storm-following IR animation, any storm
    python -m studio storm-anim --id EP182026 --live        # a current storm, from the live best track
    python -m studio anniversary         # "N years ago today" animation for landmark storms (daily)
    python -m studio status              # show tracked events and storms

Add --dry-run (or DRY_RUN=1) to build media + text into ./out without posting.
"""
import argparse
import json
import sys
import traceback
from datetime import datetime, timezone

from . import captions, config, engine, geo


# --- state ------------------------------------------------------------------
def load_state():
    try:
        return json.loads(config.STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.STATE_FILE.write_text(json.dumps(state, indent=1, default=str))


# --- posting ----------------------------------------------------------------
class Poster:
    def __init__(self, dry):
        self.dry = dry
        self.n = 0

    def post(self, kind, media, facts, reply_to=None):
        """media: list of file paths (1 video, or up to 4 images). Returns post id (or 'dry-N')."""
        text = captions.write(kind, facts)
        self.n += 1
        tag = f"{self.n:02d}_{kind}"
        (config.OUT_DIR / f"{tag}.txt").write_text(
            f"{text}\n\n--- media: {[str(m) for m in media]}\n--- reply_to: {reply_to}\n--- facts:\n"
            f"{json.dumps(facts, indent=1, default=str)}\n")
        print(f"\n[{kind}] ({len(text)} chars) {text}\n  media={[m.name for m in media]} reply_to={reply_to}")
        if self.dry:
            return f"dry-{self.n}"
        from . import x_client
        ids = [x_client.upload_media(m) for m in media]
        pid = x_client.create_post(text, ids, reply_to=reply_to)
        print(f"  posted id {pid}")
        return pid


# --- actions -----------------------------------------------------------------
def act_quick_tag(a, warnings, poster):
    from .products import severe
    w = a["warning"]
    pts = [p for r in w["rings"] for p in r]
    bbox = geo.bbox_for(pts, min_w=2.8, pad=0.6)
    lon, lat = w["center"]
    near, dist = geo.nearest_place(lat, lon, min_pop=20000)
    lbl = severe.high_end_label(w)
    out = config.OUT_DIR / f"quick_{w['key'].replace('.', '_')}.mp4"
    severe.build_live_loop(out, warnings, bbox, f"{lbl} · {near}", minutes=60, highlight_keys={w["key"]})
    p = w["params"]
    exp = datetime.fromisoformat(w["expires"]) if w.get("expires") else None
    from .render import local_time_label
    facts = {"label": lbl, "near": f"{near}" if dist and dist < 15 else f"{dist} mi from {near}",
             "area": w["area"][:200], "issued_by": w["sender"], "headline": w["headline"],
             "detection": p.get("tornadoDetection") and f"Tornado {p['tornadoDetection'].lower()}",
             "tornado_damage_threat": p.get("tornadoDamageThreat"), "max_hail_in": p.get("maxHailSize"),
             "max_wind": p.get("maxWindGust"), "flash_flood_threat": p.get("flashFloodDamageThreat"),
             "expires_local": local_time_label(exp).split(" · ")[0] if exp else "",
             "warning_text_excerpt": w["description"][:600],
             "hashtags": geo.state_hashtags(w["states"])}
    poster.post("quick_tag", [out], facts)


def act_event(a, state, warnings, poster):
    from .products import severe
    ev = engine.event(state, a["event_id"])
    c = a["cluster"]
    tor_places = []
    for w in c["members"]:
        if w["ph"] == "TO":
            name, _ = geo.nearest_place(w["center"][1], w["center"][0], min_pop=50000)
            if name and name not in tor_places:
                tor_places.append(name)
    title = f"Severe Weather · {', '.join(c['states'][:4])}"
    out = config.OUT_DIR / f"event{ev['id']}_{a['type']}.mp4"
    loop = severe.build_live_loop(out, warnings, c["bbox"], title)
    facts = {"counts": c["counts"], "states": c["states"], "tornado_warnings_near": tor_places[:6],
             "peak_tornado_warnings_this_event": ev["peak_tor"], **loop,
             "hashtags": geo.state_hashtags(c["states"]) + ["#severewx"]}
    pid = poster.post(a["type"], [out], facts, reply_to=ev.get("root_post") if a["type"] == "event_update" else None)
    if a["type"] == "event_start":
        ev["root_post"] = pid


def act_summary(a, state, poster, title=None, bbox=None):
    from .products import severe
    ev = engine.event(state, a["event_id"]) if a.get("event_id") else None
    start, end = datetime.fromisoformat(a["start"]), datetime.fromisoformat(a["end"])
    bbox = bbox or ev["bbox"]
    bbox = geo.bbox_for([(bbox[0], bbox[2]), (bbox[1], bbox[3])], pad=0.2)   # 16:9, capped size
    states = ev["states"] if ev else a.get("states", [])
    title = title or f"Event Recap · {', '.join(states[:4])}"
    out = config.OUT_DIR / f"summary_{start:%Y%m%d%H%M}.mp4"
    facts = severe.build_summary(out, start, end, bbox, title)
    facts["states"] = states
    facts["hashtags"] = geo.state_hashtags(states) + ["#severewx"]
    poster.post("event_summary", [out], facts, reply_to=ev.get("root_post") if ev else None)


def act_tropical(a, poster):
    from .products import tropical
    s = a["storm"]
    facts = tropical.storm_facts(s)
    facts["why_now"] = a["reasons"]
    if "rapid_intensification" in a["reasons"]:
        facts["wind_gain_kt_24h"] = a["kt_gain_24h"]
    loop = tropical.build_loop(config.OUT_DIR / f"{s['id']}_loop.mp4", s)
    cone = tropical.fetch_cone(s, config.OUT_DIR / f"{s['id']}_cone.png")
    if loop:
        pid = poster.post("tropical", [loop], facts)
        if cone:
            poster.post("tropical", [cone], {**facts, "note": "NHC forecast cone graphic, reply to the loop"},
                        reply_to=pid)
    elif cone:
        poster.post("tropical", [cone], facts)
    else:
        print("[tropical] no media available; skipping post")


def act_hazard(a, poster):
    from .products import hazards
    out = config.OUT_DIR / f"hazard_{a['group']}.png"
    facts = hazards.build(out, a)
    poster.post("hazard", [out], facts)


def act_storm_anim(poster, storm_id=None, name=None, year=None, live=False, kind="storm_anim", extra=None):
    from .products import storm_anim
    tag = storm_id or f"{name}{year}"
    out = config.OUT_DIR / f"{tag}_lifecycle.mp4"
    facts = storm_anim.build(out, storm_id=storm_id, name=name, year=year, live=live)
    facts.update(extra or {})
    poster.post(kind, [out], facts)


# --- commands ----------------------------------------------------------------
def cmd_watch(args, poster):
    now = datetime.now(timezone.utc)
    state = load_state()
    actions, state, warnings = engine.plan(now, state)
    print(f"{now:%Y-%m-%d %H:%MZ}: {len(warnings)} warnings with polygons; {len(actions)} action(s): "
          f"{[x['type'] for x in actions]}")
    if args.plan_only:
        # Nothing to post -> keep the bookkeeping (storm intensity history, event activity).
        # Something to post -> leave state alone; the build step re-decides and saves.
        if not actions:
            save_state(state)
        _gh_output("build_needed", "true" if actions else "false")
        return 0
    failures = 0
    for a in actions:
        try:
            if a["type"] == "quick_tag":
                act_quick_tag(a, warnings, poster)
            elif a["type"] in ("event_start", "event_update"):
                act_event(a, state, warnings, poster)
            elif a["type"] == "event_summary":
                act_summary(a, state, poster)
            elif a["type"] == "tropical":
                act_tropical(a, poster)
            elif a["type"] == "hazard":
                act_hazard(a, poster)
            elif a["type"] == "storm_recap":
                act_storm_anim(poster, storm_id=a["id"], live=True, kind="storm_recap")
        except Exception:  # noqa: BLE001 — one failed product must not block the others
            failures += 1
            traceback.print_exc()
    save_state(state)
    return 1 if failures and failures == len(actions) else 0


def cmd_upper_air(args, poster):
    from .products import upper_air
    cycle = (datetime.strptime(args.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
             if args.cycle else upper_air.latest_cycle())
    state = load_state()
    if state.get("upper_air_last") == cycle.isoformat() and not args.force:
        print(f"{cycle:%Y%m%d%HZ} already posted")
        return 0
    paths, facts = upper_air.build(config.OUT_DIR, cycle)
    if cycle.hour == 0 and not facts["notable"] and not args.force:
        print("00Z analysis not notable; not posting.", facts)
        return 0
    facts["hashtags"] = ["#wxtwitter", "#jetstream" if not facts["notable"] else "#anomaly"]
    poster.post("upper_air", paths, facts)
    state["upper_air_last"] = cycle.isoformat()
    save_state(state)
    return 0


def cmd_replay(args, poster):
    from .products import severe
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    if args.bbox:
        bbox = [float(x) for x in args.bbox.split(",")]
        states = []
    else:
        polys = [w for w in severe.fetch_sbw(start, end) if w["rings"] and w["ph"] in ("TO", "SV")]
        for w in polys:
            pts = [p for r in w["rings"] for p in r]
            w["center"] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
            w["states"] = []
        if not polys:
            print("No warnings found in that window.")
            return 1
        best = severe.clusters(polys, link_deg=2.0)[0]
        bbox = best["bbox"]
        lsr = severe.fetch_lsr(start, end, bbox)
        states = sorted({r["state"] for r in lsr if r.get("state")})
    a = {"start": start.isoformat(), "end": end.isoformat(), "states": states}
    act_summary(a, {}, poster, title=args.title or f"Event Recap · {', '.join(states[:4]) or 'Severe Weather'}",
                bbox=bbox)
    return 0


def cmd_hazard(args, poster):
    """Build (and post) a hazard analysis map right now, even below the trigger threshold."""
    from . import net
    from .products import hazards
    c = hazards.count(net.get_json(config.NWS_ALERTS))[args.group]
    print(f"{args.group}: {c['zones']} zones (trigger {hazards.GROUPS[args.group]['threshold']}) in {c['states']}")
    if not c["zones"] and not args.force:
        print("Nothing active; use --force to draw the map anyway.")
        return 0
    act_hazard({"type": "hazard", "group": args.group, "phase": "manual", "zones": c["zones"],
                "states": c["states"], "day": 1}, poster)
    return 0


def cmd_storm_anim(args, poster):
    """Storm-following IR animation for any storm: --id AL122005, or --name KATRINA --year 2005."""
    act_storm_anim(poster, storm_id=args.id, name=args.name, year=args.year, live=args.live)
    return 0


def anniversaries_due(today, hurdat_text, storms=None):
    """Landmark storms whose strongest landfall (or peak, if no landfall) happened on this calendar day."""
    from .products import storm_anim
    due = []
    for name, year in storms or config.ANNIVERSARY_STORMS:
        try:
            tr = storm_anim.parse_hurdat(hurdat_text, name=name, year=year)
        except KeyError:
            continue
        lf = [p for p in tr["points"] if p["rec"] == "L"]
        key = max(lf or tr["points"], key=lambda p: (p["kt"], -p["mb"]))
        if (key["t"].month, key["t"].day) == (today.month, today.day) and today.year > year:
            due.append({"id": tr["id"], "name": tr["name"], "year": year, "years_ago": today.year - year,
                        "date": f"{key['t']:%b %-d, %Y}", "landfall": bool(lf),
                        "kt": key["kt"], "lat": key["lat"], "lon": key["lon"]})
    # round anniversaries (5, 10, 15...) first, then the strongest storm
    return sorted(due, key=lambda d: (d["years_ago"] % 5 == 0, d["kt"]), reverse=True)


def cmd_anniversary(args, poster):
    from .products import storm_anim
    today = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.now(timezone.utc).date()
    state = load_state()
    done = set(state.get("anniversaries_posted", []))
    text = storm_anim._hurdat_file("AL").read_text()
    due = [d for d in anniversaries_due(today, text) if f"{d['id']}:{today.year}" not in done]
    print(f"{today}: {len(due)} anniversary storm(s): {[d['name'] for d in due]}")
    for d in due[:1]:                                  # at most one per day
        place, _ = geo.nearest_place(d["lat"], d["lon"], min_pop=20000)
        act_storm_anim(poster, storm_id=d["id"], kind="anniversary",
                       extra={"years_ago": d["years_ago"], "anniversary_of": d["date"],
                              "what_happened_that_day": f"{'landfall near ' + place if d['landfall'] else 'peak intensity'}"})
        done.add(f"{d['id']}:{today.year}")
    state["anniversaries_posted"] = sorted(done)[-200:]
    save_state(state)
    return 0


def cmd_status(args, poster):
    print(json.dumps(load_state(), indent=1, default=str))
    return 0


def _gh_output(k, v):
    import os
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as fh:
            fh.write(f"{k}={v}\n")
    print(f"{k}={v}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="studio")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch"); w.add_argument("--plan-only", action="store_true")   # noqa: E702
    u = sub.add_parser("upper-air"); u.add_argument("--cycle"); u.add_argument("--force", action="store_true")  # noqa: E702
    r = sub.add_parser("replay")
    r.add_argument("--start", required=True); r.add_argument("--end", required=True)  # noqa: E702
    r.add_argument("--bbox"); r.add_argument("--title")  # noqa: E702
    hz = sub.add_parser("hazard")
    hz.add_argument("--group", required=True, choices=["blizzard", "ice", "winter", "cold", "heat", "wind",
                                                        "fire", "flood", "dust"])
    hz.add_argument("--force", action="store_true")
    sa = sub.add_parser("storm-anim")
    sa.add_argument("--id"); sa.add_argument("--name"); sa.add_argument("--year", type=int)   # noqa: E702
    sa.add_argument("--live", action="store_true")
    an = sub.add_parser("anniversary"); an.add_argument("--date")   # noqa: E702
    sub.add_parser("status")
    for sp in sub.choices.values():
        sp.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    poster = Poster(a.dry_run or config.DRY_RUN)
    return {"watch": cmd_watch, "upper-air": cmd_upper_air, "replay": cmd_replay,
            "hazard": cmd_hazard, "storm-anim": cmd_storm_anim,
            "anniversary": cmd_anniversary, "status": cmd_status}[a.cmd](a, poster)


if __name__ == "__main__":
    sys.exit(main())
