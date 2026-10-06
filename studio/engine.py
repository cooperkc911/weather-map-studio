"""The watcher: decides what (if anything) is worth posting right now, then builds and posts it.

Decision functions are pure (data + state in, actions + new state out) so they can be tested
and so the cheap "plan" step can run before any heavy plotting libraries are installed.
"""
from datetime import datetime, timedelta

from . import config
from . import net
from .products import hazards, severe, tropical

MAX_QUICK_TAGS_PER_HOUR = 4


def _t(s):
    return datetime.fromisoformat(s)


def _overlap(a, b, grow=1.0):
    return not (a[1] + grow < b[0] or b[1] + grow < a[0] or a[3] + grow < b[2] or b[3] + grow < a[2])


def decide_severe(warnings, state, now):
    st = state.setdefault("severe", {"events": [], "tagged": {}, "next_id": 1})
    actions = []

    # forget old quick-tag keys (warnings last < 3h)
    st["tagged"] = {k: v for k, v in st["tagged"].items() if now - _t(v) < timedelta(hours=6)}

    # 1) high-end warnings -> immediate quick tag (rate limited)
    recent = sum(1 for v in st["tagged"].values() if now - _t(v) < timedelta(hours=1))
    for w in sorted(warnings, key=lambda w: severe.high_end_label(w) != "TORNADO EMERGENCY"):
        if severe.is_high_end(w) and w["key"] not in st["tagged"] and recent < MAX_QUICK_TAGS_PER_HOUR:
            st["tagged"][w["key"]] = now.isoformat()
            recent += 1
            actions.append({"type": "quick_tag", "warning": w})

    # 2) clusters -> start / update events
    cl = severe.clusters(warnings)
    active = [e for e in st["events"] if not e.get("ended")]
    for c in cl:
        match = next((e for e in active if _overlap(e["bbox"], c["bbox"])), None)
        if match:
            match["bbox"] = [min(match["bbox"][0], c["bbox"][0]), max(match["bbox"][1], c["bbox"][1]),
                             min(match["bbox"][2], c["bbox"][2]), max(match["bbox"][3], c["bbox"][3])]
            if c["total"] >= config.EVENT_END_COUNT:
                match["last_active"] = now.isoformat()
            match["peak_tor"] = max(match.get("peak_tor", 0), c["counts"]["TO"])
            match["peak_total"] = max(match.get("peak_total", 0), c["total"])
            match["states"] = sorted(set(match["states"]) | set(c["states"]))
            gap_ok = now - _t(match["last_post"]) >= timedelta(minutes=config.EVENT_UPDATE_GAP_MIN)
            if gap_ok and severe.meets_event_threshold(c):
                match["last_post"] = now.isoformat()
                actions.append({"type": "event_update", "event_id": match["id"], "cluster": c})
        elif severe.meets_event_threshold(c):
            ev = {"id": st["next_id"], "start": now.isoformat(), "last_active": now.isoformat(),
                  "last_post": now.isoformat(), "bbox": c["bbox"], "states": c["states"],
                  "peak_tor": c["counts"]["TO"], "peak_total": c["total"], "root_post": None}
            st["next_id"] += 1
            st["events"].append(ev)
            active.append(ev)
            actions.append({"type": "event_start", "event_id": ev["id"], "cluster": c})

    # 3) quiet long enough -> summary
    for e in active:
        if now - _t(e["last_active"]) >= timedelta(minutes=config.EVENT_END_QUIET_MIN):
            e["ended"] = now.isoformat()
            start = max(_t(e["start"]) - timedelta(minutes=45), now - timedelta(hours=config.EVENT_MAX_HOURS))
            actions.append({"type": "event_summary", "event_id": e["id"], "start": start.isoformat(),
                            "end": _t(e["last_active"]).isoformat()})
    st["events"] = [e for e in st["events"] if not e.get("ended") or now - _t(e["ended"]) < timedelta(days=2)]
    return actions, state


def event(state, eid):
    return next((e for e in state["severe"]["events"] if e["id"] == eid), None)


def plan(now, state, alerts=None, storms=None):
    """alerts: raw NWS alert GeoJSON (fetched once, shared by the severe and hazard checks)."""
    alerts = net.get_json(config.NWS_ALERTS) if alerts is None else alerts
    storms = tropical.fetch() if storms is None else storms
    warnings = severe.parse_live(alerts)
    a1, state = decide_severe(warnings, state, now)
    a2, state = tropical.decide(storms, state, now)
    a3, state = hazards.decide(hazards.count(alerts), state, now)
    return a1 + a2 + a3, state, warnings
