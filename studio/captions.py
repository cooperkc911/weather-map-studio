"""Post text written by Claude from the facts each product returns, with a template fallback.

Captions never contain links: X charges ~13x more for a post with a URL, and text that looks
like a domain (e.g. "weather.gov") gets auto-linked, so those are stripped too.
"""
import json
import os
import re

from . import config

MAX_LEN = 275

PERSONA = os.getenv("BOT_PERSONA") or "cranky"     # "cranky" or "straight"

CRANKY = """You are the voice of a cranky, deadpan veteran weather desk on X. Think: a tired guy who has
seen 40 winters, grumbles about the weather like it personally wronged him, and still gives you the
real numbers. Dry, short sentences. Annoyed AT THE WEATHER, never at people, never at victims.
The joke is always secondary to the information; one wry line at most, then the facts.

Examples of the voice (style only, never reuse their numbers):
- "Extreme Cold Warnings for 412 zones across ND, MN, SD, IA. Fargo wind chill -38. Bismarck -35. I'm not going outside and neither should you. #NDwx #MNwx"
- "Red Flag Warnings over 300 zones. Gusts 68 mph at Amarillo, humidity 6% at Lubbock. The air is basically a hair dryer. Don't park on dry grass. #TXwx #firewx"
- "500 mb trough digging into the Rockies, -27 dam below normal near Denver. 160 kt jet over the Four Corners. Somebody's getting weather. #wxtwitter"
"""

STRAIGHT = """You are the voice of a sharp, calm operational meteorologist on X: dense with real numbers,
no fluff, no hype words."""

RULES = """This is NOT a forecast account: describe what is happening now and what the data shows.

Hard rules:
- Use ONLY the facts provided. Never invent numbers, places, times, injuries, damage, or forecasts.
- Max 270 characters total, including hashtags.
- No links, no website or domain names, no @mentions. At most one emoji.
- End with the hashtags given in the facts (pick at most 3).
- If "tone" is "serious": NO jokes, NO sarcasm, NO grumbling. Plain, urgent, factual. Lives may be at stake.
- Post types:
  quick_tag: start with "🚨" + the alert label in caps + place, then the threat details
             (observed/radar-indicated, damage threat, hail/wind) and expiry time.
  event_start / event_update: what is unfolding (counts, states, where the tornado warnings are).
  event_summary: recap with totals (warnings issued, tornado/hail/wind reports, biggest hail, peak gust).
  tropical: storm label and why it matters now (new storm, category jump, rapid intensification with
            the kt gain, or a major hurricane near the US), then wind/pressure/position.
  upper_air: the most anomalous features (500 mb ridge/trough anomalies in dam, 850 mb temp anomalies
             in C, jet max) with locations.
  hazard: a national hazard is big right now (blizzard, ice, winter storm, extreme cold, heat, high wind,
          fire weather, flooding). Say how big (zones, states), the most extreme current observations
          (wind chills, heat index, gusts, snow/ice reports) and the surface low if given.
  storm_recap: a storm just died; recap its whole life (dates, peak wind/pressure, landfalls, days).
  storm_anim: a storm-following animation of a storm's life (same facts as storm_recap).
  anniversary: "N years ago today" for a landmark storm; say what happened that day, the peak, landfalls.
               Respectful: these storms killed people. No jokes.
Return only the post text."""


def tone(kind, facts):
    if kind in ("quick_tag", "anniversary"):
        return "serious"
    if kind in ("storm_recap", "storm_anim") and str(facts.get("peak_category")) in ("4", "5"):
        return "serious"
    if kind == "tropical" and ("major_hurricane_near_us" in facts.get("why_now", []) or
                               "Category 4" in str(facts.get("storm")) or "Category 5" in str(facts.get("storm"))):
        return "serious"
    if kind == "hazard" and facts.get("hazard") in ("Flood Warnings", "Ice Storm Warnings", "Blizzard Warnings"):
        return "serious" if facts.get("zones_under_warning", 0) >= 60 else "dry"
    if kind in ("event_start", "event_update") and facts.get("counts", {}).get("TO", 0) >= 8:
        return "serious"
    return "dry" if PERSONA == "cranky" else "straight"


def system_prompt():
    return (CRANKY if PERSONA == "cranky" else STRAIGHT) + "\n\n" + RULES


SYSTEM = system_prompt()

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_DOMAIN_RE = re.compile(r"\b[\w-]+(\.[\w-]+)*\.(gov|com|org|net|edu|io|us|co|info|ly|me|tv|ai)\b\S*", re.I)
_MENTION_RE = re.compile(r"(?<!\w)@\w+")


def sanitize(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = _URL_RE.sub("", text)
    text = _DOMAIN_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if len(text) > MAX_LEN:
        cut = text[:MAX_LEN - 1]
        text = cut[:cut.rfind(" ")].rstrip(" ,;:-") + "…"
    return text


def _tags(facts):
    return " ".join(facts.get("hashtags", [])[:3])


def template(kind, f):
    try:
        if kind == "quick_tag":
            t = f"🚨 {f['label']} — {f['near']}. {f.get('detection') or ''} Until {f['expires_local']}."
        elif kind in ("event_start", "event_update"):
            c = f["counts"]
            t = (f"Severe weather unfolding across {', '.join(f['states'][:4])}: {c['TO']} tornado, "
                 f"{c['SV']} severe t-storm and {c['FF']} flash flood warnings in effect.")
        elif kind == "event_summary":
            wi = f["warnings_issued"]
            t = (f"Event recap ({f['window']}): {wi['tornado']} tornado warnings, {f['tornado_reports']} tornado, "
                 f"{f['hail_reports']} hail and {f['wind_reports']} wind reports.")
        elif kind == "tropical":
            t = f"{f['storm']}: {f['max_wind_mph']} mph, {f['pressure_mb']} mb at {f['position']}."
        elif kind == "hazard":
            t = (f"{f['hazard']} now cover {f['zones_under_warning']} zones across "
                 f"{', '.join(f['states'][:5])}.") + (" Of course they do." if f.get("tone") == "dry" else "")
            for k in ("lowest_wind_chills", "highest_heat_index", "peak_gusts", "notable_obs"):
                if f.get(k):
                    t += f" {f[k][0]}."
                    break
        elif kind in ("storm_recap", "storm_anim", "anniversary"):
            t = (f"{f.get('years_ago', '')} years ago today: " if kind == "anniversary" else "") + \
                f"{f['storm']} ({f['dates']}). Peak {f['peak']}." + \
                (f" Landfall: {f['landfalls'][-1]}." if f.get("landfalls") else "")
        elif kind == "upper_air":
            t = (f"Upper-air analysis: 500 mb {f['deepest_trough_500mb']}; ridge {f['strongest_ridge_500mb']}. "
                 f"250 mb jet max {f['jet_max_250mb']}.")
        else:
            t = "Weather update."
    except (KeyError, TypeError):
        t = "Weather update."
    return sanitize(f"{t} {_tags(f)}")


def write(kind, facts):
    facts = {**facts, "tone": tone(kind, facts)}
    if not os.getenv("ANTHROPIC_API_KEY"):
        return template(kind, facts)
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=config.ANTHROPIC_MODEL, max_tokens=250, system=system_prompt(),
            messages=[{"role": "user", "content":
                       f"Post type: {kind}\nFacts (JSON):\n{json.dumps(facts, indent=1, default=str)}"}])
        text = sanitize("".join(b.text for b in msg.content if b.type == "text"))
        return text if len(text) >= 20 else template(kind, facts)
    except Exception as e:  # noqa: BLE001 — captions must never block a post
        print(f"[captions] Claude failed ({e}); using template")
        return template(kind, facts)
