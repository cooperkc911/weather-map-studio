"""Offline tests: every trigger and product is exercised with synthetic data (no network)."""
import io
import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

from studio import captions, config, engine, geo, radar
from studio.products import severe, tropical, upper_air

NOW = datetime(2026, 5, 6, 21, 0, tzinfo=timezone.utc)


# --- helpers -----------------------------------------------------------------
def box(cx, cy, w=0.3, h=0.2):
    return [[[cx - w, cy - h], [cx + w, cy - h], [cx + w, cy + h], [cx - w, cy + h], [cx - w, cy - h]]]


def warning(i, ph="TO", cx=-97.0, cy=36.0, **params):
    ev = {"TO": "Tornado Warning", "SV": "Severe Thunderstorm Warning", "FF": "Flash Flood Warning"}[ph]
    return {"key": f"OUN.{ph}.{i}", "event": ev, "ph": ph, "rings": box(cx, cy), "center": (cx, cy),
            "issued": NOW.isoformat(), "expires": (NOW + timedelta(minutes=45)).isoformat(),
            "area": "Oklahoma, OK", "states": ["OK"], "headline": "", "description": "", "params": params,
            "sender": "NWS Norman OK"}


def fake_png(t, bbox_center=(-97, 36)):
    """Small palette PNG mimicking the n0q composite (0.05 deg/pixel)."""
    W, H = 1220, 540
    yy, xx = np.mgrid[0:H, 0:W]
    lon, lat = -126 + xx * 0.05, 50 - yy * 0.05
    d = 60 * np.exp(-(((lon - bbox_center[0]) / 0.5) ** 2 + ((lat - bbox_center[1]) / 0.3) ** 2))
    idx = np.clip((d + 32) * 2, 0, 255).astype(np.uint8)
    im = Image.fromarray(idx, "P")
    im.putpalette([v for i in range(256) for v in ((0, 0, 0) if i < 94 else (255, 60, 0))])
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


@pytest.fixture
def coarse_radar(monkeypatch):
    monkeypatch.setattr(config, "RADAR_RES", 0.05)

    def fetch(t, bbox, back_steps=3):
        arr, ext = radar.crop(fake_png(t), bbox)
        return t, arr, ext
    return fetch


# --- radar ---------------------------------------------------------------------
def test_radar_crop_and_times(coarse_radar):
    t, arr, ext = coarse_radar(NOW, [-99, -95, 35, 37])
    assert arr.shape[2] == 4 and arr[..., 3].max() == 235 and arr[..., 3].min() == 0
    assert abs(ext[0] + 99) < 0.06 and abs(ext[3] - 37) < 0.06
    ts = radar.frame_times(NOW - timedelta(hours=6), NOW, 20)
    assert len(ts) <= 21 and ts[-1] == NOW and all(x.minute % 5 == 0 for x in ts)


# --- severe triggers ----------------------------------------------------------
def test_clusters_split_far_apart_groups():
    ws = [warning(i, "TO", -97 + i * 0.4, 36) for i in range(4)] + [warning(9, "SV", -80, 40)]
    cl = severe.clusters(ws)
    assert len(cl) == 2 and cl[0]["counts"]["TO"] == 4 and severe.meets_event_threshold(cl[0])


def test_event_lifecycle_start_update_summary(monkeypatch):
    monkeypatch.setattr(config, "EVENT_UPDATE_GAP_MIN", 90)
    monkeypatch.setattr(config, "EVENT_END_QUIET_MIN", 90)
    ws = [warning(i, "TO", -97 + i * 0.4, 36) for i in range(5)]
    a, st = engine.decide_severe(ws, {}, NOW)
    assert [x["type"] for x in a] == ["event_start"]
    a, st = engine.decide_severe(ws, st, NOW + timedelta(minutes=30))
    assert a == []                                        # too soon for an update
    a, st = engine.decide_severe(ws, st, NOW + timedelta(minutes=100))
    assert [x["type"] for x in a] == ["event_update"]
    a, st = engine.decide_severe([], st, NOW + timedelta(minutes=150))
    assert a == []                                        # quiet, but not long enough
    a, st = engine.decide_severe([], st, NOW + timedelta(minutes=200))
    assert [x["type"] for x in a] == ["event_summary"]
    a, st = engine.decide_severe([], st, NOW + timedelta(minutes=260))
    assert a == []                                        # summary only once


def test_small_cluster_does_not_trigger():
    a, _ = engine.decide_severe([warning(1, "TO"), warning(2, "SV", -96.5)], {}, NOW)
    assert a == []


def test_quick_tag_once_per_warning():
    pds = warning(7, "TO", tornadoDamageThreat="CONSIDERABLE", tornadoDetection="OBSERVED")
    emergency = warning(8, "TO", -90, 33, tornadoDamageThreat="CATASTROPHIC")
    a, st = engine.decide_severe([pds, emergency], {}, NOW)
    assert [x["type"] for x in a] == ["quick_tag", "quick_tag"]
    assert severe.high_end_label(a[0]["warning"]) == "TORNADO EMERGENCY"   # emergencies first
    a, st = engine.decide_severe([pds, emergency], st, NOW + timedelta(minutes=10))
    assert a == []


def test_parse_live_alert_json():
    feat = {"geometry": {"type": "Polygon", "coordinates": box(-97, 36)},
            "properties": {"event": "Tornado Warning", "areaDesc": "Cleveland, OK; McClain, OK",
                           "sent": NOW.isoformat(), "expires": NOW.isoformat(), "senderName": "NWS Norman OK",
                           "parameters": {"VTEC": ["/O.NEW.KOUN.TO.W.0042.260506T2100Z-260506T2145Z/"],
                                          "tornadoDamageThreat": ["CONSIDERABLE"], "NWSheadline": ["X"]}}}
    w = severe.parse_live({"features": [feat, {"geometry": None, "properties": {"event": "Heat Advisory"}}]})
    assert len(w) == 1 and w[0]["key"] == "KOUN.TO.0042" and w[0]["states"] == ["OK"]
    assert severe.is_high_end(w[0])


def test_report_stats():
    lsrs = [{"type": "H", "mag": "2.75", "city": "Moore", "state": "OK", "time": NOW, "lat": 35, "lon": -97},
            {"type": "G", "mag": "81", "city": "Norman", "state": "OK", "time": NOW, "lat": 35, "lon": -97},
            {"type": "T", "mag": "", "city": "Newcastle", "state": "OK", "time": NOW, "lat": 35, "lon": -97}]
    s = severe.report_stats(lsrs)
    assert s["tornado_reports"] == 1 and "2.75" in s["largest_hail"] and "81 mph" in s["peak_gust"]


def test_live_loop_and_summary_videos(tmp_path, coarse_radar, monkeypatch):
    monkeypatch.setattr(config, "SUMMARY_FRAMES", 8)
    ws = [warning(i, "TO", -97 + i * 0.3, 36) for i in range(3)]
    bbox = geo.bbox_for([(-98, 35), (-95, 37)])
    f = tmp_path / "loop.mp4"
    info = severe.build_live_loop(f, ws, bbox, "Test", minutes=30, now=NOW, radar_fetch=coarse_radar,
                                  highlight_keys={"OUN.TO.0"})
    assert info["warnings_in_view"]["TO"] == 3 and f.stat().st_size > 5000
    sbw = [{"ph": "TO", "rings": box(-97, 36), "begin": NOW, "end": NOW + timedelta(hours=1),
            "status": "NEW", "key": "OUN.TO.1"}]
    lsr = [{"type": "T", "lat": 36, "lon": -97, "time": NOW + timedelta(minutes=20), "mag": "", "city": "A",
            "state": "OK", "remark": ""}]
    g = tmp_path / "sum.mp4"
    facts = severe.build_summary(g, NOW, NOW + timedelta(hours=2), bbox, "Recap", sbw=sbw, lsrs=lsr,
                                 radar_fetch=coarse_radar)
    assert facts["warnings_issued"]["tornado"] == 1 and facts["tornado_reports"] == 1
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(g)], capture_output=True, text=True).stdout)
    assert dur > 2


# --- tropical -----------------------------------------------------------------
def storm(kt, adv="5", lat=25.0, lon=-60.0, sid="AL092026"):
    return {"id": sid, "bin": "AT4", "name": "Ida", "cls": "HU" if kt >= 64 else "TS", "kt": kt, "mb": 980,
            "lat": lat, "lon": lon, "move_dir": 300, "move_mph": 12, "updated": "", "adv": adv}


def test_tropical_triggers():
    a, st = tropical.decide([storm(30)], {}, NOW)
    assert a == []                                           # depression: not yet
    a, st = tropical.decide([storm(40)], st, NOW + timedelta(hours=6))
    assert a[0]["reasons"] == ["new_storm"]
    a, st = tropical.decide([storm(55)], st, NOW + timedelta(hours=12))
    assert a == []
    a, st = tropical.decide([storm(75)], st, NOW + timedelta(hours=18))
    assert set(a[0]["reasons"]) == {"category_up", "rapid_intensification"}   # 40 -> 75 kt in 12 h
    a, st = tropical.decide([storm(80)], st, NOW + timedelta(hours=24))
    assert a == []                                           # still Cat 1, RI already posted
    a, st = tropical.decide([], st, NOW + timedelta(hours=30))
    assert st["storms"] == {}                                # dissipated storms are forgotten


def test_major_hurricane_near_us_every_full_advisory():
    st = {}
    a, st = tropical.decide([storm(110, "20", 27.0, -88.0)], st, NOW)
    assert "major_hurricane_near_us" in a[0]["reasons"]
    a, st = tropical.decide([storm(110, "20A", 27.2, -88.2)], st, NOW + timedelta(hours=3))
    assert a == []                                           # intermediate advisory
    a, st = tropical.decide([storm(110, "21", 27.5, -88.5)], st, NOW + timedelta(hours=6))
    assert a[0]["reasons"] == ["major_hurricane_near_us"]


def test_tropical_helpers(tmp_path):
    assert [tropical.category(k) for k in (30, 34, 64, 96, 140)] == [0, 0.5, 1, 3, 5]
    html = "".join(f'<a href="2026279{h:02d}00_GOES19-ABI-AL092026-GEOCOLOR-{s}.jpg">'
                   for h in range(10, 20) for s in ("500x500", "1000x1000", "2000x2000"))
    names = tropical.list_frames(html, 4)
    assert len(names) == 4 and all("1000x1000" in n for n in names) and names[-1].startswith("202627919")
    assert tropical.frame_time(names[-1]) == datetime(2026, 10, 6, 19, 0, tzinfo=timezone.utc)
    frames = [(NOW + timedelta(minutes=10 * i), Image.new("RGB", (1000, 1000), (20, 40, 90 + i))) for i in range(9)]
    assert tropical.build_loop(tmp_path / "ida.mp4", storm(120), frames=frames).exists()
    f = tropical.storm_facts(storm(120, lat=27, lon=-88))
    assert f["max_wind_mph"] == 140 and "Major Hurricane Ida (Category 4)" == f["storm"]


# --- upper air ------------------------------------------------------------------
def synth_gfs():
    lat, lon = np.arange(10, 75.01, 0.5), np.arange(190, 330.01, 0.5)
    LO, LA = np.meshgrid(lon, lat)
    trough = -30 * np.exp(-(((LO - 262) / 7) ** 2 + ((LA - 42) / 6) ** 2))
    base = 588 - 1.25 * (LA - 25)
    g = {"lat": lat, "lon": lon, "z500": base + trough, "z250": base * 1.85 + trough,
         "u250": np.full(LA.shape, 40.0), "v250": np.zeros(LA.shape), "t850": 28 - 0.75 * (LA - 15) + trough * 0.4}
    return g, {"z500": base, "t850": 28 - 0.75 * (LA - 15)}


def test_upper_air_maps_and_facts(tmp_path):
    g, clim = synth_gfs()
    cycle = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    paths, f = upper_air.build(tmp_path, cycle, g=g, clim=clim)
    assert len(paths) == 3 and all(Image.open(p).size == (1600, 1000) for p in paths)
    assert f["deepest_trough_500mb"].startswith("-30 dam") and "NE" in f["deepest_trough_500mb"]
    assert f["notable"] is True and f["jet_max_250mb"].startswith("78 kt")


def test_climatology_time_interpolation():
    cube = np.stack([np.full((3, 4), m, float) for m in range(12)])   # value = month index
    clat, clon = np.array([90.0, 0, -90]), np.array([0.0, 90, 180, 270])
    mid_jan = upper_air.interp_month(cube, clat, clon, datetime(2026, 1, 16, tzinfo=timezone.utc),
                                     np.array([40.0]), np.array([260.0]))
    jan_feb = upper_air.interp_month(cube, clat, clon, datetime(2026, 1, 31, tzinfo=timezone.utc),
                                     np.array([40.0]), np.array([260.0]))
    assert abs(mid_jan[0, 0] - 0) < 0.05 and 0.4 < jan_feb[0, 0] < 0.6


def test_aws_idx_ranges():
    idx = "1:0:d=2026100612:HGT:250 mb:anl:\n2:500:d=2026100612:TMP:250 mb:anl:\n3:900:d=2026100612:HGT:500 mb:anl:\n"
    assert upper_air._aws_ranges(idx) == [(0, 499), (900, "")]


def test_latest_cycle():
    assert upper_air.latest_cycle(datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)).hour == 0
    assert upper_air.latest_cycle(datetime(2026, 10, 6, 17, 0, tzinfo=timezone.utc)).hour == 12


# --- text & X -------------------------------------------------------------------
def test_sanitizer_and_templates(monkeypatch):
    t = captions.sanitize('"🚨 TORNADO EMERGENCY. See https://x.co/a or weather.gov/oun @NWSNorman #okwx"')
    assert "http" not in t and ".gov" not in t and "@" not in t and "#okwx" in t
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t = captions.write("event_start", {"counts": {"TO": 6, "SV": 9, "FF": 1}, "states": ["OK", "KS"],
                                       "hashtags": ["#okwx", "#kswx"]})
    assert "6 tornado" in t and t.endswith("#okwx #kswx")
    assert geo.state_hashtags(["OK", "KS", "MO", "AR"]) == ["#OKwx", "#KSwx", "#MOwx"]


def test_x_upload_post_and_reply(tmp_path, monkeypatch):
    from studio import x_client
    for k in ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET"]:
        monkeypatch.setenv(k, "t")
    calls = []

    class R:
        def __init__(self, d, url):
            self.status_code, self._d, self.content, self.url = 200, d, b"x", url
            self.request = type("Q", (), {"method": "POST"})()

        def json(self):
            return self._d

    def post(url, **kw):
        calls.append((url.split("/2/")[1], kw.get("json"), kw.get("data")))
        if url.endswith("/initialize"):
            return R({"data": {"id": "1"}}, url)
        if url.endswith("/finalize"):
            return R({"data": {"id": "1"}}, url)
        if url.endswith("/tweets"):
            return R({"data": {"id": "99"}}, url)
        return R({}, url)

    monkeypatch.setattr(x_client.requests, "post", post)
    monkeypatch.setattr(x_client, "CHUNK", 4)
    f = tmp_path / "a.png"
    f.write_bytes(b"0123456789")
    assert x_client.upload_media(f) == "1"
    assert calls[0][1]["media_category"] == "tweet_image"
    assert [c[2]["segment_index"] for c in calls if c[0].endswith("append")] == ["0", "1", "2"]
    assert x_client.create_post("hi", ["1"], reply_to="42") == "99"
    assert calls[-1][1] == {"text": "hi", "media": {"media_ids": ["1"]}, "reply": {"in_reply_to_tweet_id": "42"}}


# --- end to end: the real CLI path with every network call replaced -----------------
def test_watch_command_end_to_end(tmp_path, monkeypatch, coarse_radar):
    import studio.__main__ as cli
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ws = [warning(i, "TO", -97 + i * 0.4, 36) for i in range(5)]
    ws.append(warning(99, "TO", -95.5, 36.4, tornadoDamageThreat="CATASTROPHIC", tornadoDetection="OBSERVED"))
    feats = [{"geometry": {"type": "Polygon", "coordinates": w["rings"]},
              "properties": {"event": w["event"], "areaDesc": "Tulsa, OK", "sent": NOW.isoformat(),
                             "expires": (NOW + timedelta(hours=1)).isoformat(), "senderName": "NWS",
                             "parameters": {"VTEC": [f"/O.NEW.KTSA.TO.W.{i:04d}.X/"],
                                            **{k: [v] for k, v in w["params"].items()}}}}
             for i, w in enumerate(ws)]
    feats += [{"geometry": None, "properties": {"event": "Extreme Cold Warning",
                                                "geocode": {"UGC": [f"NDZ{i:03d}" for i in range(200)]}}}]
    from studio import net
    monkeypatch.setattr(net, "get_json", lambda url: {"features": feats})
    from studio.products import hazards
    monkeypatch.setattr(hazards, "fetch_zone_shapes", lambda codes: [box(-100, 47)[0]])
    monkeypatch.setattr(hazards, "fetch_metars", lambda: [
        {"id": "KFAR", "name": "Fargo, ND", "lat": 46.9, "lon": -96.8, "temp_f": -20, "wind_mph": 15,
         "gust_mph": 25, "wind_chill_f": -41, "heat_index_f": -20, "rh": 60, "wx": ""}])
    monkeypatch.setattr(hazards, "fetch_fronts", lambda now=None: None)
    monkeypatch.setattr(radar, "fetch", coarse_radar)
    monkeypatch.setattr(tropical, "fetch", lambda: [storm(100, lat=24, lon=-70)])
    frames = [(NOW + timedelta(minutes=10 * i), Image.new("RGB", (800, 800), (10, 30, 80))) for i in range(9)]
    monkeypatch.setattr(tropical, "satellite_frames", lambda s, count=None: (frames, "x"))

    def cone(s, out):
        Image.new("RGB", (600, 400), "white").save(out)
        return out
    monkeypatch.setattr(tropical, "fetch_cone", cone)
    assert cli.main(["watch", "--plan-only"]) == 0
    assert not (tmp_path / "state.json").exists()          # actions pending -> plan leaves state alone
    assert cli.main(["watch", "--dry-run"]) == 0
    out = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert any(n.startswith("quick_") and n.endswith(".mp4") for n in out)
    assert "event1_event_start.mp4" in out and "AL092026_loop.mp4" in out and "hazard_cold.png" in out and "hazard_cold.png" in out
    texts = [p.read_text() for p in (tmp_path / "out").glob("*.txt")]
    assert any("TORNADO EMERGENCY" in t for t in texts)
    assert any("reply_to: dry-" in t for t in texts)       # cone posted as a reply to the loop
    # second run a minute later: nothing new to post
    assert cli.main(["watch", "--plan-only"]) == 0


# --- national hazard scan ---------------------------------------------------------
def test_hazard_count_and_lifecycle():
    from studio.products import hazards
    alerts = {"features": [{"properties": {"event": "Blizzard Warning",
                                           "geocode": {"UGC": [f"SDZ{i:03d}" for i in range(20)]}}},
                           {"properties": {"event": "Winter Storm Warning", "geocode": {"UGC": ["MNZ001"]}}}]}
    c = hazards.count(alerts)
    assert c["blizzard"]["zones"] == 20 and c["blizzard"]["states"] == ["SD"] and c["winter"]["zones"] == 1
    a, st = hazards.decide(c, {}, NOW)
    assert [(x["group"], x["phase"]) for x in a] == [("blizzard", "onset")]
    a, st = hazards.decide(c, st, NOW + timedelta(hours=3))
    assert a == []
    c["blizzard"]["zones"] = 40
    a, st = hazards.decide(c, st, NOW + timedelta(hours=7))
    assert a[0]["phase"] == "expanding"
    a, st = hazards.decide(c, st, NOW + timedelta(hours=32))
    assert a[0]["phase"] == "ongoing" and a[0]["day"] == 2
    c["blizzard"]["zones"] = 2
    a, st = hazards.decide(c, st, NOW + timedelta(hours=40))
    assert a == [] and "blizzard" not in st["hazards"]


def test_hazard_math_fronts_and_map(tmp_path):
    from studio.products import hazards
    assert round(hazards.wind_chill_f(-10, 20)) == -35
    assert round(hazards.heat_index_f(96, 60)) == 116   # NWS heat index chart
    fr = hazards.parse_codsus("VALID 010612Z\nHIGHS 1050 4801010\nLOWS 994 4200780\nCOLD 4200780 3900820\n3500880\n$$",
                              datetime(2026, 1, 6, 14, tzinfo=timezone.utc))
    assert fr["valid"].hour == 12 and fr["centers"][1] == {"kind": "L", "mb": 994, "lon": -78.0, "lat": 42.0}
    assert len(fr["fronts"][0]["pts"]) == 3
    obs = [{"id": "A", "name": "Fargo, ND", "lat": 46.9, "lon": -96.8, "temp_f": -20, "wind_mph": 15, "gust_mph": 0,
            "wind_chill_f": -41, "heat_index_f": -20, "rh": 60, "wx": "BLSN"}]
    f = hazards.build(tmp_path / "h.png", {"group": "blizzard", "phase": "onset", "zones": 30, "states": ["ND"]},
                      NOW, rings=[box(-97, 47)[0]], obs=obs, fronts=fr)
    assert (tmp_path / "h.png").exists() and f["stations_reporting_snow_or_ice"] == 1


def test_tone_rules():
    assert captions.tone("quick_tag", {}) == "serious"
    assert captions.tone("anniversary", {}) == "serious"
    assert captions.tone("tropical", {"why_now": ["major_hurricane_near_us"]}) == "serious"
    assert captions.tone("hazard", {"hazard": "Extreme Cold Warnings"}) in ("dry", "straight")


# --- storm animations ----------------------------------------------------------------
HURDAT = """AL992005,            TESTY,      5,
20050823, 1800,  , TD, 23.1N,  75.1W,  30, 1008,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -999
20050824, 0000,  , TS, 24.0N,  76.0W,  50,  995,   60,   60,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -999
20050824, 0600,  , HU, 25.0N,  77.5W,  90,  960,  100,   90,   80,   90,   50,   50,   40,   40,   30,   30,   20,   20, -999
20050824, 0930, L, HU, 25.5N,  78.5W,  95,  955, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999,   20
20050824, 1200,  , HU, 26.0N,  79.5W,  80,  965,  100,   90,   80,   90,   50,   50,   40,   40,   30,   30,   20,   20, -999
"""

ATCF = """EP, 18, 2026100500,   , BEST,   0, 180N, 1150W,  40, 1000, TS,  34, NEQ,   60,   50,   40,   60, 1010,  200,  30,  50,   0,   E,   0,    ,   0,   0,     INVEST,
EP, 18, 2026100506,   , BEST,   0, 185N, 1160W,  70,  985, HU,  34, NEQ,   90,   80,   60,   80, 1010,  200,  20,  85,   0,   E,   0,    ,   0,   0,     RACHEL,
EP, 18, 2026100506,   , BEST,   0, 185N, 1160W,  70,  985, HU,  64, NEQ,   20,   15,   10,   20, 1010,  200,  20,  85,   0,   E,   0,    ,   0,   0,     RACHEL,
EP, 18, 2026100512,   , BEST,   0, 190N, 1170W,  90,  970, HU,  34, NEQ,  100,   90,   70,   90, 1010,  200,  15, 105,   0,   E,   0,    ,   0,   0,     RACHEL,
"""


def test_track_parsers():
    from studio.products import storm_anim as sa
    tr = sa.parse_hurdat(HURDAT, name="TESTY", year=2005)
    assert tr["id"] == "AL992005" and len(tr["points"]) == 5 and tr["points"][3]["rec"] == "L"
    assert tr["points"][0]["lon"] == -75.1
    tr = sa.parse_atcf(ATCF)
    assert tr["name"] == "Rachel" and len(tr["points"]) == 3
    assert tr["points"][1]["radii"][0] == 90 and tr["points"][1]["radii"][8] == 20 and tr["points"][1]["lat"] == 18.5
    assert sa.goes_bucket(datetime(2026, 1, 1, tzinfo=timezone.utc), -80) == "noaa-goes19"
    assert sa.goes_bucket(datetime(2021, 1, 1, tzinfo=timezone.utc), -130) == "noaa-goes17"
    assert sa.category(140) == 5 and sa.category(50, "EX") == "EX" and sa.mph(150) == 175


def test_goes_navigation_round_trip():
    """Build a fake GOES fixed grid, mark one lat/lon, and make sure sampling finds it."""
    import xarray as xr
    from studio.products import storm_anim as sa
    xs = np.linspace(-0.151844, 0.151844, 1500)
    ys = np.linspace(0.151844, -0.151844, 1500)
    attrs = {"semi_major_axis": 6378137.0, "semi_minor_axis": 6356752.31414, "perspective_point_height": 35786023.0,
             "longitude_of_projection_origin": -75.2}
    lat, lon = 25.0, -80.0
    # forward-navigate the target point to find its pixel, put a cold spot there
    one = sa.goes_to_latlon(xr.Dataset({"CMI": (("y", "x"), np.arange(1500 * 1500, dtype=np.float32).reshape(1500, 1500)),
                                        "goes_imager_projection": ((), 0, attrs)}, coords={"x": xs, "y": ys}),
                            np.array([[lat]]), np.array([[lon]]))
    idx = int(one[0, 0] + 273.15)
    cmi = np.full((1500, 1500), 300.0, np.float32)
    cmi.flat[idx] = 190.0
    ds = xr.Dataset({"CMI": (("y", "x"), cmi), "goes_imager_projection": ((), 0, attrs)}, coords={"x": xs, "y": ys})
    LA, LO = np.meshgrid(np.array([24.9, 25.0, 25.1]), np.array([-80.1, -80.0, -79.9]), indexing="ij")
    out = sa.goes_to_latlon(ds, LA, LO)
    assert out[1, 1] < -80 and out[0, 0] > 20
    far = sa.goes_to_latlon(ds, np.array([[10.0]]), np.array([[100.0]]))   # other side of the planet
    assert np.isnan(far[0, 0])


def test_storm_animation_renders(tmp_path, monkeypatch):
    from studio.products import storm_anim as sa
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    tr = sa.parse_hurdat(HURDAT, storm_id="AL992005")
    lats, lons = np.arange(10, 40, 0.1), np.arange(-95, -60, 0.1)
    LO, LA = np.meshgrid(lons, lats)
    ir = [(datetime(2005, 8, 23, 18, tzinfo=timezone.utc) + timedelta(hours=3 * i),
           (20 - 100 * np.exp(-(((LO + 75 + i) / 3) ** 2 + ((LA - 23 - i * 0.4) / 3) ** 2))).astype(np.float32),
           lats, lons) for i in range(8)]
    f = sa.build(tmp_path / "a.mp4", track=tr, ir_frames=ir, max_frames=12)
    assert (tmp_path / "a.mp4").stat().st_size > 10000
    assert f["peak"].startswith("110 mph") and f["landfalls"] and f["peak_category"] == 2


def test_recap_when_hurricane_dissipates():
    a, st = tropical.decide([storm(90)], {}, NOW)
    a, st = tropical.decide([], st, NOW + timedelta(hours=6))
    assert a == [{"type": "storm_recap", "id": "AL092026", "name": "Ida", "max_kt": 90}]
    a, st = tropical.decide([storm(45, sid="AL102026")], {}, NOW)
    a, st = tropical.decide([], st, NOW + timedelta(hours=6))
    assert a == []                                    # tropical storms don't get a recap by default


def test_anniversaries():
    from datetime import date
    from studio.__main__ import anniversaries_due
    due = anniversaries_due(date(2026, 8, 24), HURDAT, storms=[("TESTY", 2005)])
    assert due and due[0]["years_ago"] == 21 and due[0]["landfall"]
    assert anniversaries_due(date(2026, 8, 25), HURDAT, storms=[("TESTY", 2005)]) == []
