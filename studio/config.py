"""Central configuration. Everything tunable lives here or in environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(os.getenv("OUT_DIR") or ROOT / "out")
STATE_FILE = Path(os.getenv("STATE_FILE") or ROOT / ".state" / "state.json")
CACHE_DIR = Path(os.getenv("CACHE_DIR") or ROOT / ".cache")

BRAND = os.getenv("BOT_BRAND") or "Weather Map Studio"
HANDLE = os.getenv("BOT_HANDLE") or ""
DISPLAY_TZ = os.getenv("DISPLAY_TZ") or "America/Chicago"
CONTACT = os.getenv("NWS_CONTACT") or "weatherbot-contact@example.com"
USER_AGENT = f"({BRAND} X bot, {CONTACT})"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5-20251001"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"


def _i(name, default):
    return int(os.getenv(name) or default)


# --- Data sources (NOAA / NWS / Iowa Environmental Mesonet) ---------------
NWS_ALERTS = "https://api.weather.gov/alerts/active?status=actual"
IEM_RADAR = "https://mesonet.agron.iastate.edu/archive/data/{t:%Y/%m/%d}/GIS/uscomp/n0q_{t:%Y%m%d%H%M}.png"
# n0q composite geometry: upper-left corner and degrees per pixel
RADAR_UL = (-126.0, 50.0)
RADAR_RES = 0.005
IEM_SBW = "https://mesonet.agron.iastate.edu/api/1/vtec/sbw_interval.geojson"
IEM_LSR = "https://mesonet.agron.iastate.edu/geojson/lsr.php"
NHC_STORMS = "https://www.nhc.noaa.gov/CurrentStorms.json"
NHC_CONE = "https://www.nhc.noaa.gov/storm_graphics/{bin}/{ID}_5day_cone_with_line_and_wind.png"
GOES_FLOATER = "https://cdn.star.nesdis.noaa.gov/FLOATER/data/{ID}/GEOCOLOR/"
GOES_SECTOR = "https://cdn.star.nesdis.noaa.gov/{sat}/ABI/SECTOR/{sector}/GEOCOLOR/"
GFS_NOMADS = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{d}%2F{h}%2Fatmos"
              "&file=gfs.t{h}z.pgrb2.0p25.f000&var_HGT=on&var_TMP=on&var_UGRD=on&var_VGRD=on"
              "&lev_250_mb=on&lev_500_mb=on&lev_850_mb=on"
              "&subregion=&toplat=75&leftlon=190&rightlon=330&bottomlat=10")
GFS_AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f000"
PSL_LTM = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/{var}.mon.ltm.1991-2020.nc"

# --- Severe weather triggers ---------------------------------------------
# An "event" starts when a cluster of nearby warnings reaches either count.
EVENT_TOR_COUNT = _i("EVENT_TOR_COUNT", 4)          # tornado warnings in one cluster
EVENT_ANY_COUNT = _i("EVENT_ANY_COUNT", 15)         # tornado + severe t-storm + flash flood warnings
CLUSTER_LINK_DEG = float(os.getenv("CLUSTER_LINK_DEG") or 3.0)   # warnings this close join a cluster
EVENT_UPDATE_GAP_MIN = _i("EVENT_UPDATE_GAP_MIN", 90)  # min minutes between in-event update loops
EVENT_END_QUIET_MIN = _i("EVENT_END_QUIET_MIN", 90)    # quiet this long -> event over -> summary
EVENT_END_COUNT = _i("EVENT_END_COUNT", 2)             # "quiet" = fewer warnings than this
EVENT_MAX_HOURS = _i("EVENT_MAX_HOURS", 14)            # summary covers at most this window
LOOP_MINUTES = _i("LOOP_MINUTES", 90)                  # length of live loops
SUMMARY_FRAMES = _i("SUMMARY_FRAMES", 72)              # frames in the end-of-event time lapse

# Tropical triggers
TROPICAL_NEW_MIN = os.getenv("TROPICAL_NEW_MIN") or "TS"   # post new storms once they reach this
RI_KT_24H = _i("RI_KT_24H", 30)                              # rapid intensification threshold
TROPICAL_SATELLITE_FRAMES = _i("TROPICAL_SATELLITE_FRAMES", 36)
RECAP_MIN_KT = _i("RECAP_MIN_KT", 64)        # life-recap animation when a storm this strong dissipates

# "Years ago today" animations for landmark storms (name, year). Posted on the anniversary of
# their strongest landfall (from HURDAT2). Edit freely.
ANNIVERSARY_STORMS = [
    ("ANDREW", 1992), ("HUGO", 1989), ("OPAL", 1995), ("FRAN", 1996), ("GEORGES", 1998), ("MITCH", 1998),
    ("FLOYD", 1999), ("ISABEL", 2003), ("CHARLEY", 2004), ("FRANCES", 2004), ("IVAN", 2004), ("JEANNE", 2004),
    ("DENNIS", 2005), ("KATRINA", 2005), ("RITA", 2005), ("WILMA", 2005), ("GUSTAV", 2008), ("IKE", 2008),
    ("IRENE", 2011), ("SANDY", 2012), ("MATTHEW", 2016), ("HARVEY", 2017), ("IRMA", 2017), ("MARIA", 2017),
    ("FLORENCE", 2018), ("MICHAEL", 2018), ("DORIAN", 2019), ("LAURA", 2020), ("DELTA", 2020), ("ETA", 2020),
    ("IDA", 2021), ("IAN", 2022), ("IDALIA", 2023), ("BERYL", 2024), ("HELENE", 2024), ("MILTON", 2024),
]

# Upper air: always post the 12Z analysis; post 00Z only if anomalies are this large
UPPER_NOTABLE_HGT_DAM = _i("UPPER_NOTABLE_HGT_DAM", 24)   # 500 mb height anomaly (decameters)
UPPER_NOTABLE_T850_C = _i("UPPER_NOTABLE_T850_C", 14)     # 850 mb temperature anomaly (C)
