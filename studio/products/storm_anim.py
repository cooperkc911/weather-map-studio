"""Storm-following hurricane animation: infrared satellite + best track + intensity markers + wind radii.

Works for any Atlantic / East Pacific storm:
  * track:     NHC HURDAT2 (historical, has landfall flags) or the live ATCF best track (b-deck)
  * infrared:  GOES-16/17/18/19 band 13 from NOAA's AWS bucket (2018+), else NCEI GridSat-B1 (1983+)
"""
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import config, geo, net

UTC = timezone.utc
W, H, DPI = 19.2, 10.8, 100
FPS = 24
CAT_COLOR = {"TD": "#5ebaff", "TS": "#00faf4", 1: "#ffffcc", 2: "#ffe775", 3: "#ffc140", 4: "#ff8f20",
             5: "#ff6060", "EX": "#bbbbbb"}
CAT_NAME = {"TD": "Tropical Depression", "TS": "Tropical Storm", "EX": "Post-Tropical Low",
            1: "Category 1 Hurricane", 2: "Category 2 Hurricane", 3: "Category 3 Hurricane",
            4: "Category 4 Hurricane", 5: "Category 5 Hurricane"}
NON_TROPICAL = {"EX", "LO", "WV", "DB", "PT"}


def category(kt, status="HU"):
    if status in NON_TROPICAL:
        return "EX"
    for c, lo in ((5, 137), (4, 113), (3, 96), (2, 83), (1, 64)):
        if kt >= lo:
            return c
    return "TS" if kt >= 34 else "TD"


def mph(kt):
    return int(round(kt * 1.15078 / 5) * 5)


# =========================================================== tracks
HURDAT_INDEX = "https://www.nhc.noaa.gov/data/hurdat/"
ATCF_BTK = "https://ftp.nhc.noaa.gov/atcf/btk/b{basin}{num}{year}.dat"
ATCF_ARCHIVE = "https://ftp.nhc.noaa.gov/atcf/archive/{year}/b{basin}{num}{year}.dat.gz"


def _hurdat_file(basin="AL"):
    """Download (and cache) the newest HURDAT2 file for the basin."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pat = r"hurdat2-1851-[\d-]+\.txt" if basin == "AL" else r"hurdat2-nepac-[\d-]+\.txt"
    names = re.findall(pat, net.get(HURDAT_INDEX).text)
    if not names:
        raise RuntimeError("no HURDAT2 file found")
    # newest data year wins; within a year, the last one listed
    best = max(enumerate(names), key=lambda kv: (re.findall(r"-(\d{4})-", kv[1])[-1], kv[0]))[1]
    path = config.CACHE_DIR / best
    if not path.exists():
        path.write_bytes(net.get(HURDAT_INDEX + best, timeout=120).content)
    return path


def _ll(s):
    v = float(s[:-1])
    return -v if s[-1] in "SW" else v


def parse_hurdat(text, storm_id=None, name=None, year=None):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        head = [x.strip() for x in lines[i].split(",")]
        n = int(head[2])
        sid, sname = head[0], head[1]
        if (storm_id and sid == storm_id.upper()) or (name and year and sname == name.upper() and sid[4:8] == str(year)):
            pts = []
            for ln in lines[i + 1:i + 1 + n]:
                f = [x.strip() for x in ln.split(",")]
                pts.append({"t": datetime.strptime(f[0] + f[1], "%Y%m%d%H%M").replace(tzinfo=UTC),
                            "rec": f[2], "status": f[3], "lat": _ll(f[4]), "lon": _ll(f[5]),
                            "kt": int(f[6]), "mb": int(f[7]),
                            "radii": [int(x) for x in f[8:20]] if len(f) >= 20 else [-999] * 12})
            return {"id": sid, "name": sname.title(), "points": pts}
        i += n + 1
    raise KeyError(f"storm not found: {storm_id or (name, year)}")


def parse_atcf(text, name=None):
    """ATCF b-deck: one row per time per wind-radii threshold (34/50/64 kt)."""
    by_t = {}
    auto = name is None
    for ln in text.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 17 or f[4] != "BEST":
            continue
        t = datetime.strptime(f[2], "%Y%m%d%H").replace(tzinfo=UTC)
        lat = int(f[6][:-1]) / 10 * (1 if f[6][-1] == "N" else -1)
        lon = int(f[7][:-1]) / 10 * (-1 if f[7][-1] == "W" else 1)
        p = by_t.setdefault(t, {"t": t, "rec": "", "status": f[10], "lat": lat, "lon": lon, "kt": int(f[8] or 0),
                                "mb": int(f[9] or 0), "radii": [0] * 12})
        rad = int(f[11] or 0) if f[11].isdigit() else 0
        if rad in (34, 50, 64):
            k = {34: 0, 50: 4, 64: 8}[rad]
            p["radii"][k:k + 4] = [int(x or 0) for x in f[13:17]]
        if len(f) > 27 and f[27] and not re.match(r"(GENESIS|INVEST|NONAME|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)", f[27]):
            found = f[27]
            name = name if name and not auto else found
    pts = [by_t[t] for t in sorted(by_t)]
    # trim the pre-genesis disturbance and post-tropical tail beyond 2 days
    first = next((i for i, p in enumerate(pts) if p["status"] in ("TD", "TS", "HU", "SD", "SS")), 0)
    pts = pts[max(0, first - 1):]
    return {"name": (name or "Storm").title(), "points": pts}


def load_track(storm_id=None, name=None, year=None, live=False):
    """storm_id like AL122005 / EP182026. live=True reads the ATCF b-deck (current storms)."""
    if live or (storm_id and int(storm_id[-4:]) >= datetime.now(UTC).year):
        basin, num, yr = storm_id[:2].lower(), storm_id[2:4], storm_id[4:]
        try:
            text = net.get(ATCF_BTK.format(basin=basin, num=num, year=yr)).text
        except Exception:  # noqa: BLE001
            import gzip
            text = gzip.decompress(net.get(ATCF_ARCHIVE.format(basin=basin, num=num, year=yr)).content).decode()
        tr = parse_atcf(text, name)
        tr["id"] = storm_id.upper()
        tr["active"] = datetime.now(UTC) - tr["points"][-1]["t"] < timedelta(hours=12) and \
            tr["points"][-1]["status"] in ("TD", "TS", "HU", "SD", "SS")
        return tr
    basin = (storm_id or "AL")[:2].upper()
    return parse_hurdat(_hurdat_file("AL" if basin == "AL" else "EP").read_text(), storm_id, name, year)


# =========================================================== infrared
IR_DIR_NAME = "ir_frames"
GRIDSAT = "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/{y}/"


def _gridsat_one(t, box, out):
    fn = f"GRIDSAT-B1.{t:%Y.%m.%d.%H}.v02r01.nc"
    tmp = out.parent / fn
    subprocess.run(["curl", "-sSf", "-m", "300", "-o", str(tmp), GRIDSAT.format(y=t.year) + fn], check=True)
    return tmp


def fetch_gridsat(times, box, workdir):
    import xarray as xr
    w, e, s, n = box
    workdir.mkdir(parents=True, exist_ok=True)
    out = []
    with ThreadPoolExecutor(4) as ex:                 # download in parallel ...
        files = list(ex.map(lambda t: _try(_gridsat_one, t, box, workdir / f"{t:%Y%m%d%H}.npz"), times))
    for t, tmp in zip(times, files):                  # ... decode one at a time (HDF5 is not thread-safe)
        if not tmp:
            continue
        try:
            ds = xr.open_dataset(tmp)
            sub = ds.irwin_cdr.isel(time=0).sel(lat=slice(s, n), lon=slice(w, e))
            out.append((t, sub.values.astype(np.float32) - 273.15, sub.lat.values, sub.lon.values))
            ds.close()
        except Exception as ex:  # noqa: BLE001
            print(f"[ir] gridsat {t}: {ex}")
        finally:
            os.remove(tmp)
    return out


def _try(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        print(f"[ir] {a[0]}: {e}")
        return None


def goes_bucket(t, lon):
    west = lon < -115 or lon > 150
    if west:
        if t < datetime(2019, 2, 12, tzinfo=UTC):
            return "noaa-goes16"               # GOES-17 not yet operational: use East
        return "noaa-goes17" if t < datetime(2023, 1, 4, tzinfo=UTC) else "noaa-goes18"
    return "noaa-goes16" if t < datetime(2025, 4, 7, tzinfo=UTC) else "noaa-goes19"


def _goes_key(bucket, t):
    prefix = f"ABI-L2-CMIPF/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
    xml = net.get(f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}").text
    keys = [k for k in re.findall(r"<Key>([^<]+)</Key>", xml) if "M6C13" in k or "M3C13" in k or "M4C13" in k]
    return keys[0] if keys else None


def goes_to_latlon(ds, lat2d, lon2d):
    """Sample a GOES fixed-grid CMI image at lat/lon points (GOES-R PUG navigation equations)."""
    pr = ds["goes_imager_projection"].attrs
    a, b = pr["semi_major_axis"], pr["semi_minor_axis"]
    Hh = pr["perspective_point_height"] + a
    lon0 = math.radians(pr["longitude_of_projection_origin"])
    phi, lam = np.radians(lat2d), np.radians(lon2d)
    phic = np.arctan((b * b) / (a * a) * np.tan(phi))
    e2 = (a * a - b * b) / (a * a)
    rc = b / np.sqrt(1 - e2 * np.cos(phic) ** 2)
    sx = Hh - rc * np.cos(phic) * np.cos(lam - lon0)
    sy = -rc * np.cos(phic) * np.sin(lam - lon0)
    sz = rc * np.sin(phic)
    visible = Hh * (Hh - sx) >= sy ** 2 + (a * a) / (b * b) * sz ** 2
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    y = np.arctan(sz / sx)
    xs, ys = ds["x"].values, ds["y"].values                 # radians, x increasing, y decreasing
    ix = np.clip(np.round((x - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((y - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1)
    # read only the window we need
    x0, x1, y0, y1 = ix.min(), ix.max() + 1, iy.min(), iy.max() + 1
    win = ds["CMI"].isel(x=slice(x0, x1), y=slice(y0, y1)).values.astype(np.float32)
    out = win[iy - y0, ix - x0]
    out[~visible] = np.nan
    return out - 273.15


def fetch_goes(times, box, lon_c, workdir, res=0.04):
    import xarray as xr
    w, e, s, n = box
    workdir.mkdir(parents=True, exist_ok=True)
    lats = np.arange(s, n + res / 2, res)
    lons = np.arange(w, e + res / 2, res)
    LON, LAT = np.meshgrid(lons, lats)

    def dl(t):
        bucket = goes_bucket(t, lon_c)
        key = _goes_key(bucket, t)
        if not key:
            return None
        p = workdir / os.path.basename(key)
        subprocess.run(["curl", "-sSf", "-m", "300", "-o", str(p), f"https://{bucket}.s3.amazonaws.com/{key}"],
                       check=True)
        return p
    with ThreadPoolExecutor(4) as ex:
        files = list(ex.map(lambda t: _try(dl, t), times))
    out = []
    for t, p in zip(times, files):
        if not p:
            continue
        try:
            ds = xr.open_dataset(p)
            out.append((t, goes_to_latlon(ds, LAT, LON), lats, lons))
            ds.close()
        except Exception as ex:  # noqa: BLE001
            print(f"[ir] goes {t}: {ex}")
        finally:
            os.remove(p)
    return out


def fetch_ir(track, box, workdir, step_h=3):
    pts = track["points"]
    t0 = pts[0]["t"].replace(minute=0) - timedelta(hours=pts[0]["t"].hour % step_h)
    times = []
    t = t0
    while t <= pts[-1]["t"] + timedelta(hours=step_h):
        times.append(t)
        t += timedelta(hours=step_h)
    workdir.mkdir(parents=True, exist_ok=True)
    lon_c = float(np.median([p["lon"] for p in pts]))
    frames = []
    if t0.year >= 2018:
        frames = fetch_goes(times, box, lon_c, workdir)
    if len(frames) < len(times) * 0.6 and t0.year >= 1983:
        frames = fetch_gridsat(times, box, workdir) or frames
    if len(frames) < 3:
        raise RuntimeError("not enough infrared imagery for this storm")
    return frames


# =========================================================== renderer
IR_VMIN, IR_VMAX = -90, 40


def _ir_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    def c(tc):
        return (tc - IR_VMIN) / (IR_VMAX - IR_VMIN)
    return LinearSegmentedColormap.from_list("enhir", [
        (0.0, "#ffffff"), (c(-85), "#e8e8e8"), (c(-80), "#ff00ff"), (c(-75), "#c00000"), (c(-70), "#ff2a00"),
        (c(-64), "#ffaa00"), (c(-58), "#ffff00"), (c(-52), "#00c000"), (c(-44), "#0050ff"),
        (c(-34), "#9fd3ff"), (c(-30), "#d8d8d8"), (c(0), "#7a7a7a"), (c(20), "#2b2b2b"), (1.0, "#0c0c0c")])


class Storm:
    def __init__(self, track, ir_frames):
        from scipy.interpolate import PchipInterpolator
        self.name = track["name"]
        self.id = track.get("id", "")
        self.pts = track["points"]
        self.T0, self.T1 = self.pts[0]["t"], self.pts[-1]["t"]
        self.syn = [p for p in self.pts if p["t"].minute == 0 and p["t"].hour % 6 == 0]
        self.landfalls = [p for p in self.pts if p["rec"] == "L"]
        self.peak = max(self.pts, key=lambda p: (p["kt"], -p["mb"]))
        self.secs = np.array([(p["t"] - self.T0).total_seconds() for p in self.pts])
        self.f_lat = PchipInterpolator(self.secs, [p["lat"] for p in self.pts])
        self.f_lon = PchipInterpolator(self.secs, [p["lon"] for p in self.pts])
        rad = [p for p in self.pts if p["radii"][0] not in (-999,)]
        self.rsecs = np.array([(p["t"] - self.T0).total_seconds() for p in rad]) if rad else None
        self.rad = np.array([p["radii"] for p in rad], float) if rad else None
        self.ir = sorted(ir_frames, key=lambda f: f[0])
        self.active = track.get("active", False)
        self.cmap = _ir_cmap()
        basin = self.id[:2]
        self.basin = {"AL": "Atlantic", "EP": "Eastern Pacific", "CP": "Central Pacific"}.get(basin, "")
        self.kind = ("Hurricane" if self.peak["kt"] >= 64 else "Tropical Storm" if self.peak["kt"] >= 34
                     else "Tropical Depression")
        self.landfall_text = [describe_landfall(p["lat"], p["lon"]) for p in self.landfalls]

    # --- state
    def at(self, t):
        t = min(max(t, self.T0), self.T1)
        s = (t - self.T0).total_seconds()
        kt = float(np.interp(s, self.secs, [p["kt"] for p in self.pts]))
        mb = float(np.interp(s, self.secs, [p["mb"] for p in self.pts]))
        radii = (np.array([np.interp(s, self.rsecs, self.rad[:, i]) for i in range(12)])
                 if self.rad is not None and len(self.rad) else np.zeros(12))
        status = next(p["status"] for p in reversed(self.pts) if p["t"] <= t)
        return {"lat": float(self.f_lat(s)), "lon": float(self.f_lon(s)), "kt": kt, "mb": mb,
                "radii": radii, "status": status}

    def ir_at(self, t):
        times = [f[0] for f in self.ir]
        j = max(0, min(len(times) - 2, int(np.searchsorted(times, t, side="right")) - 1))
        span = (times[j + 1] - times[j]).total_seconds() or 1
        a = min(max((t - times[j]).total_seconds() / span, 0), 1)
        f0, f1 = self.ir[j], self.ir[j + 1]
        if f0[1].shape == f1[1].shape:
            tb = (1 - a) * f0[1] + a * f1[1]
        else:
            tb = f0[1] if a < 0.5 else f1[1]
        f = f0 if a < 0.5 else f1
        return tb, f[2], f[3], f[0]

    # --- camera
    def follow(self, t):
        xs, ys = [], []
        for dh in np.linspace(-5, 5, 11):
            s = self.at(t + timedelta(hours=float(dh)))
            xs.append(s["lon"]); ys.append(s["lat"])   # noqa: E702
        return float(np.mean(xs)), float(np.mean(ys))

    @staticmethod
    def view(cx, cy, width):
        k = math.cos(math.radians(cy))
        h = width * k * H / W
        return (cx - width / 2, cx + width / 2, cy - h / 2, cy + h / 2)

    # areas of the frame covered by panels (x0, y0, x1, y1 as fractions of the frame)
    PANELS = [(0, 0.735, 0.32, 1), (0.585, 0, 1, 0.345), (0, 0, 1, 0.08), (0, 0, 0.2, 0.2)]

    def full_view(self):
        """Smallest view that shows the whole track without any point hidden under a panel."""
        lons = np.array([p["lon"] for p in self.pts])
        lats = np.array([p["lat"] for p in self.pts])
        cx0, cy0 = (lons.min() + lons.max()) / 2, (lats.min() + lats.max()) / 2
        k = math.cos(math.radians(cy0))
        minw = max(np.ptp(lons) + 1, np.ptp(lats) * W / H / k + 1, 8)
        for width in np.linspace(minw, minw * 4, 40):
            height = width * k * H / W
            ok = []
            for fy in np.linspace(-0.35, 0.35, 15):
                for fx in np.linspace(-0.35, 0.35, 15):
                    v = self.view(cx0 + fx * width, cy0 + fy * height, width)
                    x = (lons - v[0]) / (v[1] - v[0])
                    y = (lats - v[2]) / (v[3] - v[2])
                    # room for date labels to the right of points and marker size at the edges
                    if x.min() < 0.05 or x.max() > 0.91 or y.min() < 0.06 or y.max() > 0.93:
                        continue
                    if any(((x > x0 - 0.02) & (x < x1 + 0.06) & (y > y0 - 0.03) & (y < y1 + 0.03)).any()
                           for x0, y0, x1, y1 in self.PANELS):
                        continue
                    ok.append((abs(fx) + abs(fy), v))
            if ok:
                return min(ok, key=lambda o: o[0])[1]
        return self.view(cx0, cy0, minw * 2)

    def schedule(self, view_w=21.0, target_frames=560):
        dur = (self.T1 - self.T0).total_seconds() / 60
        step = max(15, dur / target_frames)
        times = [self.T0 + timedelta(minutes=step * i) for i in range(int(dur // step) + 1)]
        fr, i = [], 0
        first = self.view(*self.follow(self.T0), view_w)
        for _ in range(int(FPS * 1.5)):
            fr.append((i, self.T0, first, "hold")); i += 1   # noqa: E702
        for t in times:
            fr.append((i, t, self.view(*self.follow(t), view_w), "follow")); i += 1   # noqa: E702
        last, full = fr[-1][2], self.full_view()
        n = int(FPS * 2.5)
        for k in range(1, n + 1):
            x = k / n
            x = x * x * (3 - 2 * x)
            fr.append((i, self.T1, tuple(last[j] + (full[j] - last[j]) * x for j in range(4)), "zoom")); i += 1  # noqa: E702
        for _ in range(int(FPS * 4)):
            fr.append((i, self.T1 + timedelta(seconds=1), full, "end")); i += 1   # noqa: E702
        return fr


_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def describe_landfall(lat, lon):
    """'near Slidell, LA' when close, otherwise '45 mi SE of New Orleans, LA'."""
    near, d = geo.nearest_place(lat, lon, min_pop=20000)
    if near and d is not None and d <= 15:
        return f"near {near}"
    cands = geo.places(150000)
    if not cands:
        return f"{lat:.1f}N {abs(lon):.1f}W"
    name, plat, plon, _, adm = min(cands, key=lambda p: geo.haversine_mi(lat, lon, p[1], p[2]))
    bd = round(geo.haversine_mi(lat, lon, plat, plon))
    st = geo.US_STATES.get(adm, adm)
    big = f"{name}, {st}" if st else name
    brg = (math.degrees(math.atan2((lon - plon) * math.cos(math.radians(lat)), lat - plat)) + 360) % 360
    return f"{bd} mi {_DIRS[int((brg + 11.25) // 22.5) % 16]} of {big}"


def _radii_polygon(lat, lon, r4):
    pts = []
    for q, (a0, a1) in enumerate(((0, 90), (90, 180), (180, 270), (270, 360))):
        for ang in np.linspace(a0, a1, 16):
            dy = r4[q] / 60 * math.cos(math.radians(ang))
            dx = r4[q] / 60 * math.sin(math.radians(ang)) / math.cos(math.radians(lat))
            pts.append((lon + dx, lat + dy))
    pts.append(pts[0])
    return np.array(pts)


def _glyph(ax, x, y, size, color, rot, hurricane=True):
    from matplotlib import patheffects
    pe = [patheffects.withStroke(linewidth=4, foreground="black")]
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(x + size * 0.38 * np.cos(th), y + size * 0.38 * np.sin(th), color=color, lw=3.2, zorder=20, path_effects=pe)
    if hurricane:
        for arm in (0, np.pi):
            a = np.linspace(0, 1.25, 25)
            r = size * (0.38 + 0.62 * a / 1.25)
            ang = rot + arm + a * 1.9
            ax.plot(x + r * np.cos(ang), y + r * np.sin(ang), color=color, lw=3.6, zorder=20,
                    solid_capstyle="round", path_effects=pe)


_STORM = None     # set before forking the worker pool
_LINES = None


def render_frame(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.lines import Line2D
    idx, t, view, phase, outdir = args
    S = _STORM
    end = t > S.T1
    s = S.at(t)
    cat = category(s["kt"], s["status"])
    w, e, so, n = view
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(w, e); ax.set_ylim(so, n); ax.axis("off")   # noqa: E702
    tb, lats, lons, irt = S.ir_at(min(t, S.T1))
    ax.imshow(tb, extent=(lons[0], lons[-1], lats[0], lats[-1]), origin="lower", cmap=S.cmap, vmin=IR_VMIN,
              vmax=IR_VMAX, interpolation="bicubic", aspect="auto", zorder=1)
    for ln in _LINES:
        a = np.array(ln)
        if a[:, 0].max() < w - 2 or a[:, 0].min() > e + 2 or a[:, 1].max() < so - 2 or a[:, 1].min() > n + 2:
            continue
        ax.plot(a[:, 0], a[:, 1], color="#e9f1ff", lw=0.9, alpha=0.75, zorder=3)
    pe = [patheffects.withStroke(linewidth=3, foreground="black")]
    ax.plot([p["lon"] for p in S.syn], [p["lat"] for p in S.syn], color="white", lw=1.2, ls=(0, (2, 3)),
            alpha=0.45, zorder=5)
    nseg = int((min(t, S.T1) - S.T0).total_seconds() // 900)
    st = [S.at(S.T0 + timedelta(minutes=15 * k)) for k in range(nseg + 1)] + [S.at(min(t, S.T1))]
    for a, b in zip(st[:-1], st[1:]):
        ax.plot([a["lon"], b["lon"]], [a["lat"], b["lat"]], color="black", lw=6.5, zorder=6, solid_capstyle="round")
        ax.plot([a["lon"], b["lon"]], [a["lat"], b["lat"]], color=CAT_COLOR[category(a["kt"], a["status"])],
                lw=3.8, zorder=7, solid_capstyle="round")
    min_sep = (e - w) * 0.016
    last_xy, last_lab = None, None
    for p in S.syn:
        if p["t"] > t:
            break
        if last_xy and math.hypot((p["lon"] - last_xy[0]) * 1.0, (p["lat"] - last_xy[1]) / math.cos(math.radians(p["lat"]))) < min_sep:
            continue
        last_xy = (p["lon"], p["lat"])
        c = category(p["kt"], p["status"])
        ax.scatter([p["lon"]], [p["lat"]], s=230, c=CAT_COLOR[c], edgecolors="black", linewidths=1.5, zorder=8)
        ax.text(p["lon"], p["lat"], {"TD": "D", "TS": "S", "EX": "X"}.get(c, str(c)), fontsize=9, weight="bold",
                ha="center", va="center", color="black", zorder=9)
        if p["t"].hour == 0 and (not last_lab or math.hypot(p["lon"] - last_lab[0], p["lat"] - last_lab[1]) > min_sep * 3):
            last_lab = (p["lon"], p["lat"])
            ax.text(p["lon"] + (e - w) * 0.012, p["lat"], p["t"].strftime("%b %-d"), fontsize=12, color="white",
                    va="center", zorder=9, path_effects=pe)
    prev = None
    for k, (p, txt) in enumerate(zip(S.landfalls, S.landfall_text)):
        if p["t"] <= t:
            ax.scatter([p["lon"]], [p["lat"]], marker="*", s=600, c="#ff2bd6", edgecolors="black", linewidths=1.5,
                       zorder=10)
            if end:
                crowded = prev is not None and abs(p["lat"] - prev[1]) < 1.6 and abs(p["lon"] - prev[0]) < 4
                side = -1 if crowded and prev[2] == 1 else 1           # alternate sides when stars are close
                c = category(p["kt"])
                ax.text(p["lon"] + side * (e - w) * 0.014, p["lat"],
                        f"{txt}\n{p['t']:%b %-d} · {'Cat ' + str(c) if isinstance(c, int) else c}",
                        fontsize=11, color="#ff9bee", weight="bold", va="center", ha="left" if side > 0 else "right",
                        zorder=12, path_effects=pe)
                prev = (p["lon"], p["lat"], side)
    pk = S.peak
    inv = w < pk["lon"] < e - (e - w) * 0.42 and so + (n - so) * 0.2 < pk["lat"] < n - (n - so) * 0.4
    if pk["t"] <= t and inv:
        ax.annotate(f"Peak {mph(pk['kt'])} mph · {pk['mb']} mb\n{pk['t']:%b %-d %H%MZ}", (pk["lon"], pk["lat"]),
                    xytext=(-150, -70), textcoords="offset points", fontsize=13, color="white", weight="bold",
                    zorder=12, path_effects=pe, arrowprops=dict(arrowstyle="-", color="white", lw=1.5))
    if not end:
        r = s["radii"]
        for k, col, lw in ((0, "#ffe14d", 1.6), (8, "#ff3030", 2.2)):
            if max(r[k:k + 4]) > 0:
                poly = _radii_polygon(s["lat"], s["lon"], r[k:k + 4])
                ax.plot(poly[:, 0], poly[:, 1], color=col, lw=lw, zorder=11,
                        path_effects=[patheffects.withStroke(linewidth=lw + 2, foreground="black")])
        size = (e - w) * (0.014 + 0.012 * min(s["kt"], 150) / 150)
        _glyph(ax, s["lon"], s["lat"], size, CAT_COLOR[cat], -(t - S.T0).total_seconds() / 3600 * 0.9,
               hurricane=cat not in ("TD", "EX"))
    # ---- overlay
    ov = fig.add_axes([0, 0, 1, 1], zorder=50)
    ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off"); ov.patch.set_alpha(0)   # noqa: E702
    ov.add_patch(plt.Rectangle((0.012, 0.755), 0.29, 0.225, color="#070b12", alpha=0.82))
    ov.text(0.024, 0.95, f"{S.kind.upper()} {S.name.upper()}", color="white", fontsize=26, weight="bold", va="center")
    ov.text(0.024, 0.913, f"{S.T0:%B %Y}" + (f" · {S.basin} Basin" if S.basin else ""), color="#aab4c3",
            fontsize=13, va="center")
    from zoneinfo import ZoneInfo
    loc = min(t, S.T1).astimezone(ZoneInfo(config.DISPLAY_TZ))
    ov.text(0.024, 0.873, ("Track so far (still active)" if S.active else "Full track: birth to death") if end else
            f"{loc:%a %b %-d, %-I:%M %p} {loc.tzname()}   ({min(t, S.T1):%H%MZ})", color="#ffd25a", fontsize=15,
            weight="bold", va="center")
    if end:
        c = category(pk["kt"])
        ov.text(0.024, 0.823, f"Peak: {mph(pk['kt'])} mph · {pk['mb']} mb" + (f" (Cat {c})" if isinstance(c, int) else ""),
                color="white", fontsize=17, weight="bold", va="center")
        days = (S.T1 - S.T0).total_seconds() / 86400
        lf = f"{len(S.landfalls)} landfall{'s' if len(S.landfalls) != 1 else ''} · " if S.landfalls else ""
        ov.text(0.024, 0.778, f"{lf}{days:.0f} days · {S.T0:%b %-d}–{S.T1:%b %-d}", color="white", fontsize=17,
                va="center")
    else:
        ov.add_patch(plt.Rectangle((0.024, 0.808), 0.012, 0.03, color=CAT_COLOR[cat]))
        ov.text(0.042, 0.823, CAT_NAME[cat], color="white", fontsize=17, weight="bold", va="center")
        ov.text(0.024, 0.778, f"{mph(s['kt'])} mph  ·  {round(s['kt'] / 5) * 5:.0f} kt  ·  {s['mb']:.0f} mb",
                color="white", fontsize=17, va="center")
    shown = [(p, txt) for p, txt in zip(S.landfalls, S.landfall_text) if p["t"] <= t] if not end else []
    shown = shown[-4:]
    if shown:
        ov.add_patch(plt.Rectangle((0.012, 0.75 - 0.032 * len(shown) - 0.008), 0.42, 0.032 * len(shown) + 0.006,
                                   color="#070b12", alpha=0.82))
    y = 0.732
    for k, (p, txt) in enumerate(shown):
        c = category(p["kt"])
        ov.text(0.02, y, f"★ Landfall: {txt} · {p['t']:%b %-d %H%MZ} · "
                f"{'Cat ' + str(c) if isinstance(c, int) else c}, {mph(p['kt'])} mph",
                color="#ff8be9", fontsize=12.5, weight="bold", va="center", path_effects=pe)
        y -= 0.032
    # intensity chart
    ov.add_patch(plt.Rectangle((0.600, 0.083), 0.392, 0.245, color="#070b12", alpha=0.86))
    ov.text(0.612, 0.307, "Max sustained wind (mph)", color="white", fontsize=13, weight="bold", va="center")
    cax = fig.add_axes([0.625, 0.115, 0.36, 0.17], zorder=60)
    cax.patch.set_alpha(0)
    for lo, hi, c in ((0, 39, "TD"), (39, 74, "TS"), (74, 96, 1), (96, 111, 2), (111, 130, 3), (130, 157, 4),
                      (157, 220, 5)):
        cax.axhspan(lo, hi, color=CAT_COLOR[c], alpha=0.16, lw=0)
    ct = [(p["t"] - S.T0).total_seconds() / 86400 for p in S.pts]
    cv = [mph(p["kt"]) for p in S.pts]
    cax.plot(ct, cv, color="white", lw=1.0, alpha=0.35)
    nd = (min(t, S.T1) - S.T0).total_seconds() / 86400
    m = [i for i, x in enumerate(ct) if x <= nd]
    cax.plot([ct[i] for i in m] + [nd], [cv[i] for i in m] + [mph(s["kt"])], color="#ffd25a", lw=2.6)
    cax.scatter([nd], [mph(s["kt"])], s=60, c=CAT_COLOR[cat], edgecolors="black", zorder=5)
    for p in S.landfalls:
        if p["t"] <= t:
            cax.axvline((p["t"] - S.T0).total_seconds() / 86400, color="#ff2bd6", lw=1.2, ls="--")
    days = (S.T1 - S.T0).total_seconds() / 86400
    cax.set_xlim(0, days); cax.set_ylim(20, max(190, max(cv) + 10))   # noqa: E702
    d0 = (S.T0 + timedelta(days=1)).replace(hour=0, minute=0)
    stepd = max(1, int(days // 9) + 1)
    ticks = [d0 + timedelta(days=k) for k in range(0, int(days) + 1, stepd) if d0 + timedelta(days=k) <= S.T1]
    cax.set_xticks([(x - S.T0).total_seconds() / 86400 for x in ticks])
    cax.set_xticklabels([f"{x:%-d}" for x in ticks], color="#c8d0dc", fontsize=10)
    cax.set_xlabel(f"{S.T0:%B %Y}", color="#c8d0dc", fontsize=10, labelpad=1)
    cax.set_yticks([40, 75, 111, 157]); cax.set_yticklabels(["40", "75", "111", "157"], color="#c8d0dc", fontsize=10)  # noqa: E702
    cax.tick_params(length=2, pad=2)
    for sp in cax.spines.values():
        sp.set_visible(False)
    hs = [Line2D([], [], marker="o", ls="", ms=11, mfc=CAT_COLOR[c], mec="black", label=lab)
          for c, lab in (("TD", "Dep."), ("TS", "Storm"), (1, "Cat 1"), (2, "Cat 2"), (3, "Cat 3"), (4, "Cat 4"),
                         (5, "Cat 5"))]
    hs += [Line2D([], [], color="#ffe14d", lw=2.5, label="34 kt winds"),
           Line2D([], [], color="#ff3030", lw=2.5, label="64 kt winds")]
    leg = ov.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.008, 0.08), ncol=3, fontsize=11.5, frameon=True,
                    handletextpad=0.4, columnspacing=1.0)
    leg.get_frame().set_facecolor("#070b12"); leg.get_frame().set_alpha(0.82); leg.get_frame().set_edgecolor("none")  # noqa: E702
    for x in leg.get_texts():
        x.set_color("white")
    ov.add_patch(plt.Rectangle((0, 0), 1, 0.07, color="#070b12", alpha=0.88))
    bar = fig.add_axes([0.012, 0.022, 0.2, 0.013], zorder=60)
    bar.imshow(np.linspace(IR_VMIN, 20, 256)[None, :], cmap=S.cmap, vmin=IR_VMIN, vmax=IR_VMAX, aspect="auto",
               extent=(IR_VMIN, 20, 0, 1))
    bar.set_yticks([]); bar.set_xticks([-80, -60, -40, -20, 0, 20])   # noqa: E702
    bar.tick_params(colors="#c8d0dc", labelsize=10, length=2)
    ov.text(0.012, 0.055, f"Infrared cloud-top temperature (°C) · {irt:%b %-d %H%MZ}", color="white", fontsize=11,
            va="center")
    ov.text(0.988, 0.035, f"Track: NHC best track  ·  Infrared: NOAA GOES / GridSat  ·  {config.HANDLE or config.BRAND}",
            color="#9aa6b6", fontsize=11, va="center", ha="right")
    path = f"{outdir}/f{idx:05d}.png"
    fig.savefig(path, facecolor="black")
    plt.close(fig)
    return path


def build(out_path, storm_id=None, name=None, year=None, live=False, track=None, ir_frames=None,
          max_frames=None, workdir=None):
    """Build the MP4. Returns facts for the caption."""
    global _STORM, _LINES
    from multiprocessing import get_context
    track = track or load_track(storm_id, name, year, live)
    pts = track["points"]
    lons, lats = [p["lon"] for p in pts], [p["lat"] for p in pts]
    box = (min(lons) - 32, max(lons) + 32, max(min(lats) - 18, -60), min(max(lats) + 18, 66))
    workdir = workdir or (config.CACHE_DIR / f"anim_{track.get('id', 'storm')}")
    if ir_frames is None:
        ir_frames = fetch_ir(track, box, workdir / "ir")
    S = Storm(track, ir_frames)
    _STORM = S
    bm = geo.basemap()
    _LINES = bm["coast"] + bm["countries"] + bm["states"]
    sched = S.schedule()
    if max_frames:
        sched = sched[:max_frames]
    frames_dir = workdir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    jobs = [(i, t, v, ph, str(frames_dir)) for i, t, v, ph in sched]
    with get_context("fork").Pool(os.cpu_count() or 2) as pool:
        for k, _ in enumerate(pool.imap(render_frame, jobs, chunksize=4)):
            if k % 100 == 0:
                print(f"[anim] frame {k}/{len(jobs)}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{frames_dir}/f%05d.png",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out_path)], check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    pk = S.peak
    return {"storm": f"{S.kind} {S.name}", "year": S.T0.year, "id": S.id,
            "dates": f"{S.T0:%b %-d}–{S.T1:%b %-d, %Y}",
            "peak": f"{mph(pk['kt'])} mph, {pk['mb']} mb on {pk['t']:%b %-d}",
            "peak_category": category(pk["kt"]),
            "landfalls": [f"{txt} on {p['t']:%b %-d} at {mph(p['kt'])} mph" for p, txt in zip(S.landfalls, S.landfall_text)],
            "duration_days": round((S.T1 - S.T0).total_seconds() / 86400, 1),
            "hashtags": [f"#{S.name}", "#hurricane" if pk["kt"] >= 64 else "#tropics"]}
