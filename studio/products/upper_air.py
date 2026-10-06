"""Upper-air analysis from the GFS analysis (f000) with anomalies vs. the 1991-2020 climatology.

Produces three maps (posted together as one 3-image post):
  1. 500 mb heights + height anomaly
  2. 250 mb jet stream (wind speed, heights, barbs)
  3. 850 mb temperature anomaly + isotherms
"""
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import config, geo, net
from ..geo import Lambert

LEVELS = {("HGT", "500 mb"), ("HGT", "250 mb"), ("UGRD", "250 mb"), ("VGRD", "250 mb"), ("TMP", "850 mb")}


# --- Data -----------------------------------------------------------------
def latest_cycle(now=None):
    """Most recent 00Z/12Z cycle whose analysis should be on the servers (~4h latency)."""
    now = (now or datetime.now(timezone.utc)) - timedelta(hours=4)
    return now.replace(hour=0 if now.hour < 12 else 12, minute=0, second=0, microsecond=0)


def _aws_ranges(idx_text):
    rows = [ln.split(":") for ln in idx_text.strip().splitlines()]
    out = []
    for i, r in enumerate(rows):
        if (r[3], r[4]) in LEVELS:
            start = int(r[1])
            end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else ""
            out.append((start, end))
    return out


def fetch_gfs(cycle):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CACHE_DIR / f"gfs_{cycle:%Y%m%d%H}_anl.grib2"
    if path.exists() and path.stat().st_size > 10000:
        return path
    d, h = f"{cycle:%Y%m%d}", f"{cycle:%H}"
    try:
        data = net.get(config.GFS_NOMADS.format(d=d, h=h), timeout=120).content
        if not data.startswith(b"GRIB"):
            raise RuntimeError("NOMADS did not return GRIB")
    except Exception as e:  # noqa: BLE001
        print(f"[upper-air] NOMADS failed ({e}); trying AWS open data")
        url = config.GFS_AWS.format(d=d, h=h)
        chunks = []
        for start, end in _aws_ranges(net.get(url + ".idx").text):
            chunks.append(net.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120).content)
        data = b"".join(chunks)
    path.write_bytes(data)
    return path


def load_gfs(path):
    """Decode one field at a time (cfgrib can't merge variables with different level sets)."""
    import xarray as xr

    def field(short, level):
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": short}, "indexpath": ""})
        da = ds[short]
        if "isobaricInhPa" in da.dims:
            da = da.sel(isobaricInhPa=level)
        elif float(da.isobaricInhPa) != level:
            raise ValueError(f"{short} at {level} mb not in file")
        da = da.sortby("latitude").sel(latitude=slice(10, 75), longitude=slice(190, 330))
        return da

    z500 = field("gh", 500)
    return {
        "lat": z500.latitude.values, "lon": z500.longitude.values,
        "z500": z500.values / 10.0,                          # dam
        "z250": field("gh", 250).values / 10.0,
        "u250": field("u", 250).values, "v250": field("v", 250).values,
        "t850": field("t", 850).values - 273.15,             # C
    }


def climatology(var, level, when, lat, lon):
    """Monthly 1991-2020 mean, interpolated to the day of year and onto the GFS grid."""
    import xarray as xr
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = config.CACHE_DIR / f"{var}.mon.ltm.1991-2020.nc"
    if not f.exists():
        f.write_bytes(net.get(config.PSL_LTM.format(var=var), timeout=180).content)
    ds = xr.open_dataset(f, decode_times=False)
    da = ds[var].sel(level=level)
    return interp_month(da.values, da.lat.values, da.lon.values, when, lat, lon)


def interp_month(cube, clat, clon, when, lat, lon):
    """cube: (12, nlat, nlon). Linear in time between mid-month means, bilinear in space."""
    from scipy.interpolate import RegularGridInterpolator
    doy = when.timetuple().tm_yday + when.hour / 24
    mids = np.array([15.5 + 30.42 * m for m in range(12)])
    i1 = int(np.searchsorted(mids, doy)) % 12
    i0 = (i1 - 1) % 12
    span = (mids[i1] - mids[i0]) % 365.25 or 30.42
    w = ((doy - mids[i0]) % 365.25) / span
    field = (1 - w) * cube[i0] + w * cube[i1]
    if clat[0] > clat[-1]:
        clat, field = clat[::-1], field[::-1]
    fn = RegularGridInterpolator((clat, clon), field, bounds_error=False, fill_value=None)
    LA, LO = np.meshgrid(lat, lon % 360, indexing="ij")
    return fn(np.stack([LA, LO], axis=-1))


def analyze(g, when, clim=None):
    """Adds anomaly fields. clim: optional dict of precomputed climatology (tests)."""
    if clim is None:
        clim = {"z500": climatology("hgt", 500, when, g["lat"], g["lon"]) / 10.0,
                "t850": climatology("air", 850, when, g["lat"], g["lon"])}
        if np.nanmean(clim["t850"]) > 150:   # file in Kelvin
            clim["t850"] = clim["t850"] - 273.15
    g["z500a"] = g["z500"] - clim["z500"]
    g["t850a"] = g["t850"] - clim["t850"]
    g["wspd250"] = np.hypot(g["u250"], g["v250"]) * 1.94384   # kt
    return g


# --- Facts for captions -----------------------------------------------------
_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def describe_point(lat, lon):
    lon = (lon + 180) % 360 - 180
    place, dist = geo.nearest_place(lat, lon)
    if place is None:
        return f"{abs(lat):.0f}{'N' if lat >= 0 else 'S'} {abs(lon):.0f}W"
    if dist < 60:
        return f"near {place}"
    plat, plon = next((p[1], p[2]) for p in geo.places(100000) if place.startswith(p[0]))
    brg = (np.degrees(np.arctan2((lon - plon) * np.cos(np.radians(lat)), lat - plat)) + 360) % 360
    return f"about {dist} mi {_DIRS[int((brg + 11.25) // 22.5) % 16]} of {place}"


def _ext(field, lat, lon, fn):
    i = np.unravel_index(fn(field), field.shape)
    return float(field[i]), float(lat[i[0]]), float(lon[i[1]])


def facts(g, cycle):
    lat, lon = g["lat"], g["lon"]
    hi = _ext(g["z500a"], lat, lon, np.nanargmax)
    lo = _ext(g["z500a"], lat, lon, np.nanargmin)
    warm = _ext(g["t850a"], lat, lon, np.nanargmax)
    cold = _ext(g["t850a"], lat, lon, np.nanargmin)
    jet = _ext(g["wspd250"], lat, lon, np.nanargmax)
    notable = (max(abs(hi[0]), abs(lo[0])) >= config.UPPER_NOTABLE_HGT_DAM or
               max(abs(warm[0]), abs(cold[0])) >= config.UPPER_NOTABLE_T850_C)
    return {
        "product": f"GFS analysis valid {cycle:%HZ %b %-d}: 500 mb heights/anomaly, 250 mb jet, 850 mb temp anomaly",
        "strongest_ridge_500mb": f"+{hi[0]:.0f} dam anomaly {describe_point(hi[1], hi[2])}",
        "deepest_trough_500mb": f"{lo[0]:.0f} dam anomaly {describe_point(lo[1], lo[2])}",
        "warmest_850mb_anomaly": f"+{warm[0]:.0f}C {describe_point(warm[1], warm[2])}",
        "coldest_850mb_anomaly": f"{cold[0]:.0f}C {describe_point(cold[1], cold[2])}",
        "jet_max_250mb": f"{jet[0]:.0f} kt {describe_point(jet[1], jet[2])}",
        "notable": notable,
    }


# --- Maps -------------------------------------------------------------------
def _frame(title, subtitle, credit="GFS analysis (NOAA/NCEP) · anomalies vs 1991-2020 NCEP reanalysis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 10), dpi=100, facecolor="#0e141e")
    ax = fig.add_axes([0.0, 0.105, 1.0, 0.825])
    ax.set_facecolor("#0e141e")
    ax.axis("off")
    fig.text(0.015, 0.965, title, color="white", fontsize=24, weight="bold", va="center")
    fig.text(0.985, 0.965, subtitle, color="#ffd25a", fontsize=17, weight="bold", va="center", ha="right")
    fig.text(0.015, 0.02, credit, color="#aab4c3", fontsize=11, va="center")
    fig.text(0.985, 0.02, config.HANDLE or config.BRAND, color="#aab4c3", fontsize=11, va="center", ha="right")
    return fig, ax


def _extent(ax, proj):
    pts = [(-128, 22), (-64, 22), (-145, 55), (-50, 55), (-100, 62), (-100, 17)]
    xs, ys = proj([p[0] for p in pts], [p[1] for p in pts])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    want = 16 * 1.0 / (10 * 0.825)  # axes aspect (w/h)
    w, h = x1 - x0, y1 - y0
    if w / h < want:
        cx = (x0 + x1) / 2; w = h * want; x0, x1 = cx - w / 2, cx + w / 2   # noqa: E702
    else:
        cy = (y0 + y1) / 2; h = w / want; y0, y1 = cy - h / 2, cy + h / 2   # noqa: E702
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _colorbar(fig, cf, label):
    cax = fig.add_axes([0.30, 0.078, 0.40, 0.014])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors="white", labelsize=10)
    cb.outline.set_edgecolor("#556")
    cb.set_label(label, color="white", fontsize=11, labelpad=2)


def render_maps(g, cycle, out_dir):
    import matplotlib.pyplot as plt
    proj = Lambert()
    LO, LA = np.meshgrid(g["lon"], g["lat"])
    X, Y = proj(LO, LA)
    when = f"Valid {cycle:%HZ %a %b %-d, %Y}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. 500 mb heights + anomaly
    fig, ax = _frame("500 mb Heights & Anomaly", when)
    _extent(ax, proj)
    cf = ax.contourf(X, Y, g["z500a"], levels=np.arange(-36, 37, 3), cmap="RdBu_r", extend="both", zorder=1)
    cs = ax.contour(X, Y, g["z500"], levels=np.arange(480, 612, 6), colors="black", linewidths=1.3, zorder=6)
    ax.clabel(cs, fmt="%d", fontsize=9, inline=True)
    geo.draw_basemap(ax, proj, color="#2b3440", lw=0.8)
    _colorbar(fig, cf, "Height anomaly (dam)")
    paths.append(out_dir / "upper_500mb.png"); fig.savefig(paths[-1], facecolor=fig.get_facecolor()); plt.close(fig)  # noqa: E702

    # 2. 250 mb jet
    fig, ax = _frame("250 mb Jet Stream", when)
    _extent(ax, proj)
    cf = ax.contourf(X, Y, g["wspd250"], levels=np.arange(50, 201, 10), cmap="plasma", extend="max", zorder=1)
    cs = ax.contour(X, Y, g["z250"], levels=np.arange(900, 1120, 12), colors="white", linewidths=0.9, zorder=6)
    ax.clabel(cs, fmt="%d", fontsize=8, inline=True)
    s = (slice(None, None, 14), slice(None, None, 14))
    ub, vb = proj.rotate(g["u250"][s] * 1.94384, g["v250"][s] * 1.94384, LO[s])
    ax.barbs(X[s], Y[s], ub, vb, length=5.5,
             color="#e8eef6", linewidth=0.6, zorder=7)
    geo.draw_basemap(ax, proj, color="#7d8a9a", lw=0.7)
    _colorbar(fig, cf, "Wind speed (kt)")
    paths.append(out_dir / "upper_250mb.png"); fig.savefig(paths[-1], facecolor=fig.get_facecolor()); plt.close(fig)  # noqa: E702

    # 3. 850 mb temperature anomaly
    fig, ax = _frame("850 mb Temperature Anomaly", when)
    _extent(ax, proj)
    cf = ax.contourf(X, Y, g["t850a"], levels=np.arange(-20, 21, 2), cmap="RdBu_r", extend="both", zorder=1)
    cs = ax.contour(X, Y, g["t850"], levels=np.arange(-40, 41, 5), colors="black", linewidths=0.8, zorder=6)
    ax.clabel(cs, fmt="%d°", fontsize=8, inline=True)
    ax.contour(X, Y, g["t850"], levels=[0], colors="#5c2dbb", linewidths=2.2, linestyles="--", zorder=6)
    geo.draw_basemap(ax, proj, color="#2b3440", lw=0.8)
    _colorbar(fig, cf, "Temperature anomaly (°C)   ·   dashed purple = 0°C")
    paths.append(out_dir / "upper_850mb.png"); fig.savefig(paths[-1], facecolor=fig.get_facecolor()); plt.close(fig)  # noqa: E702

    return paths


def build(out_dir, cycle=None, g=None, clim=None):
    cycle = cycle or latest_cycle()
    if g is None:
        g = load_gfs(fetch_gfs(cycle))
    g = analyze(g, cycle, clim)
    paths = render_maps(g, cycle, out_dir)
    return paths, facts(g, cycle)
