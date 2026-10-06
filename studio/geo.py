"""Basemap drawing, projections and place lookups (bundled Natural Earth data, no downloads)."""
import json
import math
from functools import lru_cache

import numpy as np

from . import config

US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


@lru_cache(maxsize=1)
def basemap():
    return json.loads((config.DATA_DIR / "basemap.json").read_text())


def places(min_pop=0):
    return [p for p in basemap()["places"] if p[3] >= min_pop]


def nearest_place(lat, lon, min_pop=100000):
    """'Tulsa, OK' style label for the closest sizeable city, with distance in miles."""
    best, bd = None, 1e9
    for name, plat, plon, pop, adm in places(min_pop):
        d = haversine_mi(lat, lon, plat, plon)
        if d < bd:
            best, bd = (name, US_STATES.get(adm, adm)), d
    if not best:
        return None, None
    return f"{best[0]}, {best[1]}" if best[1] else best[0], round(bd)


def haversine_mi(lat1, lon1, lat2, lon2):
    r = math.radians
    a = (math.sin(r(lat2 - lat1) / 2) ** 2 +
         math.cos(r(lat1)) * math.cos(r(lat2)) * math.sin(r(lon2 - lon1) / 2) ** 2)
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def state_hashtags(states, limit=3):
    return [f"#{s}wx" for s in states[:limit] if len(s) == 2]


# --- Lambert conformal conic (North America analysis maps) ----------------
class Lambert:
    def __init__(self, lon0=-100.0, lat0=40.0, lat1=33.0, lat2=45.0):
        r = np.radians
        self.lon0 = lon0
        self.n = (np.log(np.cos(r(lat1)) / np.cos(r(lat2))) /
                  np.log(np.tan(np.pi / 4 + r(lat2) / 2) / np.tan(np.pi / 4 + r(lat1) / 2)))
        self.F = np.cos(r(lat1)) * np.tan(np.pi / 4 + r(lat1) / 2) ** self.n / self.n
        self.rho0 = self.F / np.tan(np.pi / 4 + r(lat0) / 2) ** self.n

    def __call__(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        lon = (lon + 180) % 360 - 180
        rho = self.F / np.tan(np.pi / 4 + np.radians(lat) / 2) ** self.n
        th = self.n * np.radians(lon - self.lon0)
        return rho * np.sin(th), self.rho0 - rho * np.cos(th)


    def rotate(self, u, v, lon):
        """Earth-relative wind components -> map-relative (so barbs point the right way)."""
        th = self.n * np.radians(((np.asarray(lon) + 180) % 360 - 180) - self.lon0)
        return u * np.cos(th) - v * np.sin(th), u * np.sin(th) + v * np.cos(th)


class PlateCarree:
    def __call__(self, lon, lat):
        return np.asarray(lon, float), np.asarray(lat, float)


def draw_basemap(ax, proj, land=None, lw=0.7, color="#9fb0c4", states=True, lakes=True):
    bm = basemap()
    layers = [("coast", lw * 1.1), ("countries", lw)] + ([("states", lw * 0.6)] if states else [])
    for key, w in layers:
        for line in bm[key]:
            a = np.array(line)
            x, y = proj(a[:, 0], a[:, 1])
            ax.plot(x, y, color=color, lw=w, zorder=5, solid_capstyle="round")
    if lakes:
        for poly in bm["lakes"]:
            a = np.array(poly)
            x, y = proj(a[:, 0], a[:, 1])
            ax.plot(x, y, color=color, lw=lw * 0.5, zorder=5)


def bbox_for(points, min_w=4.0, aspect=16 / 9, pad=0.8, max_w=24.0):
    """Lon/lat box around points, padded, matching the video aspect after cos(lat) scaling."""
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    w0, e0, s0, n0 = min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad
    lat_c, lon_c = (s0 + n0) / 2, (w0 + e0) / 2
    k = math.cos(math.radians(lat_c))
    w = max(e0 - w0, min_w, (n0 - s0) * aspect / k)
    w = min(w, max_w)
    h = w * k / aspect
    return [lon_c - w / 2, lon_c + w / 2, lat_c - h / 2, lat_c + h / 2]


def union_bbox(a, b):
    return [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]
