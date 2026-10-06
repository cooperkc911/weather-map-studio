"""NEXRAD national composite (IEM n0q, every 5 minutes), cropped to any lon/lat box."""
import io
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from . import config, net

MIN_DBZ = float(__import__("os").getenv("RADAR_MIN_DBZ") or 15)   # hide weaker echoes (clutter)


def floor5(t: datetime) -> datetime:
    return t.replace(minute=t.minute - t.minute % 5, second=0, microsecond=0)


def frame_times(start: datetime, end: datetime, count: int):
    """Evenly spaced 5-minute-aligned times from start to end (inclusive), at most `count`."""
    start, end = floor5(start), floor5(end)
    steps = int((end - start).total_seconds() // 300)
    stride = max(1, -(-steps // max(1, count - 1)))
    times = [start + timedelta(minutes=5 * i) for i in range(0, steps + 1, stride)]
    if times[-1] != end:
        times.append(end)
    return times


def crop(png_bytes: bytes, bbox):
    """Crop the CONUS composite to bbox [w, e, s, n]; returns RGBA array and actual extent."""
    im = Image.open(io.BytesIO(png_bytes))
    x0r, y0r = config.RADAR_UL
    res = config.RADAR_RES
    w, e, s, n = bbox
    px0 = max(0, int((w - x0r) / res))
    px1 = min(im.width, int((e - x0r) / res) + 1)
    py0 = max(0, int((y0r - n) / res))
    py1 = min(im.height, int((y0r - s) / res) + 1)
    if px1 <= px0 or py1 <= py0:
        return None, bbox
    c = im.crop((px0, py0, px1, py1))
    extent = [x0r + px0 * res, x0r + px1 * res, y0r - py1 * res, y0r - py0 * res]
    if c.mode == "P":
        idx = np.array(c)
        pal = np.array(c.getpalette()[:768], dtype=np.uint8).reshape(-1, 3)
        rgba = np.zeros(idx.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = pal[idx]
        min_idx = int((MIN_DBZ + 32) * 2)          # n0q index = (dBZ + 32) * 2
        rgba[..., 3] = np.where(idx >= min_idx, 235, 0)
    else:
        rgba = np.array(c.convert("RGBA"))
    return rgba, extent


def fetch(t: datetime, bbox, back_steps=3):
    """Radar crop at time t (or up to back_steps*5 min earlier if that file isn't there yet)."""
    t = floor5(t)
    for k in range(back_steps + 1):
        tt = t - timedelta(minutes=5 * k)
        try:
            data = net.get(config.IEM_RADAR.format(t=tt), timeout=60).content
            arr, ext = crop(data, bbox)
            return tt, arr, ext
        except Exception:  # noqa: BLE001
            continue
    return t, None, bbox


def now_utc():
    return datetime.now(timezone.utc)
