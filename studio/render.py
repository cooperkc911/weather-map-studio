"""Image branding and time-lapse video encoding (Pillow + ffmpeg)."""
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont

from . import config

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def font(size: int):
    for p in _FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def local_time_label(dt: datetime) -> str:
    tz = ZoneInfo(config.DISPLAY_TZ)
    loc = dt.astimezone(tz)
    return f"{loc:%b %-d, %-I:%M %p} {loc.tzname()} · {dt.astimezone(timezone.utc):%H:%MZ}"


def brand(img: Image.Image, title: str, subtitle: str = "", max_w: int = 1280,
          credit: str = "Data: NOAA / National Weather Service") -> Image.Image:
    """Resize to max_w, then add a dark header bar with title/subtitle and a footer credit."""
    img = img.convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    w = img.width
    bar = max(40, w // 22)
    foot = max(22, w // 48)
    canvas = Image.new("RGB", (w, img.height + bar + foot), (14, 20, 30))
    canvas.paste(img, (0, bar))
    d = ImageDraw.Draw(canvas)
    f_title, f_sub, f_foot = font(int(bar * 0.48)), font(int(bar * 0.36)), font(int(foot * 0.6))
    pad = int(bar * 0.3)
    d.text((pad, bar / 2), title, font=f_title, fill=(255, 255, 255), anchor="lm")
    if subtitle:
        d.text((w - pad, bar / 2), subtitle, font=f_sub, fill=(255, 210, 90), anchor="rm")
    d.text((pad, img.height + bar + foot / 2), credit, font=f_foot, fill=(170, 180, 195), anchor="lm")
    tag = config.HANDLE or config.BRAND
    d.text((w - pad, img.height + bar + foot / 2), tag, font=f_foot, fill=(170, 180, 195), anchor="rm")
    return canvas


def even(img: Image.Image) -> Image.Image:
    """H.264 needs even dimensions."""
    w, h = img.width - img.width % 2, img.height - img.height % 2
    return img if (w, h) == img.size else img.crop((0, 0, w, h))


def frames_to_mp4(frames: list[Image.Image], out: Path, fps: int = 12,
                  hold_last_sec: float = 1.5, loops: int = 2) -> Path:
    """Encode frames into an X-friendly MP4 (H.264, yuv420p, faststart)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed")
    if not frames:
        raise ValueError("no frames to encode")
    size = even(frames[0]).size
    seq = []
    hold = max(1, int(fps * hold_last_sec))
    for _ in range(loops):
        seq += frames + [frames[-1]] * hold
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(seq):
            fr = even(fr.convert("RGB"))
            if fr.size != size:
                fr = fr.resize(size, Image.LANCZOS)
            fr.save(f"{tmp}/f{i:05d}.png")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-preset", "medium",
               "-crf", "23", "-pix_fmt", "yuv420p", "-profile:v", "high",
               "-movflags", "+faststart", "-r", "30", str(out)]
        subprocess.run(cmd, check=True)
    return out
