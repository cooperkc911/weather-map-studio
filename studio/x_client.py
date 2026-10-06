"""Minimal X API v2 client: chunked media upload + create post (OAuth 1.0a user context).

OAuth 1.0a keys never expire, which suits an unattended GitHub Actions bot.
Endpoints: POST /2/media/upload/initialize, /2/media/upload/{id}/append,
/2/media/upload/{id}/finalize, GET /2/media/upload?command=STATUS, POST /2/tweets.
"""
import mimetypes
import os
import time
from pathlib import Path

import requests
from requests_oauthlib import OAuth1

API = "https://api.x.com/2"
CHUNK = 4 * 1024 * 1024


class XError(RuntimeError):
    pass


def _auth() -> OAuth1:
    keys = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET"]
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise XError(f"Missing X credentials: {', '.join(missing)}")
    return OAuth1(*(os.environ[k] for k in keys))


def _check(r: requests.Response) -> dict:
    if r.status_code >= 300:
        raise XError(f"X API {r.request.method} {r.url} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.content else {}


def upload_media(path: Path) -> str:
    auth = _auth()
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    category = "tweet_video" if mime.startswith("video") else (
        "tweet_gif" if mime == "image/gif" else "tweet_image")
    size = path.stat().st_size

    init = _check(requests.post(f"{API}/media/upload/initialize", auth=auth, timeout=60, json={
        "media_type": mime, "total_bytes": size, "media_category": category}))
    media_id = init["data"]["id"]

    with path.open("rb") as fh:
        idx = 0
        while chunk := fh.read(CHUNK):
            _check(requests.post(f"{API}/media/upload/{media_id}/append", auth=auth, timeout=120,
                                 data={"segment_index": str(idx)},
                                 files={"media": (path.name, chunk, "application/octet-stream")}))
            idx += 1

    fin = _check(requests.post(f"{API}/media/upload/{media_id}/finalize", auth=auth, timeout=60))
    info = (fin.get("data") or {}).get("processing_info")
    deadline = time.time() + 300
    while info and info.get("state") in ("pending", "in_progress"):
        if time.time() > deadline:
            raise XError("media processing timed out")
        time.sleep(max(1, int(info.get("check_after_secs", 2))))
        st = _check(requests.get(f"{API}/media/upload", auth=auth, timeout=30,
                                 params={"command": "STATUS", "media_id": media_id}))
        info = (st.get("data") or {}).get("processing_info")
    if info and info.get("state") == "failed":
        raise XError(f"media processing failed: {info}")
    return media_id


def create_post(text: str, media_ids: list[str] | None = None, reply_to: str | None = None) -> str:
    body = {"text": text}
    if media_ids:
        body["media"] = {"media_ids": media_ids}
    if reply_to:
        body["reply"] = {"in_reply_to_tweet_id": reply_to}
    data = _check(requests.post(f"{API}/tweets", auth=_auth(), json=body, timeout=60))
    return data["data"]["id"]
