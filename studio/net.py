"""Shared HTTP session with retries and the NWS-required User-Agent."""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

_session = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET"}))
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = config.USER_AGENT
        _session = s
    return _session


def get(url: str, **kw) -> requests.Response:
    r = session().get(url, timeout=kw.pop("timeout", 45), **kw)
    r.raise_for_status()
    return r


def get_json(url: str):
    return get(url, headers={"Accept": "application/geo+json, application/json"}).json()
