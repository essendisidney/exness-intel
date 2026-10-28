"""Macro context: economic calendar + cross-asset readings + headlines.

Calendar source: the public Forex Factory weekly JSON export
(nfs.faireconomy.media). Values shown exactly as published; nothing inferred.
Headlines: RSS feeds, title + publish time + link only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import requests

from . import config
from .data import macro_series

log = logging.getLogger(__name__)
UTC = timezone.utc
EAT = timezone(timedelta(hours=3))

CALENDAR_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
]

# Keywords for events that matter to gold/silver/oil.
PRIORITY = {
    "FOMC": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
    "Federal Funds Rate": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
    "Fed Chair": ["XAUUSD", "XAGUSD"],
    "CPI": ["XAUUSD", "XAGUSD"],
    "PCE": ["XAUUSD", "XAGUSD"],
    "Non-Farm": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
    "Unemployment": ["XAUUSD", "XAGUSD"],
    "GDP": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
    "Retail Sales": ["XAUUSD", "XAGUSD"],
    "ISM Manufacturing": ["XAGUSD", "USOIL", "UKOIL", "XAUUSD"],
    "Crude Oil Inventories": ["USOIL", "UKOIL"],
    "OPEC": ["USOIL", "UKOIL"],
}

HEADLINE_FEEDS = {
    "EIA Today in Energy": "https://www.eia.gov/rss/todayinenergy.xml",
    "OilPrice.com": "https://oilprice.com/rss/main",
    "FXStreet commodities": "https://www.fxstreet.com/rss/news",
}


def _affects(title: str) -> list[str]:
    for kw, syms in PRIORITY.items():
        if kw.lower() in title.lower():
            return syms
    return []


def fetch_calendar(now: datetime, horizon_h: int = 36) -> dict:
    for url in CALENDAR_URLS:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "exness-intel/1.0"})
            r.raise_for_status()
            rows = r.json()
            break
        except Exception as e:
            log.warning("calendar %s failed: %s", url, e)
    else:
        return {"available": False, "events": [], "source": "Forex Factory export (unreachable)"}

    out = []
    for e in rows:
        try:
            t = datetime.fromisoformat(e["date"]).astimezone(UTC)
        except Exception:
            continue
        if e.get("country") not in ("USD", "ALL", "CNY", "EUR"):
            continue
        affects = _affects(e.get("title", ""))
        high = e.get("impact") == "High"
        if not (high or affects):
            continue
        if not (now - timedelta(hours=12) <= t <= now + timedelta(hours=horizon_h)):
            continue
        out.append({
            "title": e.get("title"), "country": e.get("country"), "impact": e.get("impact"),
            "time_utc": t.isoformat(), "time_eat": t.astimezone(EAT).strftime("%a %d %b %H:%M"),
            "forecast": e.get("forecast") or "", "previous": e.get("previous") or "",
            "actual": e.get("actual") or "",
            "affects": affects or ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
            "high": high,
        })
    out.sort(key=lambda x: x["time_utc"])
    return {"available": True, "events": out, "source": "Forex Factory weekly calendar export"}


def blackout(events: list[dict], key: str, now: datetime, minutes: int = config.EVENT_BLACKOUT_MIN) -> dict | None:
    for e in events:
        if not e["high"] or key not in e["affects"]:
            continue
        t = datetime.fromisoformat(e["time_utc"])
        if -minutes <= (t - now).total_seconds() / 60 <= minutes:
            return e
    return None


def next_event(events: list[dict], key: str, now: datetime) -> dict | None:
    for e in events:
        if key in e["affects"] and datetime.fromisoformat(e["time_utc"]) >= now:
            return e
    return None


def cross_asset() -> dict:
    out = {}
    for name, t in config.MACRO_TICKERS.items():
        df = macro_series(t)
        if df is None or len(df) < 25:
            out[name] = None
            continue
        last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        ch5 = float(df["close"].iloc[-1] - df["close"].iloc[-6])
        out[name] = {"last": round(last, 3), "d1_change": round(last - prev, 3), "w1_change": round(ch5, 3),
                     "as_of": df.index[-1].strftime("%Y-%m-%d"),
                     "trend_20d": "up" if last > df["close"].iloc[-20:].mean() else "down"}
    return out


def headlines(now: datetime, max_age_h: int = 24, per_feed: int = 4) -> list[dict]:
    out = []
    for src, url in HEADLINE_FEEDS.items():
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "exness-intel/1.0"})
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as e:
            log.warning("feed %s failed: %s", src, e)
            continue
        n = 0
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = item.findtext("pubDate")
            try:
                ts = parsedate_to_datetime(pub).astimezone(UTC) if pub else None
            except Exception:
                ts = None
            if ts and (now - ts) > timedelta(hours=max_age_h):
                continue
            kw = ("gold", "silver", "oil", "crude", "opec", "fed", "dollar", "inflation", "brent", "wti", "treasury")
            if not any(k in title.lower() for k in kw):
                continue
            out.append({"source": src, "title": title, "link": link,
                        "published_eat": ts.astimezone(EAT).strftime("%d %b %H:%M") if ts else "time n/a"})
            n += 1
            if n >= per_feed:
                break
    return out


def fundamental_bias(key: str, xa: dict, gsr: float | None) -> tuple[int, str]:
    """Mechanical, transparent bias from cross-asset moves.
    Returns (-1/0/+1, explanation). Gold/silver: falling DXY & yields = +1.
    Oil: no reliable mechanical proxy here -> 0 (left to calendar/headlines)."""
    dxy, y10 = xa.get("DXY"), xa.get("US10Y")
    if key in ("XAUUSD", "XAGUSD"):
        if not dxy or not y10:
            return 0, "DXY/yield data unavailable"
        score = (-1 if dxy["trend_20d"] == "up" else 1) + (-1 if y10["trend_20d"] == "up" else 1)
        txt = f"DXY 20d trend {dxy['trend_20d']}, US10Y 20d trend {y10['trend_20d']}"
        if key == "XAGUSD" and gsr:
            txt += f"; gold/silver ratio {gsr:.1f}"
        return (1 if score >= 2 else -1 if score <= -2 else 0), txt
    if dxy:
        return 0, f"oil: DXY 20d trend {dxy['trend_20d']} (weak link; no mechanical bias applied)"
    return 0, "oil: no mechanical fundamental bias"
