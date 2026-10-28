"""Market data layer.

Priority:
  1. MT5 bridge snapshot (broker bars, live bid/ask, verified contract specs)
     if younger than MAX_SNAPSHOT_AGE_S.
  2. Public futures proxy via yfinance (CME/ICE front month). Prices differ
     from Exness CFD quotes by a basis, so proxy-based setups are never marked
     executable; levels are shifted onto broker prices only when a fresh
     broker quote exists.
Nothing is ever filled in: a missing frame or quote => DATA UNAVAILABLE.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import config
from .indicators import resample_ohlc

log = logging.getLogger(__name__)
UTC = timezone.utc


@dataclass
class Quote:
    bid: float | None
    ask: float | None
    ts: datetime | None
    source: str

    @property
    def spread(self):
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def mid(self):
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2


@dataclass
class Bundle:
    key: str
    instrument: config.Instrument
    frames: dict = field(default_factory=dict)   # tf -> DataFrame
    quote: Quote | None = None
    source_kind: str = "none"                     # mt5 | proxy | none
    source_detail: str = ""
    broker_offset: float = 0.0                    # broker_mid - proxy_close, applied to levels
    warnings: list = field(default_factory=list)
    available: bool = False


# --------------------------------------------------------------------------- market hours
def market_status(now: datetime, key: str) -> dict:
    """Approximate Exness CFD hours for metals/oil (UTC): Sun ~22:05 to
    Fri ~20:58 with a daily break ~21:00-22:05. Approximate; MT5 session
    data from the bridge overrides this when present."""
    now = now.astimezone(UTC)
    wd, hm = now.weekday(), now.hour * 60 + now.minute
    if wd == 5 or (wd == 6 and hm < 22 * 60 + 5) or (wd == 4 and hm >= 20 * 60 + 58):
        return {"open": False, "reason": "weekend close"}
    if 21 * 60 <= hm < 22 * 60 + 5:
        return {"open": False, "reason": "daily maintenance break (~21:00-22:05 UTC)"}
    return {"open": True, "reason": "within scheduled session (approximate)"}


# --------------------------------------------------------------------------- proxy (yfinance)
def _yf_download(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False,
                             auto_adjust=False, threads=False)
            if df is not None and len(df):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                df.index = pd.to_datetime(df.index, utc=True)
                return df.dropna(subset=["close"])
        except Exception as e:  # network / parse
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"no data for {ticker} {interval}: {last_err}")


def proxy_frames(ticker: str) -> dict:
    d1 = _yf_download(ticker, "2y", "1d")
    h1 = _yf_download(ticker, "180d", "1h")
    m15 = _yf_download(ticker, "30d", "15m")
    return {"D1": d1, "H4": resample_ohlc(h1, "4h"), "H1": h1, "M15": m15}


def macro_series(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        return _yf_download(ticker, period, "1d")
    except Exception as e:
        log.warning("macro %s unavailable: %s", ticker, e)
        return None


# --------------------------------------------------------------------------- MT5 snapshot
def load_snapshot(path: str | None = None) -> dict | None:
    """Load the bridge snapshot from a local path (gz or plain JSON)."""
    path = path or os.environ.get("MT5_SNAPSHOT_PATH", "data/mt5_snapshot.json.gz")
    p = Path(path)
    if not p.exists():
        return None
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def snapshot_age_s(snap: dict, now: datetime) -> float:
    ts = datetime.fromisoformat(snap["generated_utc"].replace("Z", "+00:00"))
    return (now - ts).total_seconds()


def _frame_from_rates(rates: list) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def instrument_from_snapshot(base: config.Instrument, spec: dict) -> config.Instrument:
    return replace(
        base,
        broker_symbol=spec.get("name", base.broker_symbol),
        contract_size=float(spec["trade_contract_size"]),
        tick_size=float(spec["trade_tick_size"]),
        min_lot=float(spec["volume_min"]),
        lot_step=float(spec["volume_step"]),
        specs_verified=True,
    )


# --------------------------------------------------------------------------- assemble
def build_bundle(key: str, now: datetime, snapshot: dict | None, fetch_proxy=proxy_frames) -> Bundle:
    inst = config.INSTRUMENTS[key]
    b = Bundle(key=key, instrument=inst)
    snap_sym = None
    snap_fresh = False
    if snapshot:
        age = snapshot_age_s(snapshot, now)
        snap_sym = snapshot.get("symbols", {}).get(key)
        snap_fresh = age <= config.MAX_SNAPSHOT_AGE_S
        if snap_sym is None:
            b.warnings.append(f"{key} not present in MT5 snapshot (symbol unsupported or not in Market Watch)")
        elif not snap_fresh:
            b.warnings.append(f"MT5 snapshot is {age/60:.0f} min old; broker quotes not used")
        if snap_sym and snap_sym.get("spec"):
            b.instrument = instrument_from_snapshot(inst, snap_sym["spec"])

    if snap_sym and snap_fresh and snap_sym.get("rates"):
        try:
            b.frames = {tf: _frame_from_rates(r) for tf, r in snap_sym["rates"].items()}
            t = snap_sym["tick"]
            b.quote = Quote(t["bid"], t["ask"], datetime.fromtimestamp(t["time"], UTC), "Exness MT5 (bridge)")
            b.source_kind, b.source_detail, b.available = "mt5", f"Exness MT5 {b.instrument.broker_symbol}", True
            return b
        except Exception as e:
            b.warnings.append(f"snapshot parse failed: {e}")

    for ticker, is_spot in zip(inst.proxy_tickers, inst.proxy_is_spot):
        try:
            frames = fetch_proxy(ticker)
        except Exception as e:
            b.warnings.append(f"proxy {ticker} failed: {e}")
            continue
        b.frames = frames
        last = frames["M15"].iloc[-1]
        last_ts = frames["M15"].index[-1].to_pydatetime()
        b.source_kind = "proxy"
        b.source_detail = f"Yahoo Finance {ticker} ({'spot' if is_spot else 'futures front-month'}), delayed"
        b.available = True
        if snap_sym and snap_fresh and snap_sym.get("tick"):
            t = snap_sym["tick"]
            b.quote = Quote(t["bid"], t["ask"], datetime.fromtimestamp(t["time"], UTC), "Exness MT5 (bridge)")
            b.broker_offset = b.quote.mid - float(last["close"])
        else:
            # No broker quote: bid/ask are NOT available from this source.
            b.quote = Quote(None, None, last_ts, f"{ticker} last 15m close {float(last['close']):.3f} (no bid/ask)")
            if not is_spot:
                b.warnings.append("futures proxy: price differs from Exness CFD by a basis; levels are indicative only")
        return b

    b.warnings.append("DATA UNAVAILABLE: no source returned data")
    return b


def data_age_s(b: Bundle, now: datetime) -> float | None:
    m15 = b.frames.get("M15")
    if m15 is None or not len(m15):
        return None
    # last bar *open* time + 15 min = close time of the latest (possibly forming) bar
    return (now - (m15.index[-1].to_pydatetime() + timedelta(minutes=15))).total_seconds()
