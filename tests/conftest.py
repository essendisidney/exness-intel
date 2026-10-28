import numpy as np
import pandas as pd
import pytest

from exness_intel.data import Bundle, Quote
from exness_intel.indicators import resample_ohlc
from exness_intel import config


def make_m15(days=260, drift=0.0, start=2000.0, vol=0.6, seed=1, end="2026-10-28 06:00"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=days * 96, freq="15min")
    rets = drift + rng.normal(0, vol, len(idx))
    # add gentle oscillation so pullbacks exist
    rets += 0.8 * np.sin(np.arange(len(idx)) / 40.0) * vol * 0.3
    close = start + np.cumsum(rets)
    open_ = np.concatenate([[start], close[:-1]])
    hi = np.maximum(open_, close) + rng.uniform(0.05, 0.6, len(idx)) * vol
    lo = np.minimum(open_, close) - rng.uniform(0.05, 0.6, len(idx)) * vol
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": rng.integers(50, 500, len(idx)).astype(float)}, index=idx)


def frames_from(m15):
    h1 = resample_ohlc(m15, "1h")
    return {"D1": resample_ohlc(m15, "1D"), "H4": resample_ohlc(m15, "4h"), "H1": h1, "M15": m15}


def bundle_from(m15, key="XAUUSD", quote=True):
    b = Bundle(key=key, instrument=config.INSTRUMENTS[key])
    b.frames = frames_from(m15)
    last = float(m15["close"].iloc[-1])
    ts = m15.index[-1].to_pydatetime()
    b.quote = Quote(last - 0.1, last + 0.1, ts, "test") if quote else Quote(None, None, ts, "test")
    b.source_kind = "mt5" if quote else "proxy"
    b.source_detail = "synthetic"
    b.available = True
    return b


@pytest.fixture
def up_m15():
    return make_m15(drift=0.08, seed=3)


@pytest.fixture
def down_m15():
    return make_m15(drift=-0.08, seed=4)


@pytest.fixture
def flat_m15():
    return make_m15(drift=0.0, seed=5, vol=0.3)
