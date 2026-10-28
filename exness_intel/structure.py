"""Price-structure detection. Every rule here is a documented, mechanical
definition, so results are reproducible and testable. These are measurements,
not predictions.

Definitions
-----------
Swing high/low   : fractal pivot, bar's high (low) is strictly the highest
                   (lowest) of `k` bars either side. Only confirmed pivots
                   (k bars to the right exist) are used -> no look-ahead.
BOS              : close beyond the most recent confirmed swing in the
                   direction of the prevailing structure.
CHoCH            : first close beyond the most recent confirmed swing *against*
                   the prevailing structure (direction of last BOS).
Fair Value Gap   : 3-candle imbalance. Bullish: low[i] > high[i-2].
                   Bearish: high[i] < low[i-2]. Open until price trades back
                   through the gap.
Liquidity sweep  : bar trades beyond a prior swing (wick) but closes back
                   inside it.
Demand/Supply    : last opposite-coloured candle before a displacement move
                   (body >= 1.5 x ATR) that produced a BOS; zone = that candle's range.
Divergence       : price makes a higher high (lower low) between the last two
                   swing highs (lows) while RSI makes a lower high (higher low).
Volatility regime: percentile of current ATR(14) within the last 120 bars.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Swing:
    ts: pd.Timestamp
    price: float
    kind: str  # "high" | "low"
    idx: int


def swings(df: pd.DataFrame, k: int = 2) -> list[Swing]:
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    out: list[Swing] = []
    for i in range(k, len(df) - k):
        win_h = np.concatenate([hi[i - k:i], hi[i + 1:i + k + 1]])
        win_l = np.concatenate([lo[i - k:i], lo[i + 1:i + k + 1]])
        if hi[i] > win_h.max():
            out.append(Swing(df.index[i], float(hi[i]), "high", i))
        if lo[i] < win_l.min():
            out.append(Swing(df.index[i], float(lo[i]), "low", i))
    return out


def market_structure(df: pd.DataFrame, k: int = 2) -> dict:
    """Walk bars forward, tracking last confirmed swing high/low, and record
    BOS / CHoCH events. A swing at index i is only usable from bar i+k on."""
    sw = swings(df, k)
    close = df["close"].to_numpy()
    events = []
    trend = "neutral"
    last_high = last_low = None
    sw_by_confirm: dict[int, list[Swing]] = {}
    for s in sw:
        sw_by_confirm.setdefault(s.idx + k, []).append(s)
    broken_high = broken_low = None
    for i in range(len(df)):
        for s in sw_by_confirm.get(i, []):
            if s.kind == "high":
                last_high = s
            else:
                last_low = s
        if last_high is not None and close[i] > last_high.price and broken_high is not last_high:
            kind = "BOS" if trend in ("bullish", "neutral") else "CHoCH"
            events.append({"ts": df.index[i], "type": kind, "dir": "bullish", "level": last_high.price})
            trend, broken_high = "bullish", last_high
        if last_low is not None and close[i] < last_low.price and broken_low is not last_low:
            kind = "BOS" if trend in ("bearish", "neutral") else "CHoCH"
            events.append({"ts": df.index[i], "type": kind, "dir": "bearish", "level": last_low.price})
            trend, broken_low = "bearish", last_low
    return {
        "trend": trend,
        "last_event": events[-1] if events else None,
        "events": events[-5:],
        "last_swing_high": last_high.price if last_high else None,
        "last_swing_low": last_low.price if last_low else None,
        "swings": sw,
    }


def fair_value_gaps(df: pd.DataFrame, lookback: int = 60) -> list[dict]:
    """Open (unfilled) FVGs within the lookback window."""
    d = df.iloc[-lookback:]
    hi, lo, idx = d["high"].to_numpy(), d["low"].to_numpy(), d.index
    gaps = []
    for i in range(2, len(d)):
        if lo[i] > hi[i - 2]:
            g = {"dir": "bullish", "top": float(lo[i]), "bottom": float(hi[i - 2]), "ts": idx[i]}
            later_low = lo[i + 1:].min() if i + 1 < len(d) else np.inf
            if later_low > g["bottom"]:
                gaps.append(g)
        if hi[i] < lo[i - 2]:
            g = {"dir": "bearish", "top": float(lo[i - 2]), "bottom": float(hi[i]), "ts": idx[i]}
            later_high = hi[i + 1:].max() if i + 1 < len(d) else -np.inf
            if later_high < g["top"]:
                gaps.append(g)
    return gaps[-4:]


def liquidity_sweeps(df: pd.DataFrame, sw: list[Swing], bars: int = 5) -> list[dict]:
    """Sweeps in the last `bars` bars of any of the last 6 confirmed swings."""
    out = []
    recent = df.iloc[-bars:]
    for s in [x for x in sw if x.idx < len(df) - bars][-6:]:
        for ts, row in recent.iterrows():
            if s.kind == "high" and row["high"] > s.price and row["close"] < s.price:
                out.append({"ts": ts, "dir": "bearish", "level": s.price})
            if s.kind == "low" and row["low"] < s.price and row["close"] > s.price:
                out.append({"ts": ts, "dir": "bullish", "level": s.price})
    return out


def support_resistance(df: pd.DataFrame, sw: list[Swing], price: float, atr_v: float, n: int = 3) -> dict:
    """Cluster swing prices within 0.5 ATR; rank clusters by touches."""
    levels = sorted(s.price for s in sw[-40:])
    clusters: list[list[float]] = []
    for p in levels:
        if clusters and abs(p - np.mean(clusters[-1])) <= 0.5 * atr_v:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    lv = [(float(np.mean(c)), len(c)) for c in clusters]
    sup = sorted([x for x in lv if x[0] < price], key=lambda x: price - x[0])[:n]
    res = sorted([x for x in lv if x[0] > price], key=lambda x: x[0] - price)[:n]
    return {"support": [round(x[0], 4) for x in sup], "resistance": [round(x[0], 4) for x in res]}


def zones(df: pd.DataFrame, atr_s: pd.Series, lookback: int = 80) -> list[dict]:
    d = df.iloc[-lookback:]
    a = atr_s.reindex(d.index)
    out = []
    for i in range(1, len(d)):
        body = d["close"].iloc[i] - d["open"].iloc[i]
        if pd.isna(a.iloc[i]) or abs(body) < 1.5 * a.iloc[i]:
            continue
        prev = d.iloc[i - 1]
        if body > 0 and prev["close"] < prev["open"]:
            out.append({"type": "demand", "top": float(prev["high"]), "bottom": float(prev["low"]), "ts": d.index[i - 1]})
        if body < 0 and prev["close"] > prev["open"]:
            out.append({"type": "supply", "top": float(prev["high"]), "bottom": float(prev["low"]), "ts": d.index[i - 1]})
    return out[-4:]


def rsi_divergence(df: pd.DataFrame, sw: list[Swing]) -> str | None:
    if "rsi14" not in df:
        return None
    highs = [s for s in sw if s.kind == "high"][-2:]
    lows = [s for s in sw if s.kind == "low"][-2:]
    r = df["rsi14"].to_numpy()
    if len(highs) == 2 and highs[1].price > highs[0].price and r[highs[1].idx] < r[highs[0].idx]:
        if highs[1].idx >= len(df) - 15:
            return "bearish"
    if len(lows) == 2 and lows[1].price < lows[0].price and r[lows[1].idx] > r[lows[0].idx]:
        if lows[1].idx >= len(df) - 15:
            return "bullish"
    return None


def candle_signal(df: pd.DataFrame) -> str | None:
    """Last *completed* candle: engulfing or pin bar."""
    if len(df) < 2:
        return None
    p, c = df.iloc[-2], df.iloc[-1]
    rng = c["high"] - c["low"]
    if rng <= 0:
        return None
    body = abs(c["close"] - c["open"])
    if c["close"] > c["open"] and p["close"] < p["open"] and c["close"] >= p["open"] and c["open"] <= p["close"]:
        return "bullish_engulfing"
    if c["close"] < c["open"] and p["close"] > p["open"] and c["close"] <= p["open"] and c["open"] >= p["close"]:
        return "bearish_engulfing"
    lower_wick = min(c["open"], c["close"]) - c["low"]
    upper_wick = c["high"] - max(c["open"], c["close"])
    if lower_wick >= 2 * body and lower_wick >= 0.6 * rng:
        return "bullish_pin"
    if upper_wick >= 2 * body and upper_wick >= 0.6 * rng:
        return "bearish_pin"
    return None


def volatility_regime(atr_s: pd.Series, window: int = 120) -> dict:
    s = atr_s.dropna().iloc[-window:]
    if len(s) < 20:
        return {"regime": "unknown", "percentile": None}
    pct = float((s < s.iloc[-1]).mean() * 100)
    regime = "high" if pct >= 85 else "low" if pct <= 15 else "normal"
    return {"regime": regime, "percentile": round(pct, 1)}


def breakout_retest(df: pd.DataFrame, level: float | None, direction: str, atr_v: float, bars: int = 12) -> bool:
    """Price closed beyond `level` within the last `bars` bars and has since
    come back within 0.3 ATR of it without closing back through it."""
    if level is None:
        return False
    d = df.iloc[-bars:]
    if direction == "bullish":
        broke = (d["close"] > level).any()
        retest = (d["low"].iloc[-3:] <= level + 0.3 * atr_v).any() and d["close"].iloc[-1] > level
    else:
        broke = (d["close"] < level).any()
        retest = (d["high"].iloc[-3:] >= level - 0.3 * atr_v).any() and d["close"].iloc[-1] < level
    return bool(broke and retest)


def to_jsonable(x):
    if isinstance(x, Swing):
        return asdict(x)
    return x
