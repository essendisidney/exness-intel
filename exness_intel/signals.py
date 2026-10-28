"""Signal decision engine, strategy v1 (trend-pullback with multi-timeframe
confirmation). This is a documented rule set, NOT a tested-profitable
strategy; the journal and weekly evaluation exist to find out.

Rules (long; short is the mirror image)
  1. Direction : D1 and H4 EMA stacks agree (close>EMA50>EMA200, EMA20>EMA50).
  2. Setup     : on H1, price pulled back within 1 ATR of EMA20/EMA50 in the
                 last 6 bars, OR a breakout-retest of the last H4 swing high,
                 OR price sits in an open bullish FVG / demand zone.
  3. Confirm   : on M15, at least 2 of: RSI(14) > 50, MACD hist > 0,
                 bullish engulfing/pin on the last completed candle.
  4. Room      : distance to nearest H4 resistance >= 2R.
  5. Filters   : market open, data fresh, no high-impact event for the
                 instrument within +/-90 min, score >= 60.
Stop  : below the last confirmed H1 swing low minus 0.2 ATR, at least 1 ATR.
TP1   : 2R.  TP2: nearest H4 resistance beyond TP1, else 3R.
Expiry: 12 h after issue if not triggered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config, structure as st
from .data import Bundle, market_status, data_age_s
from .indicators import add_all
from .macro import blackout, fundamental_bias, next_event
from .risk import size_trade, reward_risk

UTC = timezone.utc
TF_MIN = {"D1": 1440, "H4": 240, "H1": 60, "M15": 15}
MIN_BARS = {"D1": 210, "H4": 210, "H1": 210, "M15": 120}
DIGITS = {"XAUUSD": 2, "XAGUSD": 3, "USOIL": 3, "UKOIL": 3}
SCORE_THRESHOLD = 60
EXPIRY_H = 12

WEIGHTS = {
    "htf_trend": 20, "technical": 15, "structure": 15, "volatility": 10, "fundamental": 10,
    "costs": 10, "reward_risk": 10, "event_risk": 5, "liquidity": 5,
}


def completed(df: pd.DataFrame, tf: str, now: datetime) -> pd.DataFrame:
    """Drop the still-forming bar (no look-ahead)."""
    if df is None or not len(df):
        return df
    close_time = df.index[-1] + pd.Timedelta(minutes=TF_MIN[tf])
    return df.iloc[:-1] if close_time > now else df


def ema_bias(row) -> int:
    if pd.isna(row["ema200"]):
        return 0
    if row["close"] > row["ema50"] > row["ema200"] and row["ema20"] > row["ema50"]:
        return 1
    if row["close"] < row["ema50"] < row["ema200"] and row["ema20"] < row["ema50"]:
        return -1
    return 0


def session_liquidity(now: datetime) -> tuple[int, str]:
    h = now.astimezone(UTC).hour
    if 7 <= h < 20:
        return 5, "London/New York hours"
    if h >= 22 or h < 7:
        return 3, "Asian session"
    return 0, "pre-close / thin liquidity"


@dataclass
class Signal:
    key: str
    broker_symbol: str
    ts: str
    signal: str = "WAIT"            # BUY | SELL | WAIT
    state: str = "NO SETUP"         # NO SETUP | DEVELOPING | CONFIRMED | DATA UNAVAILABLE | MARKET CLOSED
    direction_text: str = "n/a"
    regime: str = "n/a"
    price: float | None = None
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    entry_trigger: str = ""
    entry: float | None = None
    stop: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    rr: float | None = None
    rr_to_obstacle: float | None = None
    holding: str = ""
    invalidation: str = ""
    expires_utc: str | None = None
    event: str = ""
    supporting: list = field(default_factory=list)
    contrary: list = field(default_factory=list)
    reasons_wait: list = field(default_factory=list)
    score: int = 0
    score_breakdown: dict = field(default_factory=dict)
    risk: dict | None = None
    executable: bool = False
    data_source: str = ""
    data_quality: list = field(default_factory=list)
    levels: dict = field(default_factory=dict)
    indicators: dict = field(default_factory=dict)


def _r(x, d):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def analyze(b: Bundle, now: datetime, cal_events: list, xa: dict, equity: float,
            daily_loss_used: float = 0.0, gsr: float | None = None) -> Signal:
    inst = b.instrument
    d = DIGITS.get(b.key, 3)
    s = Signal(key=b.key, broker_symbol=inst.broker_symbol, ts=now.isoformat(),
               data_source=b.source_detail, data_quality=list(b.warnings))

    if not b.available:
        s.state = "DATA UNAVAILABLE"
        s.reasons_wait.append("no market data")
        return s

    frames = {}
    for tf in ("D1", "H4", "H1", "M15"):
        df = completed(b.frames.get(tf), tf, now)
        if df is None or len(df) < MIN_BARS[tf]:
            s.state = "DATA UNAVAILABLE"
            s.reasons_wait.append(f"insufficient {tf} history ({0 if df is None else len(df)} bars)")
            return s
        frames[tf] = add_all(df)
    D1, H4, H1, M15 = (frames[k] for k in ("D1", "H4", "H1", "M15"))
    off = b.broker_offset
    last = float(M15["close"].iloc[-1])
    s.price = _r(last + off, d)
    if b.quote:
        s.bid, s.ask = _r(b.quote.bid, d), _r(b.quote.ask, d)
        s.spread = _r(b.quote.spread, d + 1) if b.quote.spread is not None else None

    # ---- context levels
    atr_h1, atr_m15, atr_d1 = float(H1["atr14"].iloc[-1]), float(M15["atr14"].iloc[-1]), float(D1["atr14"].iloc[-1])
    today = now.astimezone(UTC).date()
    m15_today = M15[M15.index.date == today]
    prev_day = D1.iloc[-1] if D1.index[-1].date() < today else D1.iloc[-2]
    ms_h4, ms_h1, ms_d1 = st.market_structure(H4), st.market_structure(H1), st.market_structure(D1)
    sr = st.support_resistance(H4, ms_h4["swings"], last, float(H4["atr14"].iloc[-1]))
    fvg = st.fair_value_gaps(H1)
    zn = st.zones(H1, H1["atr14"])
    vol = st.volatility_regime(H1["atr14"])
    sweeps = st.liquidity_sweeps(H1, ms_h1["swings"])
    div = st.rsi_divergence(H1, ms_h1["swings"])
    candle = st.candle_signal(M15)

    def sh(x):
        return None if x is None else _r(x + off, d)

    s.levels = {
        "prev_day_high": sh(prev_day["high"]), "prev_day_low": sh(prev_day["low"]),
        "session_high": sh(m15_today["high"].max()) if len(m15_today) else None,
        "session_low": sh(m15_today["low"].min()) if len(m15_today) else None,
        "support": [sh(x) for x in sr["support"]], "resistance": [sh(x) for x in sr["resistance"]],
        "h4_swing_high": sh(ms_h4["last_swing_high"]), "h4_swing_low": sh(ms_h4["last_swing_low"]),
        "fvg_h1": [{"dir": g["dir"], "top": sh(g["top"]), "bottom": sh(g["bottom"])} for g in fvg],
        "zones_h1": [{"type": z["type"], "top": sh(z["top"]), "bottom": sh(z["bottom"])} for z in zn],
        "latest_m15_bar_utc": M15.index[-1].isoformat(),
    }
    s.indicators = {
        tf: {"close": _r(f["close"].iloc[-1] + off, d), "ema20": _r(f["ema20"].iloc[-1] + off, d),
             "ema50": _r(f["ema50"].iloc[-1] + off, d), "ema200": _r(f["ema200"].iloc[-1] + off, d),
             "rsi14": _r(f["rsi14"].iloc[-1], 1), "macd_hist": _r(f["hist"].iloc[-1], d + 1),
             "atr14": _r(f["atr14"].iloc[-1], d), "bb_upper": _r(f["bb_upper"].iloc[-1] + off, d),
             "bb_lower": _r(f["bb_lower"].iloc[-1] + off, d), "bias": ema_bias(f.iloc[-1])}
        for tf, f in frames.items()
    }
    s.indicators["structure"] = {
        "D1": ms_d1["trend"], "H4": ms_h4["trend"], "H1": ms_h1["trend"],
        "H4_last_event": (f"{ms_h4['last_event']['type']} {ms_h4['last_event']['dir']} @ "
                          f"{sh(ms_h4['last_event']['level'])}") if ms_h4["last_event"] else None,
        "H1_divergence": div, "H1_sweeps": len(sweeps), "M15_candle": candle,
        "volatility": vol,
    }

    d1b, h4b = ema_bias(D1.iloc[-1]), ema_bias(H4.iloc[-1])
    direction = d1b if (d1b == h4b and d1b != 0) else 0

    # ---- regime label
    recent_choch = ms_h4["last_event"] and ms_h4["last_event"]["type"] == "CHoCH" and \
        ms_h4["last_event"]["ts"] >= H4.index[-6]
    bbw = H4["bb_width"].dropna()
    squeeze = len(bbw) > 50 and bbw.iloc[-1] <= bbw.iloc[-120:].quantile(0.15)
    broke = (len(m15_today) and (m15_today["high"].max() > prev_day["high"] or m15_today["low"].min() < prev_day["low"]))
    if vol["regime"] == "high":
        s.regime = "highly volatile"
    elif recent_choch:
        s.regime = "reversing (H4 CHoCH)"
    elif squeeze:
        s.regime = "consolidating (H4 Bollinger squeeze)"
    elif broke and direction != 0:
        s.regime = ("bullish" if direction > 0 else "bearish") + ", breaking previous-day range"
    elif direction != 0:
        s.regime = "bullish" if direction > 0 else "bearish"
    else:
        s.regime = "range-bound / mixed"
    s.direction_text = {1: "bullish", -1: "bearish", 0: "no D1/H4 agreement"}[direction]

    # ---- gates that force WAIT regardless of setup
    ms = market_status(now, b.key)
    age = data_age_s(b, now)
    stale = age is None or age > config.MAX_QUOTE_AGE_S
    bo = blackout(cal_events, b.key, now)
    ne = next_event(cal_events, b.key, now)
    if ne:
        s.event = f"{ne['title']} ({ne['country']}, {ne['impact']}) {ne['time_eat']} EAT"

    fb, fb_txt = fundamental_bias(b.key, xa, gsr)

    if direction == 0:
        s.reasons_wait.append(f"D1 bias {d1b:+d} and H4 bias {h4b:+d} do not agree")
    dir_word = "bullish" if direction >= 0 else "bearish"

    # ---- setup (H1)
    setup_notes = []
    if direction != 0:
        near = H1.iloc[-6:]
        pull = ((near["low"] <= near["ema20"] + atr_h1) & (near["high"] >= near["ema20"] - atr_h1)).any() if direction > 0 \
            else ((near["high"] >= near["ema20"] - atr_h1) & (near["low"] <= near["ema20"] + atr_h1)).any()
        if pull:
            setup_notes.append("H1 pullback to EMA20/50 zone")
        lvl = ms_h4["last_swing_high"] if direction > 0 else ms_h4["last_swing_low"]
        if st.breakout_retest(H1, lvl, dir_word, atr_h1):
            setup_notes.append("breakout-retest of H4 swing")
        for g in fvg:
            if g["dir"] == dir_word and g["bottom"] <= last <= g["top"]:
                setup_notes.append("inside open H1 FVG")
        for z in zn:
            if z["type"] == ("demand" if direction > 0 else "supply") and z["bottom"] <= last <= z["top"]:
                setup_notes.append(f"inside H1 {z['type']} zone")
        if not setup_notes:
            s.reasons_wait.append("trend agrees but no H1 pullback/retest setup")

    # ---- confirmation (M15)
    conf = []
    if direction != 0:
        r15, h15 = float(M15["rsi14"].iloc[-1]), float(M15["hist"].iloc[-1])
        if (r15 > 50) if direction > 0 else (r15 < 50):
            conf.append(f"M15 RSI {r15:.1f}")
        if (h15 > 0) if direction > 0 else (h15 < 0):
            conf.append("M15 MACD histogram aligned")
        if candle and candle.startswith(dir_word):
            conf.append(f"M15 {candle.replace('_', ' ')}")
    confirmed = len(conf) >= 2

    # ---- levels
    entry = stop = tp1 = tp2 = None
    rr_obs = None
    if direction != 0:
        spread = (b.quote.spread if b.quote and b.quote.spread is not None else inst.typical_spread)
        if confirmed:
            entry = (b.quote.ask if direction > 0 else b.quote.bid) if (b.quote and b.quote.bid) else last + off
            trig_txt = "Confirmed on last completed M15 candle; entry at market"
        else:
            m15ms = st.market_structure(M15)
            lv = m15ms["last_swing_high"] if direction > 0 else m15ms["last_swing_low"]
            lv = lv if lv is not None else last
            entry = lv + off + (spread if direction > 0 else 0)
            trig_txt = (f"M15 close {'above' if direction > 0 else 'below'} {lv + off:.{d}f} with RSI "
                        f"{'>' if direction > 0 else '<'}50 and MACD hist {'>' if direction > 0 else '<'}0")
        swing = ms_h1["last_swing_low"] if direction > 0 else ms_h1["last_swing_high"]
        if direction > 0:
            cand = (swing + off - 0.2 * atr_h1) if swing is not None and swing + off < entry else entry - 1.5 * atr_h1
            stop = min(cand, entry - 1.0 * atr_h1)
        else:
            cand = (swing + off + 0.2 * atr_h1) if swing is not None and swing + off > entry else entry + 1.5 * atr_h1
            stop = max(cand, entry + 1.0 * atr_h1)
        R = abs(entry - stop)
        tp1 = entry + 2 * R * direction
        obstacles = [x for x in (s.levels["resistance"] if direction > 0 else s.levels["support"]) if x is not None]
        obstacles = [x for x in obstacles if (x - entry) * direction > 0]
        nearest = min(obstacles, key=lambda x: abs(x - entry)) if obstacles else None
        rr_obs = reward_risk(entry, stop, nearest) if nearest is not None else None
        beyond = [x for x in obstacles if (x - tp1) * direction > 0]
        tp2 = min(beyond, key=lambda x: abs(x - entry)) if beyond else entry + 3 * R * direction
        s.entry_trigger = trig_txt
        s.entry, s.stop, s.tp1, s.tp2 = _r(entry, d), _r(stop, d), _r(tp1, d), _r(tp2, d)
        s.rr = reward_risk(entry, stop, tp1)
        s.rr_to_obstacle = rr_obs
        s.holding = "intraday to 2 days (H1 setup)"
        s.invalidation = (f"H1 close {'below' if direction > 0 else 'above'} {s.stop} or an H4 CHoCH against "
                          f"the trend; expires {EXPIRY_H}h after issue if not triggered")
        s.expires_utc = (now + timedelta(hours=EXPIRY_H)).isoformat()

    # ---- scoring (transparent)
    bd = {}
    if direction != 0:
        struct_agree = ms_d1["trend"] == dir_word
        bd["htf_trend"] = 20 if struct_agree else 14
    else:
        bd["htf_trend"] = 0
    bd["technical"] = min(15, 5 * len(conf))
    ev = ms_h4["last_event"]
    if direction != 0 and ev and ev["dir"] == dir_word:
        bd["structure"] = 15 if ev["type"] == "BOS" else 10
    else:
        bd["structure"] = 0
    bd["volatility"] = {"normal": 10, "low": 5, "high": 3}.get(vol["regime"], 3)
    bd["fundamental"] = 5 if fb == 0 else (10 if fb == direction else 0)
    if b.quote and b.quote.spread is not None and atr_m15 > 0:
        ratio = b.quote.spread / atr_m15
        bd["costs"] = 10 if ratio < 0.1 else 6 if ratio < 0.25 else 2
    else:
        bd["costs"] = 4
    obs = rr_obs if rr_obs is not None else (s.rr or 0)
    bd["reward_risk"] = 10 if obs >= 3 else 7 if obs >= 2 else 0
    ev_soon = any(e["high"] and b.key in e["affects"] and
                  0 <= (datetime.fromisoformat(e["time_utc"]) - now).total_seconds() <= 86400 for e in cal_events)
    bd["event_risk"] = 0 if bo else (2 if ev_soon else 5)
    bd["liquidity"], liq_txt = session_liquidity(now)
    s.score = int(sum(bd.values()))
    s.score_breakdown = {k: f"{v}/{WEIGHTS[k]}" for k, v in bd.items()}

    # ---- evidence lists
    if direction != 0:
        s.supporting.append(f"D1 and H4 EMA stacks both {dir_word}")
        s.supporting += setup_notes + conf
        if ev and ev["dir"] == dir_word:
            s.supporting.append(f"H4 {ev['type']} {ev['dir']}")
        if any(x["dir"] == dir_word for x in sweeps):
            s.supporting.append("H1 liquidity sweep in trade direction")
        if fb == direction:
            s.supporting.append(f"fundamentals: {fb_txt}")
        if div and div != dir_word:
            s.contrary.append(f"H1 {div} RSI divergence")
        r1 = float(H1["rsi14"].iloc[-1])
        if (direction > 0 and r1 > 75) or (direction < 0 and r1 < 25):
            s.contrary.append(f"H1 RSI stretched ({r1:.0f})")
        if fb == -direction:
            s.contrary.append(f"fundamentals: {fb_txt}")
        if rr_obs is not None and rr_obs < 2:
            s.contrary.append(f"nearest opposing level only {rr_obs}R away")
    if vol["regime"] == "high":
        s.contrary.append(f"H1 ATR at {vol['percentile']}th percentile")

    # ---- decision
    if not ms["open"]:
        s.state = "MARKET CLOSED"
        s.reasons_wait.insert(0, f"market closed: {ms['reason']}")
    if stale:
        s.reasons_wait.insert(0, f"data stale ({'n/a' if age is None else f'{age/60:.0f} min'} old)")
    if bo:
        s.reasons_wait.append(f"high-impact event within {config.EVENT_BLACKOUT_MIN} min: {bo['title']} {bo['time_eat']} EAT")
    if direction != 0 and rr_obs is not None and rr_obs < config.RiskParams().min_reward_risk:
        s.reasons_wait.append(f"only {rr_obs}R of room to the nearest {'resistance' if direction > 0 else 'support'}")
    if direction != 0 and s.score < SCORE_THRESHOLD:
        s.reasons_wait.append(f"score {s.score} below {SCORE_THRESHOLD}")

    qualifies = direction != 0 and setup_notes and not s.reasons_wait
    if qualifies:
        s.signal = "BUY" if direction > 0 else "SELL"
        s.state = "CONFIRMED" if confirmed else "DEVELOPING"
        if b.source_kind != "mt5" and b.quote and b.quote.bid is None:
            s.state = "DEVELOPING"
            s.data_quality.append("levels are from a futures proxy with no broker quote: confirm on MT5 before acting")
    elif s.state not in ("MARKET CLOSED", "DATA UNAVAILABLE"):
        s.state = "NO SETUP"

    if s.entry is not None and s.stop is not None:
        rres = size_trade(inst, equity, s.entry, s.stop,
                          s.spread if s.spread is not None else inst.typical_spread,
                          daily_loss_used=daily_loss_used)
        s.risk = rres.as_dict()
        s.executable = bool(qualifies and s.state == "CONFIRMED" and rres.executable and b.source_kind == "mt5")
    return s
