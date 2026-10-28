import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exness_intel import config, notify, structure as st
from exness_intel.data import Bundle, build_bundle, data_age_s, instrument_from_snapshot, market_status
from exness_intel.indicators import atr, bollinger, ema, macd, rsi
from exness_intel.journal import Journal, evaluate_signal, performance
from exness_intel.risk import position_exposure, size_trade
from exness_intel.signals import analyze, completed

from .conftest import bundle_from, make_m15

UTC = timezone.utc
NOW = datetime(2026, 10, 28, 6, 5, tzinfo=UTC)  # Wednesday 09:05 EAT


# ---------------------------------------------------------------- indicators
def test_ema_matches_recursive_definition():
    s = pd.Series(np.arange(1, 51, dtype=float))
    e = ema(s, 10)
    a = 2 / 11
    v = s.iloc[0]
    for x in s.iloc[1:]:
        v = a * x + (1 - a) * v
    assert e.iloc[-1] == pytest.approx(v)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(100, dtype=float))
    assert rsi(up).iloc[-1] == pytest.approx(100)
    down = pd.Series(np.arange(100, 0, -1, dtype=float))
    assert rsi(down).iloc[-1] == pytest.approx(0)
    noisy = pd.Series(np.random.default_rng(0).normal(0, 1, 500).cumsum())
    r = rsi(noisy).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_wilder_reference():
    # Wilder's classic 14-period example closes; RSI after first average ~70.46
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
              45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
    r = rsi(pd.Series(closes))
    assert r.iloc[14] == pytest.approx(70.46, abs=0.05)


def test_macd_hist_identity():
    s = pd.Series(np.random.default_rng(1).normal(0, 1, 300).cumsum() + 100)
    m = macd(s)
    assert np.allclose((m["macd"] - m["signal"]).dropna(), m["hist"].dropna())


def test_atr_constant_range():
    n = 60
    df = pd.DataFrame({"open": [10.0] * n, "high": [11.0] * n, "low": [9.0] * n, "close": [10.0] * n})
    assert atr(df).iloc[-1] == pytest.approx(2.0)


def test_bollinger_mid_is_sma():
    s = pd.Series(np.arange(50, dtype=float))
    bb = bollinger(s)
    assert bb["mid"].iloc[-1] == pytest.approx(s.iloc[-20:].mean())


def test_fvg_detection():
    idx = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": [1, 1.1, 1.5, 1.6, 1.7], "high": [1.1, 1.5, 1.8, 1.9, 2.0],
                       "low": [0.9, 1.0, 1.3, 1.5, 1.6], "close": [1.05, 1.45, 1.7, 1.8, 1.9]}, index=idx)
    gaps = st.fair_value_gaps(df)
    assert any(g["dir"] == "bullish" and g["bottom"] == 1.1 and g["top"] == 1.3 for g in gaps)


def test_structure_bos_uptrend(up_m15):
    from exness_intel.indicators import resample_ohlc
    h4 = resample_ohlc(up_m15, "4h")
    ms = st.market_structure(h4)
    assert ms["trend"] == "bullish"


# ---------------------------------------------------------------- no look-ahead
def test_completed_drops_forming_bar():
    idx = pd.date_range("2026-10-28 05:00", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    now = datetime(2026, 10, 28, 6, 5, tzinfo=UTC)  # 06:00 bar still forming
    assert completed(df, "M15", now).index[-1] == pd.Timestamp("2026-10-28 05:45", tz="UTC")


# ---------------------------------------------------------------- freshness & closures
def test_data_freshness_stale_forces_wait(up_m15):
    b = bundle_from(up_m15)
    later = NOW + timedelta(hours=3)
    assert data_age_s(b, later) > config.MAX_QUOTE_AGE_S
    s = analyze(b, later, [], {}, 10_000)
    assert s.signal == "WAIT"
    assert any("stale" in r for r in s.reasons_wait)


def test_market_closures():
    sat = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    sun_early = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)
    sun_open = datetime(2026, 11, 1, 22, 30, tzinfo=UTC)
    brk = datetime(2026, 10, 28, 21, 30, tzinfo=UTC)
    fri_late = datetime(2026, 10, 30, 21, 30, tzinfo=UTC)
    assert not market_status(sat, "XAUUSD")["open"]
    assert not market_status(sun_early, "XAUUSD")["open"]
    assert market_status(sun_open, "XAUUSD")["open"]
    assert not market_status(brk, "USOIL")["open"]
    assert not market_status(fri_late, "USOIL")["open"]
    assert market_status(NOW, "USOIL")["open"]


def test_closed_market_never_signals(up_m15):
    sat_end = "2026-10-31 11:45"
    m = make_m15(drift=0.08, seed=3, end=sat_end)
    b = bundle_from(m)
    s = analyze(b, datetime(2026, 10, 31, 12, 0, tzinfo=UTC), [], {}, 10_000)
    assert s.signal == "WAIT" and s.state == "MARKET CLOSED"


# ---------------------------------------------------------------- symbol mapping
def test_symbol_mapping_from_snapshot(up_m15):
    spec = {"name": "XAUUSDm", "trade_contract_size": 100, "trade_tick_size": 0.001,
            "volume_min": 0.01, "volume_step": 0.01}
    inst = instrument_from_snapshot(config.INSTRUMENTS["XAUUSD"], spec)
    assert inst.broker_symbol == "XAUUSDm" and inst.specs_verified and inst.tick_size == 0.001
    assert inst.tick_value_per_lot == pytest.approx(0.1)


def test_snapshot_missing_symbol_and_proxy_fallback(up_m15):
    from .conftest import frames_from
    snap = {"generated_utc": NOW.isoformat(), "symbols": {}, "account": {}, "positions": []}
    b = build_bundle("UKOIL", NOW, snap, fetch_proxy=lambda t: frames_from(up_m15))
    assert b.source_kind == "proxy"
    assert any("not present in MT5 snapshot" in w for w in b.warnings)
    assert b.quote.bid is None  # never invents a bid/ask


def test_no_data_gives_data_unavailable():
    def boom(t):
        raise RuntimeError("down")
    b = build_bundle("XAUUSD", NOW, None, fetch_proxy=boom)
    s = analyze(b, NOW, [], {}, 28.82)
    assert s.state == "DATA UNAVAILABLE" and s.signal == "WAIT" and s.price is None


# ---------------------------------------------------------------- risk
def test_risk_small_account_not_executable():
    inst = config.INSTRUMENTS["XAUUSD"]
    r = size_trade(inst, 28.82, 4000.00, 3990.00, 0.25)
    assert r.max_loss_allowed == pytest.approx(0.14, abs=0.01)
    # 0.01 lot = 1 oz -> $10 stop + $0.25 spread + $0.20 slippage
    assert r.loss_per_min_lot == pytest.approx(10.45, abs=0.01)
    assert not r.executable and "NOT EXECUTABLE FOR ACCOUNT SIZE" in r.reason


def test_risk_large_account_sizes_down_to_budget():
    inst = config.INSTRUMENTS["XAUUSD"]
    r = size_trade(inst, 10_000, 4000.00, 3990.00, 0.25)
    assert r.executable
    assert r.max_loss_at_stop <= 50.0 + 1e-9
    assert r.planned_lots == pytest.approx(0.04)


def test_risk_daily_limit():
    inst = config.INSTRUMENTS["XAUUSD"]
    r = size_trade(inst, 10_000, 4000.00, 3990.00, 0.25, daily_loss_used=200)
    assert not r.executable and "daily loss" in r.reason


def test_screenshot_oil_exposure():
    ex = position_exposure(config.INSTRUMENTS["USOIL"], 0.02, 100.685, 28.82)
    assert ex["usd_per_1_move"] == 20
    assert ex["move_to_lose_100pct"] == pytest.approx(1.441, abs=0.01)


# ---------------------------------------------------------------- signals
def test_uptrend_produces_bullish_levels_in_order(up_m15):
    s = analyze(bundle_from(up_m15), NOW, [], {}, 10_000)
    assert s.direction_text == "bullish"
    if s.entry is not None:
        assert s.stop < s.entry < s.tp1 <= s.tp2
        assert s.rr == pytest.approx(2.0, abs=0.01)


def test_downtrend_produces_bearish_levels_in_order(down_m15):
    s = analyze(bundle_from(down_m15, quote=True), NOW, [], {}, 10_000)
    assert s.direction_text == "bearish"
    if s.entry is not None:
        assert s.stop > s.entry > s.tp1 >= s.tp2


def test_flat_market_waits(flat_m15):
    s = analyze(bundle_from(flat_m15), NOW, [], {}, 10_000)
    assert s.signal == "WAIT"


def test_event_blackout_forces_wait(up_m15):
    ev = [{"title": "FOMC Statement", "country": "USD", "impact": "High", "high": True,
           "time_utc": (NOW + timedelta(minutes=30)).isoformat(), "time_eat": "x",
           "affects": ["XAUUSD"], "forecast": "", "previous": "", "actual": ""}]
    s = analyze(bundle_from(up_m15), NOW, ev, {}, 10_000)
    assert s.signal == "WAIT"
    assert s.score_breakdown["event_risk"] == "0/5"


def test_proxy_setups_never_executable(up_m15):
    s = analyze(bundle_from(up_m15, quote=False), NOW, [], {}, 1_000_000)
    assert not s.executable
    assert s.state != "CONFIRMED"


def test_score_breakdown_documented(up_m15):
    s = analyze(bundle_from(up_m15), NOW, [], {}, 10_000)
    from exness_intel.signals import WEIGHTS
    assert set(s.score_breakdown) == set(WEIGHTS) and sum(WEIGHTS.values()) == 100


# ---------------------------------------------------------------- journal / persistence / dedupe
def test_journal_persists_and_is_idempotent(tmp_path, up_m15):
    s = analyze(bundle_from(up_m15), NOW, [], {}, 10_000)
    s.signal, s.state = "BUY", "DEVELOPING"
    s.entry, s.stop, s.tp1, s.tp2 = s.entry or 100, s.stop or 99, s.tp1 or 102, s.tp2 or 103
    p = tmp_path / "j.sqlite"
    j = Journal(p)
    sid = j.record_signal(s, "morning", "ctx")
    assert j.record_signal(s, "morning", "ctx") == sid
    j2 = Journal(p)
    rows = j2.open_signals()
    assert len(rows) == 1 and rows[0]["signal_id"] == sid and rows[0]["user_executed"] == "unknown"


def test_wait_not_journaled(tmp_path, flat_m15):
    s = analyze(bundle_from(flat_m15), NOW, [], {}, 10_000)
    assert Journal(tmp_path / "j.sqlite").record_signal(s, "morning", "") is None


def test_duplicate_alerts_suppressed(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    assert j.alert_once("k1") is True
    assert j.alert_once("k1") is False


def _bars(rows, start="2026-10-28 06:15"):
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_paper_evaluation_paths():
    base = {"ts": "2026-10-28T06:05:00+00:00", "direction": "BUY", "entry": 100, "stop": 99, "tp1": 102,
            "tp2": 103, "status": "issued", "status_ts": "x", "expires_utc": "2026-10-28T18:05:00+00:00"}
    now = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
    assert evaluate_signal(base, _bars([[99.5, 100.1, 99.4, 100], [100, 102.2, 99.9, 102]]), now)[:2] == ("tp1", 2.0)
    assert evaluate_signal(base, _bars([[99.5, 100.1, 99.4, 100], [100, 100.5, 98.9, 99]]), now)[:2] == ("stopped", -1.0)
    # stop and target in the same bar -> stop first (conservative)
    assert evaluate_signal(base, _bars([[99.5, 100.1, 99.4, 100], [100, 103.5, 98.5, 101]]), now)[0] == "stopped"
    assert evaluate_signal(base, _bars([[99.5, 99.8, 98.8, 99.0]]), now)[0] == "invalidated"
    late = datetime(2026, 10, 28, 19, 0, tzinfo=UTC)
    assert evaluate_signal(base, _bars([[99.5, 99.8, 99.2, 99.5]]), late)[0] == "expired"


def test_performance_metrics():
    rows = [{"status": "tp1", "outcome_r": 2.0, "instrument": "XAUUSD", "session": "morning", "status_ts": "1"},
            {"status": "stopped", "outcome_r": -1.0, "instrument": "XAUUSD", "session": "evening", "status_ts": "2"},
            {"status": "expired", "outcome_r": None, "instrument": "USOIL", "session": "morning", "status_ts": "3"}]
    p = performance(rows)
    assert p["win_rate"] == 50.0 and p["profit_factor"] == 2.0 and p["expectancy_r"] == 0.5
    assert p["max_drawdown_r"] == 1.0 and p["total_signals"] == 3


# ---------------------------------------------------------------- delivery
def test_telegram_not_configured(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    ok, det = notify.send_telegram("x")
    assert not ok and "NOT CONFIGURED" in det


def test_telegram_success_and_failure(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SECRETTOKEN123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")

    class R:
        def __init__(self, ok): self.ok, self.status_code, self.text = ok, 200 if ok else 500, ""
        def json(self): return {"ok": self.ok}
    calls = []
    monkeypatch.setattr(notify.requests, "post", lambda *a, **k: calls.append(k) or R(True))
    assert notify.send_telegram("hello") == (True, "sent")
    assert "SECRETTOKEN123" not in json.dumps(calls[0]["json"])  # token never in payload
    monkeypatch.setattr(notify.requests, "post", lambda *a, **k: R(False))
    assert notify.send_telegram("hello")[0] is False


def test_github_issue_not_configured(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert notify.github_issue("t", "b", [])[0] is False


# ---------------------------------------------------------------- scheduling
def test_workflow_cron_is_7am_7pm_eat_window():
    wf = (Path(__file__).parent.parent / ".github/workflows/reports.yml").read_text()
    crons = re.findall(r"cron:\s*'([^']+)'", wf)
    times = set()
    for c in crons:
        m, h = c.split()[:2]
        t = (int(h) * 60 + int(m) + 180) % 1440  # UTC -> EAT
        times.add(t)
    # within 10 minutes before 07:00 and 19:00 EAT (GitHub cron can start late, never early)
    assert any(410 <= t <= 420 for t in times) and any(1130 <= t <= 1140 for t in times)


# ---------------------------------------------------------------- no live execution anywhere
def test_no_order_execution_code():
    root = Path(__file__).parent.parent
    for p in list((root / "exness_intel").glob("*.py")) + list((root / "bridge").glob("*.py")):
        src = p.read_text()
        for fn in ("order_send", "order_check", "order_close", "positions_close", "Position.close"):
            assert fn not in src, f"{fn} found in {p.name}"


def _snapshot_from(m15, now, positions=()):
    from .conftest import frames_from
    fr = frames_from(m15)
    rates = {tf: [{"time": int(ts.timestamp()), "open": r.open, "high": r.high, "low": r.low,
                   "close": r.close, "tick_volume": r.volume} for ts, r in df.tail(300).iterrows()]
             for tf, df in fr.items()}
    last = float(m15["close"].iloc[-1])
    return {"generated_utc": now.isoformat(), "account": {"balance": 30.0, "equity": 31.5, "margin": 5,
            "margin_free": 26.5, "margin_level": 630.0, "currency": "USD", "leverage": 200},
            "positions": list(positions), "today_closed_pnl": 0.0,
            "symbols": {"USOIL": {"spec": {"name": "USOILm", "trade_contract_size": 1000, "trade_tick_size": 0.001,
                                           "volume_min": 0.01, "volume_step": 0.01},
                                  "tick": {"bid": last - 0.02, "ask": last + 0.02, "time": int(now.timestamp())},
                                  "rates": rates}}}


def test_mt5_snapshot_path_and_position_review():
    from exness_intel import positions
    m = make_m15(drift=-0.08, seed=4, start=110)
    snap = _snapshot_from(m, NOW, [{"ticket": 1, "symbol": "USOILm", "type": 0, "volume": 0.02,
                                    "price_open": 100.685, "price_current": float(m["close"].iloc[-1]),
                                    "sl": 0.0, "tp": 0.0, "profit": -3.0}])
    b = build_bundle("USOIL", NOW, snap, fetch_proxy=lambda t: (_ for _ in ()).throw(AssertionError("proxy used")))
    assert b.source_kind == "mt5" and b.instrument.broker_symbol == "USOILm" and b.instrument.specs_verified
    s = analyze(b, NOW, [], {}, 31.5)
    assert s.bid is not None and s.direction_text == "bearish"
    pr = positions.review(snap, {"USOIL": s}, NOW)
    assert pr["verified"]
    st_ = pr["positions"][0]["status"]
    assert "REVIEW STOP" in st_ and "REVIEW EXIT" in st_


def test_stale_snapshot_positions_not_live():
    from exness_intel import positions
    m = make_m15(seed=4, start=110)
    snap = _snapshot_from(m, NOW - timedelta(hours=2))
    assert positions.review(snap, {}, NOW)["verified"] is False
