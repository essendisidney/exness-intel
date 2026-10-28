"""Persistent trading journal (SQLite, committed back to the repo by the
workflow so it survives between runs).

Lifecycle of a setup (status column):
  issued      -> signal published (DEVELOPING or CONFIRMED)
  triggered   -> price reached the entry (paper)
  tp1 / tp2 / stopped -> paper outcome
  invalidated -> stop level traded before entry
  expired     -> not triggered within the expiry window
`user_executed` is separate and only set from verified MT5 positions; we
never assume you took a signal."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import STRATEGY_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  signal_id TEXT PRIMARY KEY, ts TEXT, session TEXT, instrument TEXT, broker_symbol TEXT,
  market_price REAL, direction TEXT, state TEXT, entry_conditions TEXT, entry REAL, stop REAL,
  tp1 REAL, tp2 REAL, rr REAL, regime TEXT, economic_context TEXT, indicators TEXT,
  score INTEGER, score_breakdown TEXT, status TEXT, status_ts TEXT, outcome_r REAL,
  est_costs REAL, data_source TEXT, expires_utc TEXT, user_executed TEXT DEFAULT 'unknown',
  strategy_version TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, ts TEXT, kind TEXT, ok INTEGER, notes TEXT
);
CREATE TABLE IF NOT EXISTS deliveries (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, channel TEXT, ok INTEGER, detail TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
  key TEXT PRIMARY KEY, ts TEXT
);
"""


class Journal:
    def __init__(self, path: str | Path = "data/journal.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ------------------------------------------------------------------ signals
    def record_signal(self, sig, session: str, context: str) -> str | None:
        """Store BUY/SELL setups. Idempotent per (date, session, instrument)."""
        if sig.signal not in ("BUY", "SELL"):
            return None
        day = sig.ts[:10]
        sid = f"{day}-{session}-{sig.key}"
        cost = (sig.risk or {}).get("spread_cost", 0) + (sig.risk or {}).get("slippage_cost", 0)
        row = dict(
            signal_id=sid, ts=sig.ts, session=session, instrument=sig.key, broker_symbol=sig.broker_symbol,
            market_price=sig.price, direction=sig.signal, state=sig.state, entry_conditions=sig.entry_trigger,
            entry=sig.entry, stop=sig.stop, tp1=sig.tp1, tp2=sig.tp2, rr=sig.rr, regime=sig.regime,
            economic_context=context, indicators=json.dumps(sig.indicators, default=str), score=sig.score,
            score_breakdown=json.dumps(sig.score_breakdown), status="issued", status_ts=sig.ts,
            outcome_r=None, est_costs=cost, data_source=sig.data_source, expires_utc=sig.expires_utc,
            strategy_version=STRATEGY_VERSION, payload=json.dumps(sig.__dict__, default=str),
        )
        with self.conn() as c:
            c.execute(f"INSERT OR IGNORE INTO signals ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                      list(row.values()))
        return sid

    def open_signals(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM signals WHERE status IN ('issued','triggered') ORDER BY ts")]

    def signals_between(self, start: str, end: str) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM signals WHERE ts >= ? AND ts < ? ORDER BY ts", (start, end))]

    def set_status(self, sid: str, status: str, ts: str, outcome_r: float | None = None):
        with self.conn() as c:
            c.execute("UPDATE signals SET status=?, status_ts=?, outcome_r=? WHERE signal_id=?",
                      (status, ts, outcome_r, sid))

    def mark_user_executed(self, sid: str, value: str):
        with self.conn() as c:
            c.execute("UPDATE signals SET user_executed=? WHERE signal_id=?", (value, sid))

    # ------------------------------------------------------------------ runs / delivery / dedupe
    def log_run(self, run_id: str, kind: str, ok: bool, notes: str = ""):
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                      (run_id, datetime.now(timezone.utc).isoformat(), kind, int(ok), notes))

    def log_delivery(self, run_id: str, channel: str, ok: bool, detail: str = ""):
        with self.conn() as c:
            c.execute("INSERT INTO deliveries (ts, run_id, channel, ok, detail) VALUES (?,?,?,?,?)",
                      (datetime.now(timezone.utc).isoformat(), run_id, channel, int(ok), detail[:500]))

    def failed_deliveries(self, limit: int = 20) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM deliveries WHERE ok=0 ORDER BY id DESC LIMIT ?", (limit,))]

    def alert_once(self, key: str) -> bool:
        """True the first time `key` is seen (duplicate-alert prevention)."""
        with self.conn() as c:
            cur = c.execute("INSERT OR IGNORE INTO alerts VALUES (?,?)",
                            (key, datetime.now(timezone.utc).isoformat()))
            return cur.rowcount == 1


# ---------------------------------------------------------------------- paper evaluation
def evaluate_signal(sig: dict, m15: pd.DataFrame, now: datetime) -> tuple[str, float | None, str]:
    """Walk completed M15 bars after issue. Conservative: if stop and target
    fall in the same bar, the stop is assumed hit first. Returns
    (status, outcome_r, status_ts)."""
    issued = pd.Timestamp(sig["ts"])
    bars = m15[m15.index > issued]
    long = sig["direction"] == "BUY"
    entry, stop, tp1, tp2 = sig["entry"], sig["stop"], sig["tp1"], sig["tp2"]
    R = abs(entry - stop)
    status = sig["status"]
    exp = pd.Timestamp(sig["expires_utc"]) if sig.get("expires_utc") else None
    for ts, b in bars.iterrows():
        if status == "issued":
            if (long and b["low"] <= stop) or (not long and b["high"] >= stop):
                if not ((long and b["high"] >= entry) or (not long and b["low"] <= entry)):
                    return "invalidated", None, ts.isoformat()
            if (long and b["high"] >= entry) or (not long and b["low"] <= entry):
                status = "triggered"
            elif exp is not None and ts >= exp:
                return "expired", None, ts.isoformat()
            else:
                continue
        if status == "triggered":
            hit_stop = (long and b["low"] <= stop) or (not long and b["high"] >= stop)
            hit_tp2 = (long and b["high"] >= tp2) or (not long and b["low"] <= tp2)
            hit_tp1 = (long and b["high"] >= tp1) or (not long and b["low"] <= tp1)
            if hit_stop:
                return "stopped", -1.0, ts.isoformat()
            if hit_tp2:
                return "tp2", round(abs(tp2 - entry) / R, 2), ts.isoformat()
            if hit_tp1:
                return "tp1", round(abs(tp1 - entry) / R, 2), ts.isoformat()
    if status == "issued" and exp is not None and now >= exp:
        return "expired", None, now.isoformat()
    return status, None, sig["status_ts"]


def performance(rows: list[dict]) -> dict:
    total = len(rows)
    trig = [r for r in rows if r["status"] in ("triggered", "tp1", "tp2", "stopped")]
    closed = [r for r in rows if r["outcome_r"] is not None]
    wins = [r["outcome_r"] for r in closed if r["outcome_r"] > 0]
    losses = [r["outcome_r"] for r in closed if r["outcome_r"] <= 0]
    gross_w, gross_l = sum(wins), -sum(losses)
    eq, peak, mdd = 0.0, 0.0, 0.0
    for r in sorted(closed, key=lambda x: x["status_ts"]):
        eq += r["outcome_r"]
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    by_inst = {}
    for r in closed:
        by_inst.setdefault(r["instrument"], []).append(r["outcome_r"])
    by_sess = {}
    for r in closed:
        by_sess.setdefault(r["session"], []).append(r["outcome_r"])
    return {
        "total_signals": total, "triggered": len(trig), "closed": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "avg_win_r": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss_r": round(sum(losses) / len(losses), 2) if losses else None,
        "profit_factor": round(gross_w / gross_l, 2) if gross_l else None,
        "expectancy_r": round(sum(r["outcome_r"] for r in closed) / len(closed), 2) if closed else None,
        "max_drawdown_r": round(mdd, 2),
        "by_instrument": {k: {"n": len(v), "sum_r": round(sum(v), 2)} for k, v in by_inst.items()},
        "by_session": {k: {"n": len(v), "sum_r": round(sum(v), 2)} for k, v in by_sess.items()},
        "note": "Paper results in R multiples before costs; spread/slippage estimated per signal in est_costs. "
                "Small samples are not evidence of an edge.",
    }
