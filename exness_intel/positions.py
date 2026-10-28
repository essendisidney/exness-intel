"""Open-position review. Only positions present in a fresh MT5 snapshot are
treated as live. Classifications are analytical alerts, never orders."""
from __future__ import annotations

from datetime import datetime

from . import config
from .data import snapshot_age_s
from .risk import position_exposure


def review(snapshot: dict | None, signals: dict, now: datetime) -> dict:
    out = {"verified": False, "account": None, "positions": [], "notes": []}
    if not snapshot:
        out["notes"].append("NOT CONFIGURED: no MT5 bridge snapshot. Positions cannot be verified.")
        sp = config.SCREENSHOT_POSITION
        inst = config.INSTRUMENTS[sp["symbol"]]
        ex = position_exposure(inst, sp["volume"], sp["price_open"], config.INITIAL_BALANCE)
        out["screenshot"] = {**sp, "exposure_if_still_open": ex}
        return out
    age = snapshot_age_s(snapshot, now)
    if age > config.MAX_SNAPSHOT_AGE_S:
        out["notes"].append(f"MT5 snapshot is {age/60:.0f} min old: positions shown as last known, NOT live.")
    else:
        out["verified"] = True
    acc = snapshot.get("account") or {}
    out["account"] = {k: acc.get(k) for k in ("balance", "equity", "margin", "margin_free", "margin_level",
                                                "leverage", "currency")}
    equity = acc.get("equity") or config.INITIAL_BALANCE
    groups = {}
    for p in snapshot.get("positions", []):
        key = next((k for k, i in config.INSTRUMENTS.items()
                    if p["symbol"].upper().startswith(k)), None)
        inst = config.INSTRUMENTS.get(key) if key else None
        spec = (snapshot.get("symbols", {}).get(key) or {}).get("spec") if key else None
        if inst and spec:
            from .data import instrument_from_snapshot
            inst = instrument_from_snapshot(inst, spec)
        is_buy = p["type"] in (0, "BUY", "buy")
        flags = []
        sig = signals.get(key) if key else None
        ex = position_exposure(inst, p["volume"], p["price_current"], equity) if inst else None
        if not p.get("sl"):
            flags.append("REVIEW STOP")
        if sig and sig.direction_text in ("bullish", "bearish"):
            if (sig.direction_text == "bearish") == is_buy:
                flags.append("REVIEW EXIT")
            ch = sig.indicators.get("structure", {}).get("H4_last_event") or ""
            if "CHoCH" in ch and (("bearish" in ch) == is_buy):
                flags.append("INVALIDATED")
        atr = (sig.indicators.get("H1", {}).get("atr14") if sig and sig.indicators else None)
        if p.get("tp") and atr and abs(p["tp"] - p["price_current"]) <= 0.5 * atr:
            flags.append("TARGET APPROACHING")
        if ex and ex["pct_equity_per_1_move"] and ex["pct_equity_per_1_move"] > 20:
            flags.append("REVIEW STOP")
        if not flags:
            flags.append("HOLD")
        grp = inst.correlation_group if inst else "other"
        groups[grp] = groups.get(grp, 0) + 1
        out["positions"].append({
            "ticket": p.get("ticket"), "symbol": p["symbol"], "side": "BUY" if is_buy else "SELL",
            "volume": p["volume"], "entry": p["price_open"], "current": p["price_current"],
            "sl": p.get("sl") or None, "tp": p.get("tp") or None, "profit": p.get("profit"),
            "status": sorted(set(flags), key=["INVALIDATED", "REVIEW EXIT", "REVIEW STOP",
                                              "TARGET APPROACHING", "HOLD"].index),
            "exposure": ex,
        })
    for g, n in groups.items():
        if n > config.RiskParams().max_correlated_positions:
            out["notes"].append(f"Correlated exposure: {n} open positions in {g}.")
    return out
