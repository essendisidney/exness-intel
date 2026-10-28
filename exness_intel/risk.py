"""Capital-protection maths. Pure functions, unit-tested.

A stop is never treated as guaranteed: the loss estimate adds
`slippage_ticks` of adverse slippage. Standard accounts carry no commission;
spread cost is included because entry is at ask (buy) / bid (sell)."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from .config import Instrument, RiskParams


@dataclass
class RiskResult:
    equity: float
    max_loss_allowed: float
    stop_distance: float
    tick_size: float
    tick_value_per_lot: float
    contract_size: float
    min_lot: float
    loss_per_min_lot: float          # incl. slippage + spread
    allowed_lots: float              # 0 if below min lot
    planned_lots: float
    max_loss_at_stop: float          # for planned_lots (or min lot if not executable)
    spread_cost: float
    slippage_cost: float
    required_margin: float
    executable: bool
    reason: str
    specs_verified: bool

    def as_dict(self):
        return asdict(self)


def floor_step(x: float, step: float) -> float:
    return math.floor(round(x / step, 8)) * step


def size_trade(inst: Instrument, equity: float, entry: float, stop: float, spread: float,
               params: RiskParams = RiskParams(), daily_loss_used: float = 0.0) -> RiskResult:
    if equity <= 0:
        raise ValueError("equity must be positive")
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        raise ValueError("stop must differ from entry")
    tv = inst.tick_value_per_lot
    per_lot_stop = stop_dist / inst.tick_size * tv
    per_lot_slip = params.slippage_ticks * tv
    per_lot_spread = (spread or 0) / inst.tick_size * tv
    per_lot = per_lot_stop + per_lot_slip + per_lot_spread

    max_loss = equity * params.max_risk_per_trade_pct / 100
    daily_left = equity * params.max_daily_loss_pct / 100 - daily_loss_used
    budget = min(max_loss, max(daily_left, 0))
    allowed = floor_step(budget / per_lot, inst.lot_step) if per_lot > 0 else 0
    allowed = round(allowed, 2)
    loss_min = per_lot * inst.min_lot

    if allowed >= inst.min_lot:
        lots, ok, reason = allowed, True, "within risk budget"
    else:
        lots, ok = inst.min_lot, False
        reason = (f"NOT EXECUTABLE FOR ACCOUNT SIZE: minimum {inst.min_lot} lot loses "
                  f"~${loss_min:.2f} at this stop vs ${budget:.2f} budget")
        if daily_left <= 0:
            reason = "NOT EXECUTABLE: daily loss limit reached"
    margin = entry * inst.contract_size * lots / inst.leverage
    return RiskResult(
        equity=round(equity, 2), max_loss_allowed=round(budget, 2), stop_distance=round(stop_dist, 5),
        tick_size=inst.tick_size, tick_value_per_lot=tv, contract_size=inst.contract_size,
        min_lot=inst.min_lot, loss_per_min_lot=round(loss_min, 2), allowed_lots=allowed if ok else 0.0,
        planned_lots=lots if ok else 0.0, max_loss_at_stop=round(per_lot * lots, 2),
        spread_cost=round(per_lot_spread * lots, 2), slippage_cost=round(per_lot_slip * lots, 2),
        required_margin=round(margin, 2), executable=ok, reason=reason, specs_verified=inst.specs_verified,
    )


def position_exposure(inst: Instrument, volume: float, price: float, equity: float) -> dict:
    """Dollar P&L per 1.0 price move and how far price can move before the
    position alone loses 100% / 50% of equity."""
    per_unit = inst.contract_size * volume
    return {
        "usd_per_1_move": round(per_unit, 2),
        "move_to_lose_50pct": round(equity * 0.5 / per_unit, 3) if per_unit else None,
        "move_to_lose_100pct": round(equity / per_unit, 3) if per_unit else None,
        "pct_equity_per_1_move": round(per_unit / equity * 100, 1) if equity else None,
        "notional": round(price * per_unit, 2),
    }


def reward_risk(entry: float, stop: float, target: float) -> float:
    r = abs(entry - stop)
    return round(abs(target - entry) / r, 2) if r else 0.0
