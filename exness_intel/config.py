"""Static configuration. Everything marked verified=False must be confirmed
from the MT5 bridge snapshot before it is treated as broker fact."""
from __future__ import annotations

from dataclasses import dataclass, field

STRATEGY_VERSION = "v1.0.0"
TIMEZONE = "Africa/Nairobi"


@dataclass(frozen=True)
class RiskParams:
    max_risk_per_trade_pct: float = 0.5   # % of current equity
    max_daily_loss_pct: float = 2.0       # % of current equity
    min_reward_risk: float = 2.0
    slippage_ticks: int = 20              # assumed adverse slippage on stop
    max_correlated_positions: int = 1     # e.g. one oil position (USOIL/UKOIL)


@dataclass(frozen=True)
class Instrument:
    key: str                     # internal name
    broker_symbol: str           # Exness MT5 symbol (confirm via bridge; Standard accounts may have no suffix)
    name: str
    proxy_tickers: tuple         # public-data fallbacks, first that works wins
    proxy_is_spot: tuple         # per proxy ticker: True if spot, False if futures
    contract_size: float         # units per 1.0 lot
    tick_size: float
    min_lot: float
    lot_step: float
    leverage: float              # assumed account leverage for margin estimate
    typical_spread: float        # price units, estimate when no live quote
    correlation_group: str
    specs_verified: bool = False

    @property
    def tick_value_per_lot(self) -> float:
        return self.tick_size * self.contract_size


# Exness Standard defaults as commonly published. UNVERIFIED until the bridge confirms.
INSTRUMENTS: dict[str, Instrument] = {
    "XAUUSD": Instrument("XAUUSD", "XAUUSD", "Gold vs USD",
                         ("GC=F",), (False,), 100, 0.01, 0.01, 0.01, 200, 0.25, "metals"),
    "XAGUSD": Instrument("XAGUSD", "XAGUSD", "Silver vs USD",
                         ("SI=F",), (False,), 5000, 0.001, 0.01, 0.01, 200, 0.03, "metals"),
    "USOIL": Instrument("USOIL", "USOIL", "WTI crude oil",
                        ("CL=F",), (False,), 1000, 0.001, 0.01, 0.01, 200, 0.03, "oil"),
    "UKOIL": Instrument("UKOIL", "UKOIL", "Brent crude oil",
                        ("BZ=F",), (False,), 1000, 0.001, 0.01, 0.01, 200, 0.03, "oil"),
}

MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
}

# Data freshness limits (seconds) for an *actionable* signal.
MAX_QUOTE_AGE_S = 20 * 60
MAX_SNAPSHOT_AGE_S = 30 * 60

# High-impact events: no new entries this many minutes before/after.
EVENT_BLACKOUT_MIN = 90

# Initial known position (from screenshot, NOT live data).
SCREENSHOT_POSITION = {
    "symbol": "USOIL", "type": "BUY", "volume": 0.02, "price_open": 100.685,
    "profit_at_screenshot": 2.14, "source": "user screenshot 2026-10-28 (historical, unverified)",
}
INITIAL_BALANCE = 28.82
