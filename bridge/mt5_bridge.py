"""Read-only MT5 bridge — runs on YOUR Windows PC next to the MT5 desktop terminal.

What it does
  * Attaches to the MT5 terminal that is already running and logged in to
    your Exness account (you log in yourself; this script never sees or asks
    for a password, OTP or investor password).
  * Reads: account balance/equity/margin, open positions, today's closed P&L,
    symbol specifications, live bid/ask, and recent D1/H4/H1/M15 bars.
  * Uploads one compressed JSON file to a GitHub release (tag `mt5-latest`)
    in your private repo, replacing the previous one.

What it never does
  * Place, modify or close orders. The trading functions of the MetaTrader5
    package are not imported or called anywhere in this file (a test checks).

Setup (once)
  1. Install MT5 desktop for Windows and log in to Exness.
     Tools > Options > Expert Advisors: tick "Allow algorithmic trading" is NOT
     required for reading; leave it off.
  2. pip install MetaTrader5 requests
  3. Create a fine-grained GitHub token limited to this one repository with
     "Contents: Read and write" only. Set it as a user environment variable:
        setx EXNESS_INTEL_GH_TOKEN "github_pat_..."
        setx EXNESS_INTEL_REPO "essendisidney/exness-intel"
  4. Test:  python mt5_bridge.py --once --dry-run
  5. Schedule with Windows Task Scheduler every 15 minutes (see README).

Dependency: snapshots only flow while this PC is on and MT5 is running.
When they stop, the cloud reports fall back to public futures data and say so.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

WANTED = {
    "XAUUSD": ["XAUUSD", "XAUUSDm", "XAUUSDc", "XAUUSD.", "GOLD"],
    "XAGUSD": ["XAGUSD", "XAGUSDm", "XAGUSDc", "XAGUSD.", "SILVER"],
    "USOIL": ["USOIL", "USOILm", "USOILc", "WTI", "XTIUSD"],
    "UKOIL": ["UKOIL", "UKOILm", "UKOILc", "BRENT", "XBRUSD"],
}
BARS = 300


def collect(mt5) -> dict:
    tfs = {"D1": mt5.TIMEFRAME_D1, "H4": mt5.TIMEFRAME_H4, "H1": mt5.TIMEFRAME_H1, "M15": mt5.TIMEFRAME_M15}
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError(f"account_info failed: {mt5.last_error()} — is the terminal logged in?")
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "account": {k: getattr(acc, k) for k in
                    ("login", "currency", "leverage", "balance", "equity", "margin", "margin_free",
                     "margin_level", "server")},
        "symbols": {}, "positions": [], "symbol_map": {},
    }
    for key, candidates in WANTED.items():
        name = next((c for c in candidates if mt5.symbol_info(c) is not None), None)
        if name is None:
            out["symbol_map"][key] = None
            continue
        mt5.symbol_select(name, True)
        info = mt5.symbol_info(name)
        tick = mt5.symbol_info_tick(name)
        rates = {}
        for tf, code in tfs.items():
            r = mt5.copy_rates_from_pos(name, code, 0, BARS)
            if r is not None:
                rates[tf] = [{"time": int(x["time"]), "open": float(x["open"]), "high": float(x["high"]),
                              "low": float(x["low"]), "close": float(x["close"]),
                              "tick_volume": float(x["tick_volume"])} for x in r]
        out["symbol_map"][key] = name
        out["symbols"][key] = {
            "spec": {"name": name, "digits": info.digits, "trade_contract_size": info.trade_contract_size,
                     "trade_tick_size": info.trade_tick_size, "trade_tick_value": info.trade_tick_value,
                     "volume_min": info.volume_min, "volume_step": info.volume_step,
                     "volume_max": info.volume_max, "spread_points": info.spread,
                     "margin_initial": info.margin_initial, "trade_mode": info.trade_mode},
            "tick": {"bid": tick.bid, "ask": tick.ask, "time": int(tick.time)} if tick else None,
            "rates": rates,
        }
    for p in mt5.positions_get() or []:
        out["positions"].append({k: getattr(p, k) for k in
                                 ("ticket", "symbol", "type", "volume", "price_open", "price_current",
                                  "sl", "tp", "profit", "swap", "time")})
    day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    deals = mt5.history_deals_get(day, datetime.now(timezone.utc)) or []
    out["today_closed_pnl"] = round(sum(d.profit + d.swap + d.commission for d in deals if d.entry == 1), 2)
    return out


def upload(payload: bytes, repo: str, token: str) -> str:
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    api = f"https://api.github.com/repos/{repo}"
    r = requests.get(f"{api}/releases/tags/mt5-latest", headers=h, timeout=30)
    if r.status_code == 404:
        r = requests.post(f"{api}/releases", headers=h, timeout=30, json={
            "tag_name": "mt5-latest", "name": "MT5 snapshot (auto-replaced)", "prerelease": True,
            "body": "Read-only account/market snapshot uploaded by bridge/mt5_bridge.py."})
    r.raise_for_status()
    rel = r.json()
    for a in rel.get("assets", []):
        if a["name"] == "mt5_snapshot.json.gz":
            requests.delete(f"{api}/releases/assets/{a['id']}", headers=h, timeout=30)
    up = rel["upload_url"].split("{")[0] + "?name=mt5_snapshot.json.gz"
    r = requests.post(up, headers={**h, "Content-Type": "application/gzip"}, data=payload, timeout=60)
    r.raise_for_status()
    return r.json()["browser_download_url"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run once and exit (use with Task Scheduler)")
    ap.add_argument("--dry-run", action="store_true", help="print summary, do not upload")
    ap.add_argument("--every", type=int, default=900, help="seconds between snapshots in loop mode")
    a = ap.parse_args()
    import MetaTrader5 as mt5  # Windows only

    repo, token = os.environ.get("EXNESS_INTEL_REPO"), os.environ.get("EXNESS_INTEL_GH_TOKEN")
    if not a.dry_run and (not repo or not token):
        sys.exit("Set EXNESS_INTEL_REPO and EXNESS_INTEL_GH_TOKEN first (see docstring).")
    while True:
        if not mt5.initialize():
            print("MT5 initialize failed:", mt5.last_error())
        else:
            try:
                snap = collect(mt5)
                body = gzip.compress(json.dumps(snap, default=str).encode())
                acc = snap["account"]
                print(f"{snap['generated_utc']} equity={acc['equity']} positions={len(snap['positions'])} "
                      f"symbols={snap['symbol_map']} size={len(body)}B")
                if not a.dry_run:
                    print("uploaded:", upload(body, repo, token))
            except Exception as e:
                print("snapshot failed:", e)
            finally:
                mt5.shutdown()
        if a.once or a.dry_run:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
