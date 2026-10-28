"""Entry point.  python -m exness_intel.run {morning|evening|monitor|weekly}"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from . import config, macro, notify, positions, report
from .data import build_bundle, load_snapshot, snapshot_age_s
from .journal import Journal, evaluate_signal, performance
from .signals import analyze, completed

UTC = timezone.utc
EAT = timezone(timedelta(hours=3))
log = logging.getLogger("exness_intel")
ROOT = Path(os.environ.get("EXNESS_INTEL_ROOT", "."))


# ---------------------------------------------------------------------------- helpers
def gh_api(method: str, path: str, **kw):
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return None
    return requests.request(method, f"https://api.github.com/repos/{repo}{path}",
                            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                            timeout=30, **kw)


def fetch_snapshot_from_release() -> None:
    """Download the latest MT5 bridge snapshot (release 'mt5-latest' asset)."""
    r = gh_api("GET", "/releases/tags/mt5-latest")
    if r is None or r.status_code != 200:
        log.info("no mt5-latest release (bridge not configured)")
        return
    for a in r.json().get("assets", []):
        if a["name"] == "mt5_snapshot.json.gz":
            d = requests.get(a["url"], headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                                                "Accept": "application/octet-stream"}, timeout=60)
            if d.ok:
                (ROOT / "data").mkdir(exist_ok=True)
                (ROOT / "data/mt5_snapshot.json.gz").write_bytes(d.content)
                log.info("snapshot downloaded (%d bytes)", len(d.content))


def daily_loss_used(snap: dict | None) -> float:
    if not snap:
        return 0.0
    pnl = snap.get("today_closed_pnl") or 0.0
    return max(-pnl, 0.0)


def gather(now: datetime):
    if os.environ.get("GITHUB_TOKEN"):
        fetch_snapshot_from_release()
    snap = load_snapshot(str(ROOT / "data/mt5_snapshot.json.gz"))
    snap_fresh = bool(snap and snapshot_age_s(snap, now) <= config.MAX_SNAPSHOT_AGE_S)
    equity = (snap or {}).get("account", {}).get("equity") if snap_fresh else None
    equity_note = "verified (MT5)" if equity else f"UNVERIFIED — using last known balance ${config.INITIAL_BALANCE}"
    equity = equity or config.INITIAL_BALANCE
    cal = macro.fetch_calendar(now)
    xa = macro.cross_asset()
    heads = macro.headlines(now)
    bundles = {k: build_bundle(k, now, snap) for k in config.INSTRUMENTS}
    gsr = None
    try:
        g = bundles["XAUUSD"].frames["D1"]["close"].iloc[-1]
        s = bundles["XAGUSD"].frames["D1"]["close"].iloc[-1]
        gsr = float(g / s)
    except Exception:
        pass
    sigs = {k: analyze(b, now, cal["events"], xa, equity, daily_loss_used(snap), gsr) for k, b in bundles.items()}
    pr = positions.review(snap, sigs, now)
    pr.setdefault("notes", []).append(f"Equity used for sizing: {equity_note}")
    return snap, bundles, sigs, cal, xa, heads, pr


def close_old_report_issues():
    r = gh_api("GET", "/issues", params={"labels": "report", "state": "open", "per_page": 50})
    if r is None or not r.ok:
        return
    for i in r.json()[1:]:  # keep the newest open
        gh_api("PATCH", f"/issues/{i['number']}", json={"state": "closed"})


def latest_report_issue() -> int | None:
    r = gh_api("GET", "/issues", params={"labels": "report", "state": "open", "per_page": 1})
    if r is None or not r.ok or not r.json():
        return None
    return r.json()[0]["number"]


def evaluate_open(j: Journal, bundles: dict, now: datetime) -> list[tuple[dict, str, float | None]]:
    changes = []
    for sig in j.open_signals():
        b = bundles.get(sig["instrument"])
        if not b or not b.available:
            continue
        m15 = completed(b.frames["M15"], "M15", now).copy()
        for c in ("open", "high", "low", "close"):
            m15[c] = m15[c] + b.broker_offset
        status, r, ts = evaluate_signal(sig, m15, now)
        if status != sig["status"]:
            j.set_status(sig["signal_id"], status, ts, r)
            changes.append((sig, status, r))
    return changes


# ---------------------------------------------------------------------------- commands
def run_report(kind: str, now: datetime, j: Journal) -> int:
    run_id = f"{now:%Y%m%dT%H%M}-{kind}"
    snap, bundles, sigs, cal, xa, heads, pr = gather(now)
    evaluate_open(j, bundles, now)

    ctx = "; ".join(f"{e['title']} {e['time_eat']}" for e in cal["events"][:5] if e["high"]) or "no high-impact events"
    for s in sigs.values():
        j.record_signal(s, kind, ctx)

    morning_review, prev = None, None
    if kind == "evening":
        day = now.astimezone(UTC).strftime("%Y-%m-%d")
        morning_review = [r for r in j.signals_between(day, day + "T23:59") if r["session"] == "morning"]
        p = ROOT / "data/last_morning.json"
        prev = json.loads(p.read_text()) if p.exists() else None
    md = report.render(kind, now, sigs, cal, xa, heads, pr, morning_review, prev)

    t = now.astimezone(EAT)
    out = ROOT / "reports" / f"{t:%Y-%m-%d}-{kind}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    (ROOT / "reports/latest.md").write_text(md)
    if kind == "morning":
        (ROOT / "data/last_morning.json").write_text(json.dumps(
            {k: {"price": s.price, "signal": s.signal, "state": s.state} for k, s in sigs.items()}))
    (ROOT / "data/last_signals.json").write_text(json.dumps({k: s.__dict__ for k, s in sigs.items()}, default=str, indent=1))

    title = f"{'Morning' if kind == 'morning' else 'Evening'} report {t:%a %d %b %Y %H:%M} EAT — " + \
            ", ".join(f"{k} {s.signal}" for k, s in sigs.items())
    ok_i, link = notify.github_issue(title, md, ["report", kind])
    j.log_delivery(run_id, "github_issue", ok_i, link)
    if ok_i:
        close_old_report_issues()
    ok_t, det = notify.send_telegram(report.alert_text(kind, now, sigs, link if ok_i else None))
    j.log_delivery(run_id, "telegram", ok_t, det)
    j.log_run(run_id, kind, True, f"issue={ok_i} telegram={det}")
    print(md)
    print(f"\n[delivery] github_issue ok={ok_i} {link}\n[delivery] telegram ok={ok_t} {det}")
    return 0


def run_monitor(now: datetime, j: Journal) -> int:
    run_id = f"{now:%Y%m%dT%H%M}-monitor"
    snap, bundles, sigs, cal, xa, heads, pr = gather(now)
    msgs = []
    for sig, status, r in evaluate_open(j, bundles, now):
        if j.alert_once(f"{sig['signal_id']}:{status}"):
            msgs.append(f"{sig['instrument']} {sig['direction']} setup {sig['signal_id']} → {status.upper()}"
                        + (f" ({r}R paper)" if r is not None else "") + f" | entry {sig['entry']} SL {sig['stop']} TP1 {sig['tp1']}")
    for e in cal["events"]:
        mins = (datetime.fromisoformat(e["time_utc"]) - now).total_seconds() / 60
        if e["high"] and 0 <= mins <= 75 and j.alert_once(f"event:{e['title']}:{e['time_utc']}"):
            msgs.append(f"High-impact in {mins:.0f} min: {e['title']} ({e['country']}) {e['time_eat']} EAT — "
                        f"affects {', '.join(e['affects'])}. Avoid new entries.")
    for s in sigs.values():
        vol = s.indicators.get("structure", {}).get("volatility", {}) if s.indicators else {}
        if vol.get("regime") == "high" and j.alert_once(f"vol:{s.key}:{now:%Y-%m-%d}"):
            msgs.append(f"{s.key}: volatility spike (H1 ATR {vol.get('percentile')}th pct).")
    if pr.get("verified"):
        a = pr["account"] or {}
        if a.get("margin_level") and a["margin_level"] < 300 and j.alert_once(f"margin:{now:%Y-%m-%d-%H}"):
            msgs.append(f"Margin level {a['margin_level']:.0f}% — stop-out risk rising.")
        for p in pr["positions"]:
            for st_ in p["status"]:
                if st_ != "HOLD" and j.alert_once(f"pos:{p['ticket']}:{st_}:{now:%Y-%m-%d}"):
                    msgs.append(f"{p['symbol']} {p['side']} {p['volume']}: {st_}")
    if msgs:
        text = f"EXNESS AI — MONITOR {now.astimezone(EAT):%d %b %H:%M} EAT\n" + "\n".join(msgs)
        ok, det = notify.send_telegram(text)
        if not ok:
            num = latest_report_issue()
            if num:
                r = gh_api("POST", f"/issues/{num}/comments", json={"body": text})
                ok, det = bool(r is not None and r.ok), "issue comment"
        j.log_delivery(run_id, "monitor", ok, det)
        print(text)
    else:
        print("monitor: no state changes")
    j.log_run(run_id, "monitor", True, f"{len(msgs)} alerts")
    return 0


def run_weekly(now: datetime, j: Journal) -> int:
    wk = j.signals_between((now - timedelta(days=7)).isoformat(), now.isoformat())
    mtd = j.signals_between(now.strftime("%Y-%m-01"), now.isoformat())
    pw, pm = performance(wk), performance(mtd)
    fails = j.failed_deliveries(10)
    md = (f"# EXNESS AI — PERFORMANCE REVIEW {now.astimezone(EAT):%d %b %Y}\n\n"
          f"Strategy {config.STRATEGY_VERSION}. **Paper trading only** (signals evaluated against subsequent "
          f"M15 bars; stop assumed first when stop and target share a bar). Live results: NOT CONFIGURED until "
          f"MT5 deal history is exported by the bridge. Backtest: not run.\n\n"
          f"## Last 7 days\n```json\n{json.dumps(pw, indent=1)}\n```\n\n## Month to date\n```json\n"
          f"{json.dumps(pm, indent=1)}\n```\n\n## Delivery failures (last 10)\n"
          + ("\n".join(f"- {f['ts']} {f['channel']}: {f['detail']}" for f in fails) or "none") +
          "\n\nRule changes need out-of-sample evidence; a few weeks of signals is not enough to conclude anything.")
    out = ROOT / "reports" / f"{now:%Y-%m-%d}-weekly.md"
    out.write_text(md)
    ok, link = notify.github_issue(f"Weekly performance {now:%d %b %Y}", md, ["performance"])
    notify.send_telegram(f"EXNESS AI weekly review: {pw['total_signals']} signals, {pw['closed']} closed, "
                         f"expectancy {pw['expectancy_r']}R. {link if ok else ''}")
    print(md)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["morning", "evening", "monitor", "weekly"])
    ap.add_argument("--journal", default=str(ROOT / "data/journal.sqlite"))
    a = ap.parse_args(argv)
    now = datetime.now(UTC)
    j = Journal(a.journal)
    try:
        if a.kind in ("morning", "evening"):
            return run_report(a.kind, now, j)
        if a.kind == "monitor":
            return run_monitor(now, j)
        return run_weekly(now, j)
    except Exception as e:
        tb = traceback.format_exc()
        j.log_run(f"{now:%Y%m%dT%H%M}-{a.kind}", a.kind, False, tb[-1500:])
        notify.send_telegram(f"EXNESS AI {a.kind} run FAILED: {type(e).__name__}: {e}")
        print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
