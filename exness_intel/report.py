"""Markdown report rendering (morning / evening) and the short alert text."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

EAT = timezone(timedelta(hours=3))
DISCLAIMER = ("Research output, not financial advice. No trades are placed by this system. "
              "Stops are not guaranteed; gaps and slippage can exceed planned losses.")


def _fmt(x):
    return "—" if x is None else (f"{x:,.3f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x))


def _sig_row(s):
    price = s.price if s.price is not None else "DATA UNAVAILABLE"
    if s.signal == "WAIT":
        return f"| {s.key} | {_fmt(price)} | WAIT ({s.state}) | — | — | — | — | — |"
    return (f"| {s.key} | {_fmt(price)} | **{s.signal}** ({s.state}) | {s.entry_trigger} | {_fmt(s.stop)} | "
            f"{_fmt(s.tp1)} | {_fmt(s.tp2)} | 1:{_fmt(s.rr)} |")


def _setup_block(s):
    r = s.risk or {}
    lines = [
        f"**Instrument:** {s.key} — broker symbol `{s.broker_symbol}`",
        f"**Analysis timestamp:** {datetime.fromisoformat(s.ts).astimezone(EAT):%d %b %Y %H:%M} EAT",
        f"**Current bid / ask:** {_fmt(s.bid)} / {_fmt(s.ask)}" + ("" if s.bid else " (no broker quote — bid/ask NOT AVAILABLE)"),
        f"**Market direction:** {s.direction_text} · regime: {s.regime}",
        f"**Signal:** {s.signal} — {s.state}",
        f"**Entry trigger:** {s.entry_trigger}",
        f"**Proposed entry:** {_fmt(s.entry)} · **Stop:** {_fmt(s.stop)} · **TP1:** {_fmt(s.tp1)} · **TP2:** {_fmt(s.tp2)}",
        f"**Risk/reward:** 1:{_fmt(s.rr)} to TP1; room to nearest opposing level: {_fmt(s.rr_to_obstacle)}R",
        f"**Expected holding period:** {s.holding}",
        f"**Invalidation:** {s.invalidation}",
        f"**Relevant event:** {s.event or 'none found in calendar window'}",
        f"**Supporting evidence:** " + ("; ".join(s.supporting) or "—"),
        f"**Contrary evidence:** " + ("; ".join(s.contrary) or "none found"),
        f"**Score:** {s.score}/100 (" + ", ".join(f"{k} {v}" for k, v in s.score_breakdown.items()) + ")",
        f"**Data source:** {s.data_source}",
    ]
    if r:
        lines.append(
            f"**Risk check:** equity ${r['equity']}, max loss allowed ${r['max_loss_allowed']}, stop distance "
            f"{r['stop_distance']}, contract {r['contract_size']}, tick {r['tick_size']} (value/lot ${r['tick_value_per_lot']:.2f}), "
            f"min lot {r['min_lot']} → loss at min lot ≈ ${r['loss_per_min_lot']} incl. spread ${r['spread_cost']} "
            f"& slippage ${r['slippage_cost']}; margin ≈ ${r['required_margin']}. **{r['reason']}**"
            + ("" if r["specs_verified"] else " _(contract specs UNVERIFIED — not yet confirmed from MT5)_"))
    if s.data_quality:
        lines.append("**Data notes:** " + "; ".join(s.data_quality))
    return "\n\n".join(lines)


def _levels_line(s):
    L = s.levels or {}
    i = (s.indicators or {})
    h4, d1 = i.get("H4", {}), i.get("D1", {})
    stc = i.get("structure", {})
    return (f"- **{s.key}** {_fmt(s.price)} · D1 {d1.get('bias', 0):+d} / H4 {h4.get('bias', 0):+d} "
            f"(structure D1 {stc.get('D1')}, H4 {stc.get('H4')}) · H4 RSI {_fmt(h4.get('rsi14'))} · "
            f"H1 ATR {_fmt(i.get('H1', {}).get('atr14'))} · vol {stc.get('volatility', {}).get('regime')}\n"
            f"  PDH {_fmt(L.get('prev_day_high'))} / PDL {_fmt(L.get('prev_day_low'))} · session "
            f"{_fmt(L.get('session_high'))}/{_fmt(L.get('session_low'))} · support {L.get('support')} · "
            f"resistance {L.get('resistance')} · H4 last event: {stc.get('H4_last_event')}")


def _calendar(cal):
    if not cal["available"]:
        return "Economic calendar: DATA UNAVAILABLE (source unreachable). Treat the next 24h as event-risk unknown."
    if not cal["events"]:
        return "No high-impact or tracked events in the window."
    rows = ["| Time (EAT) | Event | Ctry | Impact | Actual | Forecast | Previous | Affects |", "|---|---|---|---|---|---|---|---|"]
    for e in cal["events"][:15]:
        rows.append(f"| {e['time_eat']} | {e['title']} | {e['country']} | {e['impact']} | {e['actual'] or '—'} | "
                    f"{e['forecast'] or '—'} | {e['previous'] or '—'} | {', '.join(e['affects'])} |")
    return "\n".join(rows) + f"\n\n_Source: {cal['source']}. Entries are blocked ±90 min around high-impact events for affected instruments._"


def _positions(pr):
    out = []
    if pr.get("account"):
        a = pr["account"]
        out.append(f"Account ({'verified live' if pr['verified'] else 'last known, NOT live'}): balance "
                   f"{a['balance']} {a.get('currency') or ''}, equity {a['equity']}, free margin {a['margin_free']}, "
                   f"margin level {_fmt(a['margin_level'])}%")
    for p in pr.get("positions", []):
        ex = p.get("exposure") or {}
        out.append(f"- {p['symbol']} {p['side']} {p['volume']} @ {p['entry']} → {p['current']}, P/L {p['profit']}, "
                   f"SL {p['sl'] or 'NONE'}, TP {p['tp'] or 'none'} → **{', '.join(p['status'])}**. "
                   f"Each 1.00 price move = ${ex.get('usd_per_1_move')} ({ex.get('pct_equity_per_1_move')}% of equity).")
    if pr.get("screenshot"):
        sp, ex = pr["screenshot"], pr["screenshot"]["exposure_if_still_open"]
        out.append(f"Screenshot position (historical, unverified): {sp['symbol']} {sp['type']} {sp['volume']} @ "
                   f"{sp['price_open']}. **If still open**, every $1.00 move in WTI = ${ex['usd_per_1_move']} "
                   f"(~{ex['pct_equity_per_1_move']}% of the ${28.82} balance); a ${ex['move_to_lose_100pct']} adverse move "
                   f"would erase the whole balance (before margin stop-out, which comes sooner). "
                   f"Contract size assumed 1,000 bbl/lot (UNVERIFIED).")
    for n in pr.get("notes", []):
        out.append(f"- {n}")
    return "\n".join(out) or "No open positions in the verified snapshot."


def _overview(xa, heads, sigs):
    parts = []
    for k, v in xa.items():
        if not v:
            parts.append(f"{k} DATA UNAVAILABLE")
        else:
            parts.append("{} {} ({:+}, 20d trend {}, close of {})".format(
                k, v["last"], v["d1_change"], v["trend_20d"], v["as_of"]))
    g, sv = sigs.get("XAUUSD"), sigs.get("XAGUSD")
    if g and sv and g.price and sv.price:
        parts.append(f"gold/silver ratio {g.price / sv.price:.1f}")
    txt = "Cross-asset: " + "; ".join(parts) + "."
    if heads:
        txt += "\n\nRecent headlines (titles only, unverified beyond the publisher):\n" + "\n".join(
            f"- [{h['title']}]({h['link']}) — {h['source']}, {h['published_eat']} EAT" for h in heads[:8])
    else:
        txt += "\n\nHeadlines: none retrieved."
    return txt


def render(kind: str, now: datetime, sigs: dict, cal: dict, xa: dict, heads: list, pr: dict,
           morning_review: list | None = None, prev: dict | None = None) -> str:
    title = f"EXNESS AI — {'MORNING' if kind == 'morning' else 'EVENING'} MARKET INTELLIGENCE"
    t = now.astimezone(EAT)
    actionable = [s for s in sigs.values() if s.signal in ("BUY", "SELL")]
    best = max(actionable, key=lambda s: s.score) if actionable else None
    table = "\n".join(["| Instrument | Price | Signal | Entry trigger | Stop | TP1 | TP2 | R:R |",
                       "|---|---|---|---|---|---|---|---|"] + [_sig_row(s) for s in sigs.values()])
    waits = "\n".join(f"- {s.key}: WAIT — " + "; ".join(s.reasons_wait or ["no qualifying setup"])
                      for s in sigs.values() if s.signal == "WAIT")
    levels = "\n".join(_levels_line(s) for s in sigs.values() if s.price is not None)
    best_txt = _setup_block(best) if best else "No instrument meets every rule. Nothing to rank."

    warn = []
    for s in sigs.values():
        if s.risk and not s.risk["executable"]:
            warn.append(f"- {s.key}: {s.risk['reason']}")
        if s.state == "DATA UNAVAILABLE":
            warn.append(f"- {s.key}: DATA UNAVAILABLE — " + "; ".join(s.reasons_wait))
        if s.indicators.get("structure", {}).get("volatility", {}).get("regime") == "high":
            warn.append(f"- {s.key}: elevated volatility (H1 ATR high percentile)")
    if any(s.data_source.startswith("Yahoo") for s in sigs.values()):
        warn.append("- Prices are from delayed public futures data, not your Exness feed; CFD prices differ by a "
                    "basis/spread. Nothing here is executable until the MT5 bridge is connected.")

    executable = [s for s in actionable if s.executable]
    if executable:
        final = "Actionable (confirmed, within risk budget): " + ", ".join(f"{s.key} {s.signal}" for s in executable)
    else:
        final = "**NO TRADE.** " + ("Setups exist but none is confirmed, within the 0.5% risk budget and on verified broker data: "
                                     + ", ".join(f"{s.key} {s.signal} ({s.state})" for s in actionable)
                                     if actionable else "No instrument has a qualifying setup.")

    head = f"# {title}\n\n**{t:%A %d %B %Y, %H:%M} EAT**\n\n> {DISCLAIMER}\n"
    if kind == "morning":
        body = f"""
## A. Market overview
{_overview(xa, heads, sigs)}

**Levels and readings**
{levels}

## B. Trade signal table
{table}

{waits}

## C. Best-supported setup
{best_txt}

## D. Key economic events
{_calendar(cal)}

## E. Open-position review
{_positions(pr)}

## F. Risk warnings
{chr(10).join(warn) or '- none beyond standard risk'}

## G. Final decision
{final}
"""
    else:
        changes = []
        for k, s in sigs.items():
            p = (prev or {}).get(k)
            if p and p.get("price") and s.price:
                ch = s.price - p["price"]
                changes.append(f"- {k}: {_fmt(p['price'])} → {_fmt(s.price)} ({ch:+.3f}); signal "
                               f"{p.get('signal')}/{p.get('state')} → {s.signal}/{s.state}")
            else:
                changes.append(f"- {k}: no morning baseline stored")
        rev = "\n".join(f"- {r['signal_id']}: {r['direction']} {r['instrument']} entry {r['entry']} → **{r['status']}**"
                        + (f" ({r['outcome_r']}R paper)" if r.get("outcome_r") is not None else "")
                        for r in (morning_review or [])) or "No BUY/SELL setups were issued this morning."
        us = [e for e in cal.get("events", []) if e["country"] == "USD" and e["actual"]]
        us_txt = "\n".join(f"- {e['title']}: actual {e['actual']} vs forecast {e['forecast'] or '—'} "
                           f"(prev {e['previous'] or '—'})" for e in us) or "No US releases with published actuals in the window."
        body = f"""
## A. What changed today
{chr(10).join(changes)}

{_overview(xa, heads, sigs)}

## B. Updated signals
{table}

{waits}

## C. Morning signal review
{rev}

## D. Open-position review
{_positions(pr)}

## E. US session analysis
{us_txt}

**Levels and readings**
{levels}

## F. Overnight risk
{_calendar(cal)}

- Daily CFD break ~00:00–01:05 EAT; spreads widen around it. Friday close ~23:58 EAT; weekend gaps can jump stops.
{chr(10).join(warn)}

## G. Next-session outlook
{best_txt if best else 'Watch the levels above; a setup needs D1/H4 agreement, an H1 pullback and M15 confirmation.'}

**Decision:** {final}
"""
    return head + body


def alert_text(kind: str, now: datetime, sigs: dict, link: str | None) -> str:
    t = now.astimezone(EAT)
    lines = [f"EXNESS AI — {kind.upper()} {t:%d %b %H:%M} EAT"]
    for s in sigs.values():
        if s.signal == "WAIT":
            why = (s.reasons_wait or ["no setup"])[0]
            lines.append(f"{s.key}: WAIT ({why[:70]})")
        else:
            ex = "executable" if s.executable else ((s.risk or {}).get("reason", "")[:60] or "not executable")
            lines.append(f"{s.key}: {s.signal} {s.state} | trigger: {s.entry_trigger[:80]} | SL {s.stop} TP1 {s.tp1} | {ex}")
    if not any(s.executable for s in sigs.values()):
        lines.append("Decision: NO TRADE")
    lines.append("Risk: 0.5%/trade on a ~$29 account is ~$0.14; most minimum-lot stops exceed it.")
    if link:
        lines.append(f"Full report: {link}")
    return "\n".join(lines)
