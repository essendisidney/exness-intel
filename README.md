# Exness AI trading intelligence (research only)

Scheduled market research for XAUUSD, XAGUSD, USOIL and UKOIL on an Exness MT5
Standard account. It writes reports, keeps a journal of every setup and grades
them on paper. **It never places, modifies or closes trades.**

## How it runs

| Job | When (EAT) | What |
|---|---|---|
| `Scheduled reports` | ~07:00 and ~19:00 daily | Morning / evening report → GitHub issue (+ Telegram if set up), journal commit |
| `Intraday monitor` | every 2 h, Sun–Fri | Trigger / stop / target / event / margin alerts, only on change |
| `Weekly performance` | Sun 08:10 | Paper results by instrument and session |
| `Tests` | every push | 37 unit tests |

All of it runs on GitHub Actions (free tier). Your PC does not need to be on
**unless** you want broker data (see MT5 bridge).

## Data sources, in order
1. **MT5 bridge snapshot** (your PC → GitHub release `mt5-latest`): live bid/ask,
   confirmed contract specs, account equity, open positions. Used if < 30 min old.
2. **Yahoo Finance futures** (GC=F, SI=F, CL=F, BZ=F), delayed. Prices differ
   from Exness CFDs by a basis, there is no bid/ask, and setups built on this are
   never marked executable.
3. Calendar: Forex Factory weekly export. Cross-asset: DXY, US 10y yield. Headlines: EIA, OilPrice, FXStreet RSS.

Missing data shows as `DATA UNAVAILABLE` and the instrument is `WAIT`.

## Strategy v1 (documented in `exness_intel/signals.py`)
D1 + H4 EMA trend agreement → H1 pullback / breakout-retest / FVG / demand
zone → M15 confirmation (2 of RSI, MACD hist, engulfing/pin). Stop beyond the H1
swing (≥1 ATR), TP1 = 2R, TP2 = next H4 level or 3R. Blocked ±90 min around
high-impact events, when <2R of room, score <60, market closed or data stale.
The 0–100 score ranks setups; it is **not a win probability**. No edge has been
demonstrated — that is what the journal is for.

## Risk rules
0.5% equity per trade, 2% per day, ≥2:1, no martingale/averaging/stacking.
Loss estimates include spread and 20 ticks of slippage. If the minimum lot
exceeds the budget the setup is `NOT EXECUTABLE FOR ACCOUNT SIZE`.

## Connect Telegram (optional)
1. In Telegram, message **@BotFather** → `/newbot` → copy the token.
2. Send any message to your new bot, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`.
3. Repo → Settings → Secrets and variables → Actions → add
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
4. Actions → Scheduled reports → Run workflow to test.

## Connect MT5 (optional, read-only)
Needs MT5 **desktop for Windows** logged in to Exness (the mobile app cannot be read).
See the docstring in `bridge/mt5_bridge.py`. Then schedule it:
```
schtasks /Create /SC MINUTE /MO 15 /TN ExnessIntelBridge /TR "pythonw C:\path\to\bridge\mt5_bridge.py --once"
```

## Verify it is working
- Actions tab: green runs at the times above.
- Issues tab: one open `report` issue (the latest); older ones are closed.
- `reports/` and `data/journal.sqlite` get a new commit after each run.
- `data/journal.sqlite` → `deliveries` table lists every delivery attempt and failure.

## Local run
```
pip install -r requirements.txt
python -m pytest -q
python -m exness_intel.run morning
```
