1) Current Runtime Wiring — What’s actually in use?

Reporter.apply_equity_snapshot runs both at startup and during every loop iteration when balance snapshots are fetched (`auto_trading_bot/main.py:578-632`). Alerts.evaluate and Alerts.maybe_emit_heartbeat execute alongside each snapshot, yielding a ~30s cadence via the loop heartbeat guard. Slack interactions use the new `SlackNotifier` via `slack_notify_safely` (`auto_trading_bot/alerts.py:142-149`), while the emergency helper `slack_notify.py:24-35` remains on the hot path. Docker compose injects `.env` for Slack and exposes port 9108 (`docker-compose.yml:7-16`).

| Component | New path used? | File:Line | Notes |
| Reporter.apply_equity_snapshot | Wired | auto_trading_bot/main.py:578-632 | Startup + each balance refresh |
| Alerts.evaluate | Wired | auto_trading_bot/main.py:584-632 | Runs with equity snapshot |
| Alerts.maybe_emit_heartbeat | Wired | auto_trading_bot/main.py:585-632 | Heartbeat gate ~30s |
| SlackNotifier.send | Wired | auto_trading_bot/alerts.py:142-149 | send + fallback to send_markdown |
| Legacy slack_notify helper | Legacy path | slack_notify.py:24-35 | Emergency notifications |

Verdict: Reporter / Alerts / Slack notifier wired; emergency shim still legacy.

2) Data Evidence — Are TPs being placed/filled?

Data sources (last 30 days under `data/testnet/`). No `orders_*.csv` present → TP placement data MISSING.

Trades summary (`data/testnet/trades_*.csv`):
```
status,count
CLOSED,6
OPEN,1
```
Exit reasons:
```
exit_reason,count
estimated_exit,5
NA,2
```
All exits use `estimated_exit`; no TAKE_PROFIT fills observed.

Detailed entries (`data/testnet/detailed_entries_*.csv`) show pyramiding reasons but no TP orders:
```
timestamp,symbol,side,reason
2025-09-25 12:55:50,SOL/USDT,LONG_ADD,PYRAMID_L1
...
```
Signal analysis (`data/testnet/signal_analysis_*.csv`): 75 `ENTER_*` decisions; 7,556 skips dominated by `not_new_bar`.

Stop churn: loop cancels and re-creates stops each iteration (`auto_trading_bot/main.py:771-827`). Without order logs, replacement count per episode is inferred but unquantified.

3) Episode-Level Diagnostics — Did price breach expected TP?

Trades show 6 closed episodes with positive PnL percentages (3-12%) yet no TP evidence. `state.json` retains an open short lacking TP metadata. Classification:
- no_resting_tp: 6
- insufficient_evidence: 1 (open AVAX/USDT entry from 2025-09-22)

`config.py:101-107` defines ladder triggers at +2.0R/+3.0R, but runtime code lacks TP placement logic. OHLCV per episode unavailable → MFE check limited.

Example episode: `2025-09-26 04:05:34 ETH/USDT short` (`trades_2025-09-26.csv`) exits via `estimated_exit`; no TP order found.

4) Prometheus Inventory — What can we already observe?

Metrics defined in `auto_trading_bot/metrics.py:52-72`:
| Metric | Type | Labels | Definition | Insight |
| bot_equity | Gauge | account | metrics.py:52 | Equity level |
| bot_daily_drawdown | Gauge | account | metrics.py:53 | Daily DD ratio |
| bot_avg_r_atr_30 | Gauge | account | metrics.py:54 | Rolling R |
| bot_time_drift_ms | Gauge | source | metrics.py:55 | Clock drift |
| bot_heartbeat_ts | Gauge | — | metrics.py:56 | Loop heartbeat |
| bot_trade_count_total | Counter | account | metrics.py:57 | Trades executed |
| bot_signal_emitted_total | Counter | symbol,timeframe | metrics.py:58-62 | Signal emissions |
| bot_order_errors_total | Counter | account,reason | metrics.py:63-66 | Order/auth errors |
| bot_loop_latency_ms | Histogram | — | metrics.py:68-72 | Loop latency |

PromQL sketches (read-only):
- TP vs stop rate: N/A (metric missing).
- Cancel churn ratio: `rate(bot_order_errors_total{reason="order"}[5m])` as proxy.
- Time-to-fill quantiles: N/A (no histogram).
- TP-breach-no-fill proxy: N/A (no TP placement metric).

5) Root Cause Analysis — Ranked hypotheses with evidence

1. Pyramiding adds lack TP placement.
   - Signal: No TP orders recorded; positive exits labeled `estimated_exit`.
   - Mechanism: Main loop places only stops (`auto_trading_bot/main.py:848-917`).
   - Evidence: Trades CSV, detailed entries, state.json, code review.
   - Confidence: High.

2. Stop churn prevents persistent TP orders.
   - Signal: `cancel_all` every loop.
   - Mechanism: Cancels candidate TPs; absence of re-placement leaves only stops.
   - Evidence: `auto_trading_bot/main.py:771-827`; no TP data.
   - Confidence: Medium.

3. Pyramiding ladder triggers but lacks order wiring.
   - Signal: Risk manager records PYRAMID_L1 with `final_qty` (`risk.py:193-249`).
   - Mechanism: Ladder data never translated into TP orders.
   - Evidence: Code path adds markets then re-hooks stop; trades data lacks adds/fills.
   - Confidence: Medium.

6) Minimal Fix Plan — What to implement next

- Wire TP placement immediately after entry/pyramid adds using `pyramid_locked_limit.final_qty`, ensuring proper reduceOnly and tick rounding.
  - Where: `auto_trading_bot/main.py` entry & pyramiding blocks.
  - Validation: Trades CSV shows TAKE_PROFIT fills; add counter metric for TP outcomes; log review.
  - Risk: Medium.

- Replace brute `cancel_all` with targeted updates (preserve TP order ids, adjust only when needed).
  - Where: `auto_trading_bot/main.py:771-827`.
  - Validation: Track order churn; confirm stops/TPs persist.
  - Risk: Medium.

- Recommend adding Prometheus metrics for TP placement/fill (future work for observability).
  - Where: `auto_trading_bot/metrics.py`.
  - Validation: Grafana PromQL once implemented.
  - Risk: Low.

7) Acceptance Checklist

- TP hit rate >30% within 7-day window; stop churn ratio <1.5.
- At least one episode shows TP fill when MFE exceeds first ladder level.
- Emergency alerts clean (no NA/0 fields), dedupe effective.
- Logs emit ALERTS_INPUT/HEARTBEAT/DD_CALC around entries and TP events.
