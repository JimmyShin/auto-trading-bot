Metrics Dictionary

Key Derived Metrics
- avg_r_atr: mean of R_atr_expost over selected window (no fallback). N/A if no trades provide ATR-based R.
- avg_r_usd: mean of R_usd_expost over selected window.
- win_rate: fraction of trades with positive pnl_quote_expost.
- expectancy_usd: win_rate * avg_win_usd - (1 - win_rate) * avg_loss_usd.
- fallback_percent_count: count of trades in window that used stop_basis='percent'.

R Definitions
- R_atr_expost = pnl_distance_expost / (stop_k * entry_atr_abs)
  - pnl_distance_expost = exit_price - entry_price (long), entry_price - exit_price (short)
  - Uses ATR snapshot at entry; never recomputes ATR at exit.
- R_usd_expost = pnl_quote_expost / risk_usdt_planned
  - pnl_quote_expost = gross_pnl_quote - fees_quote_actual
  - risk_usdt_planned = qty * stop_distance recorded at entry

Rolling Window (N)
- Default N=30 for Slack exit summaries.
  - Slack shows: Win rate(30), Avg R (ATR-only), Exp(USD 30), Equity, DD, fallback_trades_30 (X/N).

Null Handling
- Missing numeric values are kept blank (N/A) rather than coerced to 0.
- Avg R never substitutes USD-R or expectancy when ATR-based R is missing.

