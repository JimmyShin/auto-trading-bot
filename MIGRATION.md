Trading Logs Schema Migration (v4)

Overview
- Schema version bumped to 4 for RAW trade logs (CSV under data/<env>/trades_*.csv).
- Adds immutable entry snapshot fields to support ATR-based R and USD-based R, without fallbacks.

New RAW Columns
- schema_version: 4
- entry_ts_utc: ISO8601 (duplicates legacy timestamp)
- timeframe, leverage
- entry_atr_abs, atr_period, atr_source
- stop_basis: 'atr' | 'percent'
- stop_k, fallback_pct
- stop_distance: planned distance at entry (basis-dependent)
- risk_usdt_planned: qty * stop_distance
- exit_ts_utc (duplicates legacy exit_timestamp)
- fees_quote_actual, pnl_quote_expost
- R_atr_expost, R_usd_expost (derived from RAW at exit time)

Migration Script
- scripts/migrate_trades_schema.py migrates existing CSV files in-place.
- To run: python scripts/migrate_trades_schema.py --data-dir data

Behavior
- Fields that cannot be inferred are left blank (e.g., entry_atr_abs when not previously logged).
- stop_basis is left blank if unknown; reporting handles blanks as N/A.
- Legacy columns (risk_usdt, R_multiple) are preserved for compatibility; R_multiple is not used in reports.

