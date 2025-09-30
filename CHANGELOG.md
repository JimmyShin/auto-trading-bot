# Changelog

## [Unreleased]
- Wired LIVE-only DD emergency path to TradingMode + StateStore, refreshed heartbeat formatting with en dash fallbacks, extended Prometheus exports/tests.

## [0.2.0] - 2025-09-26

### Added
- Persistent state store backing daily peak equity and alert dedupe stamps.
- `TradingMode` enum for live/testnet detection with helpers.
- New Prometheus gauges for equity, drawdown, and activity deltas.
- State-backed daily drawdown computation sourced from exchange snapshots.
- LIVE-only emergency guardrail that flips to testnet and dedupes per UTC day.
- Slack heartbeat formatting that renders missing values as en dash instead of NA/0.
- Comprehensive unit tests covering metrics, emergency gating, slack rendering, and Prometheus exports.

### Fixed
- Prevented NA/0 placeholders in Slack and Prometheus when data exists.
- Daily drawdown now computed from intraday equity peaks instead of hardcoded values.
- AUTO_TESTNET_ON_DD emergency alert no longer fires on testnet runs.


