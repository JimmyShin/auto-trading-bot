Summary
- add optional `state_store` and `notify_func` DI hooks to `EmergencyManager`
- allow tests to inject fakes/stubs for UTC guard dedupe and kill-switch sequencing

Rationale
- tests (and future automation) need deterministic guard behavior without touching production defaults
- keeping the DI optional preserves existing runtime paths

Risks
- low: defaults untouched when hooks omitted
- guard dedupe still respects existing behavior; new state writes only when DI used

Testing
- .venv/Scripts/pytest -m "not regression" -vv -ra

Rollback
- revert commit `emergency: add optional DI hooks for guard tests`
