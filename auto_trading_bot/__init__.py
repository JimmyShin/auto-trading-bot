"""auto_trading_bot package.

Convenience exports for commonly used classes/functions.
"""

from .exchange_api import (
    ExchangeAPI,
    BalanceAuthError,
    BalanceSyncError,
    EquitySnapshot,
    EquitySnapshotError,
)  # noqa: F401
from .reporter import Reporter  # noqa: F401
from .metrics import start_metrics_server, get_metrics_manager  # noqa: F401
from .alerts import slack_notify_safely, slack_notify_exit, start_alert_scheduler  # noqa: F401
from .main import EmergencyManager, startup_sync, emergency_cleanup  # noqa: F401

