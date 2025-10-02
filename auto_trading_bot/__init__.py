"""auto_trading_bot package.

Convenience exports for commonly used classes/functions.
"""

__all__: list[str] = []


def __getattr__(name: str):
    if name in ("start_metrics_server", "get_metrics_manager"):
        from .metrics import start_metrics_server, get_metrics_manager

        return {
            "start_metrics_server": start_metrics_server,
            "get_metrics_manager": get_metrics_manager,
        }[name]
    if name == "Alerts":
        from .alerts import Alerts

        return Alerts
    if name == "EmergencyManager":
        from .main import EmergencyManager

        return EmergencyManager
    if name in ("mark_restart_intent", "consume_restart_intent"):
        from .restart import mark_restart_intent, consume_restart_intent

        return {
            "mark_restart_intent": mark_restart_intent,
            "consume_restart_intent": consume_restart_intent,
        }[name]
    raise AttributeError(f"module 'auto_trading_bot' has no attribute {name!r}")

