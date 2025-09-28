from __future__ import annotations

DASH = "—"


def fmt_optional(value, formatter):
    return DASH if value is None else formatter(value)


def fmt_currency(amount: float) -> str:
    return f"{amount:,.2f} USDT"


def fmt_percent_ratio(ratio: float) -> str:
    return f"{ratio * 100:.1f}%"


def fmt_int(value: float) -> str:
    return f"{int(value)}"


def fmt_float2(value: float) -> str:
    return f"{value:.2f}"

