import pandas as pd
import pytest

from auto_trading_bot.reporter import generate_report


def make_trades_df():
    # Four trades with returns: +10%, -5%, +20%, -10%
    rows = [
        {"ts": pd.Timestamp("2024-01-01"), "side": "long", "entry": 100.0, "exit": 110.0, "qty": 1.0},
        {"ts": pd.Timestamp("2024-01-02"), "side": "long", "entry": 100.0, "exit": 95.0, "qty": 1.0},
        {"ts": pd.Timestamp("2024-01-03"), "side": "long", "entry": 50.0, "exit": 60.0, "qty": 1.0},
        {"ts": pd.Timestamp("2024-01-04"), "side": "long", "entry": 200.0, "exit": 180.0, "qty": 1.0},
    ]
    return pd.DataFrame(rows)


def test_win_rate_expectancy_mdd():
    trades = make_trades_df()
    df = generate_report(trades, config=None)
    assert len(df) == 1

    # Win rate: 2 wins out of 4 = 0.5
    assert df.loc[0, "win_rate"] == pytest.approx(0.5, rel=1e-6)

    # Expectancy: win_rate*avg_win - loss_rate*avg_loss
    # avg_win = (0.10 + 0.20)/2 = 0.15
    # avg_loss = abs((-0.05 + -0.10)/2) = 0.075
    # expectancy = 0.5*0.15 - 0.5*0.075 = 0.0375
    assert df.loc[0, "expectancy"] == pytest.approx(0.0375, rel=1e-6)

    # MDD computed from equity curve of cumulative (1 + r)
    # Equity after trades: 1.1 -> 1.045 -> 1.254 -> 1.129
    # Max drawdown from peak 1.254 to 1.129 ~ 0.09968
    assert df.loc[0, "mdd"] == pytest.approx(0.1, rel=1e-2)


