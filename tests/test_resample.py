import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))
from orion_app import resample_series_by_tf


def test_resample_uses_last_trading_day_for_partial_periods():
    dates = pd.date_range("2024-08-26", "2024-08-28", freq="D")
    prices = [1.0, 2.0, 3.0]

    w_dates, w_prices = resample_series_by_tf(dates, prices, "W")
    assert pd.Timestamp(w_dates[-1]) == dates[-1]
    assert w_prices[-1] == prices[-1]

    m_dates, m_prices = resample_series_by_tf(dates, prices, "M")
    assert pd.Timestamp(m_dates[-1]) == dates[-1]
    assert m_prices[-1] == prices[-1]

