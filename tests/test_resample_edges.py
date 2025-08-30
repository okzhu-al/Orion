import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.data.timeframe import resample_series_by_tf

def test_resample_uses_last_trading_day_for_partial_periods():
    dates = pd.date_range("2024-08-26", "2024-08-28", freq="D")
    prices = [1.0, 2.0, 3.0]
    w_dates, w_prices = resample_series_by_tf(dates, prices, "W")
    assert pd.Timestamp(w_dates[-1]) == dates[-1]
    assert w_prices[-1] == prices[-1]
    m_dates, m_prices = resample_series_by_tf(dates, prices, "M")
    assert pd.Timestamp(m_dates[-1]) == dates[-1]
    assert m_prices[-1] == prices[-1]

def test_resample_aligns_volume_with_prices():
    dates = pd.date_range("2024-08-19", periods=7, freq="B")
    prices = range(len(dates))
    vols = [100] * len(dates)
    w_dates, w_prices, w_vols = resample_series_by_tf(dates, prices, "W", vols)
    assert len(w_dates) == len(w_prices) == len(w_vols)
    m_dates, m_prices, m_vols = resample_series_by_tf(dates, prices, "M", vols)
    assert len(m_dates) == len(m_prices) == len(m_vols)
