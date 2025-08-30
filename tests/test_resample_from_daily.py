import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pandas.testing as pdt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.data.timeframe import resample_ohlcv_daily_to_tf


def test_proxy_weekly_aggregation_close_volume():
    dates = pd.date_range('2023-01-02', periods=8, freq='B')
    df = pd.DataFrame({
        '日期': dates,
        '收盘': np.arange(1, 9, dtype=float),
        '成交量': np.arange(10, 90, 10, dtype=float),
    })
    out = resample_ohlcv_daily_to_tf(df, 'W')
    expected = pd.DataFrame({
        'open': [1., 6.],
        'high': [5., 8.],
        'low': [1., 6.],
        'close': [5., 8.],
        'volume': [150., 210.],
    }, index=[pd.Timestamp('2023-01-06'), pd.Timestamp('2023-01-11')])
    pdt.assert_frame_equal(out, expected, check_freq=False)


def test_month_index_corrected_for_incomplete_month():
    dates = pd.date_range('2023-01-25', periods=8, freq='B')
    df = pd.DataFrame({'date': dates, 'close': np.arange(8, dtype=float)})
    out = resample_ohlcv_daily_to_tf(df, 'M')
    assert out.index[-1] == pd.Timestamp('2023-02-03')


def test_true_ohlcv_weekly_matches_manual():
    dates = pd.date_range('2023-03-06', periods=5, freq='B')
    df = pd.DataFrame({
        'date': dates,
        'open': [10., 11., 12., 13., 14.],
        'high': [11., 12., 13., 14., 15.],
        'low': [9., 10., 11., 12., 13.],
        'close': [10.5, 11.5, 12.5, 13.5, 14.5],
        'volume': [100., 200., 150., 300., 250.],
    })
    out = resample_ohlcv_daily_to_tf(df, 'W')
    expected = pd.DataFrame({
        'open': [10.],
        'high': [15.],
        'low': [9.],
        'close': [14.5],
        'volume': [1000.],
    }, index=[pd.Timestamp('2023-03-10')])
    expected.index.name = 'date'
    pdt.assert_frame_equal(out, expected, check_freq=False)


def test_missing_volume_outputs_zero():
    dates = pd.date_range('2023-04-03', periods=5, freq='B')
    df = pd.DataFrame({'date': dates, 'close': [1, 2, 3, 4, 5]})
    out = resample_ohlcv_daily_to_tf(df, 'W')
    assert (out['volume'] == 0).all()
    assert out.loc[out.index[0], 'open'] == 1
    assert out.loc[out.index[0], 'close'] == 5
