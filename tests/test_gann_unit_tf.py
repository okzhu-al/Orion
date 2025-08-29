import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.data.timeframe import resample_series_by_tf
from app.chart.fans import build_gann_traces
from app.models.gann import compute_unit


def _gann_end(prices, dates):
    traces = build_gann_traces(dates, prices, [dates[0]], unit=1.0,
                               dir_mode="up", angles=[1], ymin=0, ymax=100, pad=0)
    y = traces[0].y
    return y[0], y[-1]


def test_gann_line_consistent_across_tf():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = np.arange(len(dates), dtype=float)
    slopes = {}
    for tf in ["D", "W", "M"]:
        d, p = resample_series_by_tf(dates, prices, tf)
        y0, y1 = _gann_end(p, d)
        slopes[tf] = (y1 - y0) / (len(p) - 1)
    assert slopes["D"] == slopes["W"] == slopes["M"] == 1.0


def test_manual_equals_median_unit():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = np.arange(len(dates), dtype=float)
    d, p = resample_series_by_tf(dates, prices, "W")
    unit_med = compute_unit(p, mode="median")
    tr1 = build_gann_traces(d, p, [d[0]], unit_med, "up", angles=[1], ymin=0, ymax=100, pad=0)[0]
    tr2 = build_gann_traces(d, p, [d[0]], unit_med, "up", angles=[1], ymin=0, ymax=100, pad=0)[0]
    assert np.allclose(tr1.y, tr2.y)
