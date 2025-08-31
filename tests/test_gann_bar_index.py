import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.chart.fans import build_gann_traces


def test_gann_line_uses_bar_index():
    # Two dates separated by a weekend; gap should count as 1 bar, not 3 days
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])  # Fri, Mon, Tue
    prices = np.zeros(len(dates), dtype=float)
    traces = build_gann_traces(
        dates, prices, bases=[dates[0]], unit=1.0, dir_mode="up",
        angles=[1], ymin=-10, ymax=10, pad=0.0
    )
    y = traces[0].y
    assert list(y[:3]) == [0.0, 1.0, 2.0]
