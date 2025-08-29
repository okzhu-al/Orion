import numpy as np
import pandas as pd
from app.chart.figure import build_figure


def test_price_axis_spike_label():
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    prices = pd.Series(np.linspace(1, 10, 10), index=dates)
    volumes = pd.Series(range(10), index=dates)
    fig = build_figure(dates[0], dates[-1], unit=1.0, base_dates=[],
                       fan_dir=1, all_dates=dates, all_prices=prices,
                       all_volumes=volumes)
    assert fig.layout.yaxis.spikemode == 'across+toaxis'
    assert fig.layout.yaxis2.spikemode == 'across+toaxis'
