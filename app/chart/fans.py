import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Iterable, List

from app.data.timeframe import nearest_trading_day_in
from app.models.gann import ANGLES, infer_direction_around


def build_gann_traces(dates: Iterable[pd.Timestamp], prices: Iterable[float],
                      bases: Iterable[pd.Timestamp], unit: float,
                      dir_mode: str, angles=ANGLES,
                      ymin: float = 0.0, ymax: float = 1.0, pad: float = 0.0) -> List[go.Scatter]:
    dates = pd.DatetimeIndex(dates)
    prices = np.asarray(prices, dtype=float)
    bars = np.arange(len(dates), dtype=float)
    traces = []
    for b in bases:
        b = nearest_trading_day_in(dates, pd.Timestamp(b))
        idx_arr = np.where(dates == b)[0]
        if len(idx_arr) == 0:
            continue
        b_idx = int(idx_arr[0])
        b_price = float(prices[b_idx])
        x_plot_full = bars[b_idx:]
        bars_rel = np.arange(len(x_plot_full), dtype=float)
        if dir_mode == "up":
            dir_sign = +1
        elif dir_mode == "down":
            dir_sign = -1
        else:
            dir_sign = infer_direction_around(prices, b_idx, window=5)
        for a in angles:
            slope_per_bar = dir_sign * float(a) * float(unit)
            y = b_price + slope_per_bar * bars_rel
            y_low, y_high = ymin - pad, ymax + pad
            mask = (y >= y_low) & (y <= y_high)
            if not np.any(mask):
                mask = np.zeros_like(y, dtype=bool)
                mask[0] = True
            trace = go.Scatter(
                x=x_plot_full[mask],
                y=y[mask],
                mode="lines",
                line=dict(width=1, dash=None if a == 1 else "dash"),
                hoverinfo="skip",
                name=f"Gann {a if a>=1 else 1:.0f}x{1 if a>=1 else int(round(1/a))} ({b.date()})"
            )
            traces.append(trace)
    return traces
