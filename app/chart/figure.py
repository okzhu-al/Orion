import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Iterable, List

from app.chart.fans import build_gann_traces
from app.data.timeframe import nearest_trading_day_in
from app.models.gann import ANGLES, DEC_PLACES, SCALE, compute_unit


def build_figure(start_date, end_date, unit, base_dates, fan_dir,
                 all_dates, all_prices, all_volumes, default_unit_local=None) -> go.Figure:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (all_dates >= start) & (all_dates <= end)
    dates = all_dates[mask]
    prices = all_prices[mask]
    volumes = all_volumes[mask]
    if len(dates) == 0:
        return go.Figure()

    fig = go.Figure()
    vol_ma = pd.Series(volumes).rolling(20).mean().fillna(0).to_numpy()
    customdata = np.stack([volumes, vol_ma], axis=-1)
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode="lines",
        name="收盘",
        line=dict(width=1.5),
        customdata=customdata,
        hovertemplate="%{x|%Y-%m-%d}<br>收盘=%{y:.4f}<br>成交量=%{customdata[0]:,.0f}<br>均量20=%{customdata[1]:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode="markers",
        marker=dict(size=12, color="rgba(0,0,0,0)", opacity=0.01),
        hoverinfo="skip",
        name="_click_capture",
        showlegend=False
    ))

    ymin, ymax = float(np.min(prices)), float(np.max(prices))
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)

    snap_bases = []
    base_x, base_y, base_dates_str = [], [], []
    for b in base_dates:
        b = nearest_trading_day_in(dates, pd.Timestamp(b))
        idx = np.where(dates == b)[0]
        if len(idx) == 0:
            continue
        idx = int(idx[0])
        snap_bases.append(b)
        base_x.append(dates[idx])
        base_y.append(prices[idx])
        base_dates_str.append(pd.Timestamp(b).strftime("%Y-%m-%d"))

    for tr in build_gann_traces(dates, prices, snap_bases, unit, fan_dir,
                                angles=ANGLES, ymin=ymin, ymax=ymax, pad=pad):
        fig.add_trace(tr)

    if base_x:
        fig.add_trace(go.Scatter(
            x=base_x,
            y=base_y,
            mode="markers",
            name="基点",
            marker=dict(size=8, color="#d62728"),
            customdata=np.array(base_dates_str, dtype=object),
            hovertemplate="%{customdata}<br>Base=%{y:.4f}<extra></extra>",
        ))

    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="data",
        spikecolor="#aaa", spikethickness=1,
        tickformat="%Y-%m-%d", hoverformat="%Y-%m-%d"
    )
    fig.update_yaxes(
        range=[ymin - pad, ymax + pad], showspikes=True, spikemode="across",
        spikesnap="cursor", spikecolor="#aaa", spikethickness=1
    )
    du = default_unit_local if default_unit_local is not None else unit
    fig.update_layout(
        template="plotly_white",
        xaxis_title="日期",
        yaxis_title="价格",
        title=f"1×1 单位 = {unit:.4f} (中位数 = {du:.4f})",
        showlegend=False,
        margin=dict(l=20, r=40, t=60, b=40),
        clickmode="event+select",
        hovermode="x",
        spikedistance=-1,
        hoverdistance=50,
        xaxis2=dict(
            matches="x", overlaying="x", side="top",
            showgrid=False, showline=False, zeroline=False,
            tickmode="array", tickvals=[], ticks="",
            showticklabels=True,
            showspikes=True, spikemode="across+toaxis", spikesnap="cursor",
            spikecolor="#aaa", spikethickness=1,
            tickformat="%Y-%m-%d", hoverformat="%Y-%m-%d"
        ),
        yaxis2=dict(
            matches="y", overlaying="y", side="right",
            showgrid=False, showline=False, zeroline=False,
            tickmode="array", tickvals=[], ticks="",
            showticklabels=True,
            showspikes=True, spikemode="across+toaxis", spikesnap="cursor",
            spikecolor="#aaa", spikethickness=1
        )
    )
    return fig
