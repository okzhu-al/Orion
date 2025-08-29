# gann_dash_app.py
# Dash web app: Gann Angles with multi base points, date-range control, unit slider.
# Y-axis fits selected price range only. UI in English.

import json
import numpy as np
import pandas as pd
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go

import base64, io

# ====== CONFIG ======
EXCEL_PATH = "data/399006.xlsx"  # same folder as this script
SHEET_NAME = 0
ANGLES = [1/8, 1/4, 1/3, 1/2, 1, 2, 3, 4, 8]  # relative to 1x1
APP_TITLE = "Gann Angles – Interactive"
# ====================

# ---- load price & volume series (robust to cn/en column names) ----
def _find_col(cands, cols):
    lower = {str(c).lower(): c for c in cols}
    for k in cands:
        if k in cols:
            return k
        lk = str(k).lower()
        if lk in lower:
            return lower[lk]
    for c in cols:
        for k in cands:
            if str(k).lower() in str(c).lower():
                return c
    raise ValueError(f"Cannot find any of {cands} in columns: {list(cols)}")


def _extract_price_volume(df):
    dcol = _find_col(["date", "Date", "日期", "交易日期", "时间", "time", "Time"], df.columns)
    ccol = _find_col(["close", "Close", "收盘", "收盘价", "收盘价格", "收盘价(元)"], df.columns)
    try:
        vcol = _find_col(["volume", "Volume", "成交量", "VOL", "Vol"], df.columns)
        cols = [dcol, ccol, vcol]
        df = df[cols].rename(columns={dcol: "date", ccol: "close", vcol: "volume"})
    except ValueError:
        df = df[[dcol, ccol]].rename(columns={dcol: "date", ccol: "close"})
        df["volume"] = 0.0
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    close_s = df.set_index("date")["close"]
    vol_s = df.set_index("date")["volume"].fillna(0.0)
    return close_s, vol_s


def load_price_volume(path, sheet=0):
    df = pd.read_excel(path, sheet_name=sheet)
    return _extract_price_volume(df)


close, volume = load_price_volume(EXCEL_PATH, SHEET_NAME)
ALL_DATES = close.index
ALL_PRICES = close.values.astype(float)
ALL_VOLUMES = volume.values.astype(float)
N = len(close)

def nearest_trading_day(ts: pd.Timestamp) -> pd.Timestamp:
    if ts in close.index:
        return ts
    idx = close.index.get_indexer([ts], method="nearest")[0]
    return close.index[idx]

def median_abs_close_delta(prices: np.ndarray) -> float:
    m = pd.Series(prices).diff().abs().median()
    if not np.isfinite(m) or m == 0:
        m = max(1.0, float(np.std(prices)/100.0))
    return float(m)

def resample_series_by_tf(dates_list, prices_list, tf: str, volumes_list=None):
    """Return resampled arrays by timeframe.

    D: daily (no change)
    W: weekly (last close of week, Friday anchored)
    M: monthly (last close of month)

    If the latest bar does not align with the natural period end (e.g. the
    current week or month has not finished), the returned series will use the
    last available trading day as the endpoint instead of the calendar period
    end so that the most recent partial period is still plotted.
    """

    s = pd.Series(np.asarray(prices_list, dtype=float), index=pd.DatetimeIndex(dates_list))
    v = None
    if volumes_list is not None:
        v = pd.Series(np.asarray(volumes_list, dtype=float), index=pd.DatetimeIndex(dates_list))

    if tf == "D":
        sr = s.dropna()
        if v is not None:
            vr = v.reindex(sr.index).fillna(0.0)
    else:
        freq = "W-FRI" if tf == "W" else "ME"
        sr = s.resample(freq).last()
        if v is not None:
            vr = v.resample(freq).sum()
            df = pd.concat([sr, vr], axis=1).dropna()
            sr = df.iloc[:, 0]
            vr = df.iloc[:, 1]
        else:
            sr = sr.dropna()

    dates = sr.index.to_numpy()
    if tf in ("W", "M") and len(dates) > 0:
        last_orig = pd.DatetimeIndex(dates_list)[-1]
        if dates[-1] > last_orig.to_datetime64():
            dates[-1] = last_orig.to_datetime64()

    if volumes_list is not None:
        return dates, sr.values.astype(float), vr.values.astype(float)
    return dates, sr.values.astype(float)

# ---- unit precision (4 decimals) ----
DEC_PLACES = 4
SCALE = 10 ** DEC_PLACES  # 10000 for 4 decimals

DEFAULT_UNIT_RAW = median_abs_close_delta(ALL_PRICES)
# display unit rounded to 4 decimals; clamp to minimum step
DEFAULT_UNIT = max(1.0 / SCALE, round(DEFAULT_UNIT_RAW, DEC_PLACES))

def _round_eps(x, mode="round"):
    if mode == "floor":
        return np.floor(x * SCALE) / SCALE
    if mode == "ceil":
        return np.ceil(x * SCALE) / SCALE
    return np.round(x * SCALE) / SCALE

UNIT_MIN  = max(1.0 / SCALE, _round_eps(DEFAULT_UNIT * 0.2, "floor"))
UNIT_MAX  = _round_eps(DEFAULT_UNIT * 5.0, "ceil")
UNIT_STEP = 1.0 / SCALE
SLIDER_MARKS = {
    UNIT_MIN: f"{UNIT_MIN:.4f}",
    DEFAULT_UNIT: "median",
    UNIT_MAX: f"{UNIT_MAX:.4f}"
}

# ---------- pack/unpack helpers for dcc.Store ----------
# def pack_series(dates_index, prices_array, filename):
#     return {
#         "dates": [pd.Timestamp(d).isoformat() for d in dates_index],
#         "prices": [float(p) for p in prices_array],
#         "filename": filename,
#         "default_unit": float(DEFAULT_UNIT)
#     }
def pack_series(dates_index, prices_array, volumes_array, filename):
    du_raw = median_abs_close_delta(np.asarray(prices_array, dtype=float))
    du_disp = max(1.0 / SCALE, round(du_raw, DEC_PLACES))
    return {
        "dates": [pd.Timestamp(d).isoformat() for d in dates_index],
        "prices": [float(p) for p in prices_array],
        "volumes": [float(v) for v in volumes_array],
        "filename": filename,
        "default_unit_raw": float(du_raw),
        "default_unit": float(du_disp)
    }

DEFAULT_SERIES = pack_series(ALL_DATES, ALL_PRICES, ALL_VOLUMES, EXCEL_PATH)

def parse_upload(contents, filename):
    """
    contents: "data:...;base64,<b64>"
    returns pandas Series (date-indexed close) and computed default_unit (rounded to 4 decimals)
    """
    if not contents or "," not in contents:
        raise ValueError("Invalid upload contents")
    header, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    if filename.lower().endswith(".xlsx") or "spreadsheetml" in header:
        s, v = load_price_volume(buf, SHEET_NAME)
    else:
        # try CSV
        df = pd.read_csv(buf)
        s, v = _extract_price_volume(df)
    dates = s.index
    prices = s.values.astype(float)
    du_raw = median_abs_close_delta(prices)
    du_disp = max(1.0 / SCALE, round(du_raw, DEC_PLACES))
    return s, v, du_raw, du_disp

# ===== Dash App =====
app = dash.Dash(__name__)
app.title = APP_TITLE

app.layout = html.Div(
    style={"fontFamily":"-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial",
           "padding":"16px", "maxWidth":"1200px", "margin":"0 auto"},
    children=[
        html.H2(APP_TITLE, style={"margin":"0 0 8px 0"}),
        html.Div(id="source-file", style={"color":"#555", "marginBottom":"10px"}),
        dcc.Upload(
            id="uploader",
            children=html.Div(["Drag & drop or ", html.A("select a CSV/XLSX file")]),
            multiple=False,
            style={
                "width":"100%","height":"46px","lineHeight":"46px",
                "borderWidth":"1px","borderStyle":"dashed","borderRadius":"6px",
                "textAlign":"center","marginBottom":"16px","background":"#fafafa"
            }
        ),
        dcc.Store(id="series-store", data=DEFAULT_SERIES),

        # Controls row
        html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr 1fr", "gap":"16px"}, children=[
            html.Div([
                html.Label("X-axis date range"),
                dcc.DatePickerRange(
                    id="date-range",
                    min_date_allowed=ALL_DATES.min().to_pydatetime(),
                    max_date_allowed=ALL_DATES.max().to_pydatetime(),
                    start_date=ALL_DATES.min().date(),
                    end_date=ALL_DATES.max().date(),
                    display_format="YYYY-MM-DD",
                    clearable=False
                ),
                html.Div(id="range-hint", style={"fontSize":"12px","color":"#666","marginTop":"6px"})
            ]),
            html.Div([
                html.Label("1×1 unit (price per bar)"),
                html.Div([
                    dcc.Slider(
                        id="unit-slider",
                        min=UNIT_MIN,
                        max=UNIT_MAX,
                        value=DEFAULT_UNIT,
                        step=UNIT_STEP,
                        marks=SLIDER_MARKS,       # initial marks to avoid empty render
                        included=False,           # remove dark included bar to reduce visual noise
                        tooltip={"placement":"bottom","always_visible":False},
                        updatemode="drag",        # update continuously while dragging
                        disabled=False,
                        dots=False
                    )
                ], style={"marginTop":"6px"}),
                html.Div(
                    id="unit-text",
                    style={"marginTop":"8px","fontSize":"12px","color":"#333"}
                ),
                html.Div(style={"marginTop":"10px"}, children=[
                    html.Label("Unit Source"),
                    dcc.RadioItems(
                        id="unit-source",
                        options=[
                            {"label":"Median", "value":"median"},
                            {"label":"ATR × k", "value":"atr"},
                            {"label":"Manual", "value":"manual"},
                        ],
                        value="median",
                        labelStyle={"display":"inline-block","marginRight":"12px"},
                        inputStyle={"marginRight":"4px"}
                    ),
                    html.Div([
                        html.Label("k (for ATR mode)"),
                        dcc.Slider(
                            id="k-slider",
                            min=0.2, max=3.0, step=0.1, value=1.0,
                            marks={0.2:"0.2", 1.0:"1.0", 3.0:"3.0"},
                            tooltip={"placement":"bottom","always_visible":False}
                        )
                    ], style={"marginTop":"6px"})
                ])
            ]),
            html.Div([
                html.Label("Timeframe"),
                dcc.Dropdown(
                    id="tf",
                    options=[
                        {"label": "Daily (D)", "value": "D"},
                        {"label": "Weekly (W)", "value": "W"},
                        {"label": "Monthly (M)", "value": "M"},
                    ],
                    value="D",
                    clearable=False
                ),
                html.Div("Tip: Weekly = last close of week; Monthly = last close of month.",
                         style={"fontSize":"12px","color":"#666","marginTop":"6px"}),

                html.Hr(style={"margin":"10px 0"}),

                html.Label("Fan direction"),
                dcc.Dropdown(
                    id="fan-dir",
                    options=[
                        {"label":"Auto (infer from base)", "value":"auto"},
                        {"label":"Up (from low)", "value":"up"},
                        {"label":"Down (from high)", "value":"down"},
                    ],
                    value="auto",
                    clearable=False
                ),
                html.Div("Tip: Auto = local high → Down, else Up",
                        style={"fontSize":"12px","color":"#666","marginTop":"6px"})
            ])
        ]),

        html.Hr(),

        # Bases: input + buttons
        html.Div(style={"display":"grid","gridTemplateColumns":"2fr auto auto auto","gap":"8px","alignItems":"end"}, children=[
            html.Div([
                html.Label("Base points (comma-separated, e.g. 2005-06-06, 2008-10-28)"),
                dcc.Input(id="base-input", type="text", placeholder="YYYY-MM-DD, YYYY-MM-DD, ...",
                          style={"width":"100%"})
            ]),
            html.Button("Apply bases", id="apply-bases", n_clicks=0),
            html.Button("Clear bases", id="clear-bases", n_clicks=0, style={"background":"#eee"}),
            html.Div("Tip: click the chart to add nearest trading day as a base.",
                     style={"fontSize":"12px","color":"#666","marginBottom":"4px"})
        ]),
        html.Div(id="bases-list", style={"marginTop":"6px","fontSize":"13px","color":"#333"}),

        # Hidden store for current base dates (ISO strings)
        dcc.Store(id="bases-store", data=[]),

        html.Hr(),

        dcc.Graph(
            id="price-graph",
            config={"displaylogo": False, "modeBarButtonsToAdd": ["drawopenpath","eraseshape"]},
            style={"height":"620px"}
        ),
        # Hidden store to remember last graph x-range (so panning/zoom keeps)
        dcc.Store(id="xrange-store", data=None),
    ]
)

# -------- utilities --------

def str_to_dates_list(s: str):
    out = []
    for part in (s or "").split(","):
        t = part.strip()
        if not t: continue
        try:
            dt = pd.Timestamp(t)
            out.append(dt)
        except Exception:
            pass
    return out

# Helper: find nearest trading day in a given date index
def nearest_trading_day_in(dates_index, ts: pd.Timestamp) -> pd.Timestamp:
    """Return the nearest date in dates_index to ts."""
    ts = pd.Timestamp(ts)
    if isinstance(dates_index, (pd.DatetimeIndex, pd.Index)):
        di = pd.DatetimeIndex(dates_index)
        if ts in di:
            return ts
        idx = di.get_indexer([ts], method="nearest")[0]
        return di[idx]
    # fallback if array-like
    arr = pd.to_datetime(list(dates_index))
    di = pd.DatetimeIndex(arr)
    if ts in di:
        return ts
    idx = di.get_indexer([ts], method="nearest")[0]
    return di[idx]

# Helper: infer direction (+1 up, -1 down) for a base point
def infer_direction_around(prices: np.ndarray, idx: int, window: int = 5) -> int:
    """
    Return +1 for Up, -1 for Down.
    If the base is a local high in +/- window, choose Down; otherwise Up.
    """
    i0 = max(0, idx - window)
    i1 = min(len(prices), idx + window + 1)
    segment = prices[i0:i1]
    if len(segment) == 0:
        return +1
    # If base price is near the max of the segment, treat as high → down
    if prices[idx] >= np.max(segment):
        return -1
    # If base price is near the min of the segment, treat as low → up
    if prices[idx] <= np.min(segment):
        return +1
    # fallback: compare immediate neighbors
    left = prices[idx-1] if idx-1 >= 0 else prices[idx]
    right = prices[idx+1] if idx+1 < len(prices) else prices[idx]
    return -1 if prices[idx] > left and prices[idx] > right else +1

def build_figure(start_date, end_date, unit, base_dates, fan_dir="auto", all_dates=None, all_prices=None, all_volumes=None, default_unit_local=None):
    if all_dates is None:
        all_dates = ALL_DATES
    if all_prices is None:
        all_prices = ALL_PRICES
    if all_volumes is None:
        all_volumes = ALL_VOLUMES
    # enforce 4-decimal resolution for unit
    unit = round(float(unit), DEC_PLACES)
    # constrain to selected X range
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    mask = (all_dates >= start) & (all_dates <= end)
    dates = all_dates[mask]
    prices = all_prices[mask]
    volumes = all_volumes[mask]
    n = len(dates)
    if n == 0:
        # empty range guard
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Date", yaxis_title="Price",
            title="No data in selected range",
        )
        return fig

    fig = go.Figure()

    # price line
    vol_ma = pd.Series(volumes).rolling(20).mean().fillna(0).to_numpy()
    customdata = np.stack([volumes, vol_ma], axis=-1)
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode="lines",
        name="Close",
        line=dict(width=1.5),
        customdata=customdata,
        hovertemplate="%{x|%Y-%m-%d}<br>Close=%{y:.4f}<br>Volume=%{customdata[0]:,.0f}<br>Avg20=%{customdata[1]:,.0f}<extra></extra>",
    ))
    # transparent marker layer to reliably capture clicks at each bar
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode="markers",
        marker=dict(size=12, color="rgba(0,0,0,0)", opacity=0.01),  # easier to click, nearly invisible
        hoverinfo="skip",
        name="_click_capture",
        showlegend=False
    ))

    # Y-axis autoscale strictly by price data in range (not fans)
    ymin, ymax = float(np.min(prices)), float(np.max(prices))
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    # Will update_yaxes after fans are added, but use ymin/ymax/pad for clipping

    # add fans for each base
    base_scatter_x, base_scatter_y, base_scatter_date = [], [], []
    for b in base_dates:
        # snap base to the nearest trading day **within the currently selected range**
        b = nearest_trading_day_in(dates, pd.Timestamp(b))
        # base positions (relative to selected range)
        b_sel_idx = np.where(dates == b)[0]
        if len(b_sel_idx)==0:
            continue
        b_sel_idx = int(b_sel_idx[0])
        b_px = prices[b_sel_idx]
        full_range = pd.date_range(dates[b_sel_idx], dates[-1], freq="D")
        x_rel = (full_range - dates[b_sel_idx]) / np.timedelta64(1, "D")

        # choose direction sign: +1 up, -1 down
        if fan_dir == "up":
            dir_sign = +1
        elif fan_dir == "down":
            dir_sign = -1
        else:
            dir_sign = infer_direction_around(prices, b_sel_idx, window=5)

        for a in ANGLES:
            slope = dir_sign * a * unit
            y = b_px + slope * x_rel
            # clip to current y-axis price range so lines don't vanish after autoscale
            y_low  = ymin - pad
            y_high = ymax + pad
            mask_vis = (y >= y_low) & (y <= y_high)
            if not np.any(mask_vis):
                # if completely outside, still draw the first point to indicate origin
                mask_vis = np.zeros_like(y, dtype=bool)
                mask_vis[0] = True
            x_plot = full_range[mask_vis]
            y_plot = y[mask_vis]

            line_style = dict(width=1)
            if a != 1:
                line_style["dash"] = "dash"
            fig.add_trace(go.Scatter(
                x=x_plot,
                y=y_plot,
                mode="lines",
                name=f"Gann {a if a>=1 else 1:.0f}x{1 if a>=1 else int(round(1/a))} (base {b.date()})",
                line=line_style,
                hoverinfo="skip",
            ))

        base_scatter_x.append(dates[b_sel_idx])
        base_scatter_y.append(b_px)
        base_scatter_date.append(pd.Timestamp(b).strftime("%Y-%m-%d"))

    if base_scatter_x:
        fig.add_trace(go.Scatter(
            x=base_scatter_x,
            y=base_scatter_y,
            mode="markers",
            name="Base points",
            marker=dict(size=8, color="#d62728"),
            customdata=np.array(base_scatter_date, dtype=object),
            hovertemplate="%{customdata}<br>Base=%{y:.4f}<extra></extra>",
        ))

    # X-axis (dates)
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="data",
        spikecolor="#aaa", spikethickness=1,
        tickformat="%Y-%m-%d", hoverformat="%Y-%m-%d", tickangle=20
    )

    fig.update_yaxes(
        range=[ymin - pad, ymax + pad], showspikes=True, spikemode="across",
        spikesnap="cursor", spikecolor="#aaa", spikethickness=1
    )

    du = DEFAULT_UNIT if default_unit_local is None else default_unit_local

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Price",
        title=f"1×1 unit = {unit:.4f} (median(|Δclose|) = {du:.4f})",
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

# -------- callbacks --------

@app.callback(
    Output("date-range", "min_date_allowed"),
    Output("date-range", "max_date_allowed"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Output("unit-slider", "min"),
    Output("unit-slider", "max"),
    Output("unit-slider", "value"),
    Output("unit-slider", "marks"),
    Output("source-file", "children"),
    Input("series-store", "data"),
)
def _init_from_series(data):
    dates = [pd.Timestamp(x) for x in data["dates"]]
    prices = np.array(data["prices"], dtype=float)
    filename = data.get("filename","(uploaded)")
    # raw/display defaults
    du_raw = float(data.get("default_unit_raw", DEFAULT_UNIT_RAW))
    if not np.isfinite(du_raw) or du_raw <= 0:
        du_raw = 1.0 / SCALE  # fallback for degenerate data
    default_unit_local = float(data.get("default_unit", DEFAULT_UNIT))
    if not np.isfinite(default_unit_local) or default_unit_local <= 0:
        default_unit_local = 1.0 / SCALE
    default_unit_local = round(default_unit_local, DEC_PLACES)

    # compute slider range from RAW value (quantized to 4-dec)
    unit_min  = max(1.0 / SCALE, np.floor(du_raw * 0.2 * SCALE) / SCALE)
    unit_max  = np.ceil(du_raw * 5.0 * SCALE) / SCALE

    # safety: ensure visible span (>= 50 steps) and valid ordering
    if not np.isfinite(unit_min): unit_min = 1.0 / SCALE
    if not np.isfinite(unit_max): unit_max = unit_min + 50.0 / SCALE
    if unit_max <= unit_min:
        unit_max = unit_min + 50.0 / SCALE

    # ensure plain Python floats (not numpy scalars)
    unit_min = float(unit_min)
    unit_max = float(unit_max)
    default_unit_local = float(default_unit_local)

    # --- ADDED GUARD: ensure center mark is strictly inside (min, max) to avoid edge recursion bugs
    eps = UNIT_STEP
    if default_unit_local <= unit_min:
        default_unit_local = unit_min + eps
    if default_unit_local >= unit_max:
        default_unit_local = unit_max - eps
    default_unit_local = round(default_unit_local, DEC_PLACES)

    label = f"Data source: {filename}"

    # marks MUST use numeric keys for this dcc.Slider version, rounded to DEC_PLACES
    marks = {
        round(unit_min, DEC_PLACES): f"{unit_min:.4f}",
        round(default_unit_local, DEC_PLACES): "median",
        round(unit_max, DEC_PLACES): f"{unit_max:.4f}",
    }

    return (dates[0].to_pydatetime(), dates[-1].to_pydatetime(),
            dates[0].date(), dates[-1].date(),
            unit_min, unit_max, default_unit_local,
            marks, label)

# update unit text
# @app.callback(
#     Output("unit-text", "children"),
#     Input("unit-slider", "value"),
#     Input("series-store", "data"),
# )
# def _unit_text(unit, data):
#     default_unit_local = round(float(data.get("default_unit", DEFAULT_UNIT)), DEC_PLACES)
#     return f"1×1 = {float(unit):.4f}  |  median(|Δclose|) = {default_unit_local:.4f}"

@app.callback(
    Output("unit-text", "children"),
    Input("unit-slider", "value"),
    Input("series-store", "data"),
    Input("unit-source", "value"),
    Input("k-slider", "value"),
)
def _unit_text(unit, data, source, k):
    default_unit_local = round(float(data.get("default_unit", DEFAULT_UNIT)), DEC_PLACES)
    if source == "median":
        src_txt = "Median(|Δclose|)"
    elif source == "atr":
        src_txt = f"ATR(14) × k (k={float(k):.1f})"
    else:
        src_txt = "Manual"
    return f"1×1 = {float(unit):.4f}  |  baseline(median) = {default_unit_local:.4f}  |  source = {src_txt}"

@app.callback(
    Output("unit-slider","value", allow_duplicate=True),
    Output("unit-slider","disabled"),
    Input("unit-source","value"),
    Input("k-slider","value"),
    Input("series-store","data"),
    prevent_initial_call=True
)
def _sync_unit(source, k, data):
    try:
        prices = np.array(data["prices"], dtype=float)
    except Exception:
        # 数据异常：禁用滑块，保持现值
        return no_update, True

    # 仅收盘价时，用 |Δclose| 作为 TR 近似，取 n=14 的滑动均值作为类 ATR
    tr = np.abs(np.diff(prices))
    atr_n = 14
    atr = pd.Series(tr).rolling(atr_n, min_periods=1).mean().iloc[-1]
    atr = float(atr) if np.isfinite(atr) else (1.0 / SCALE)

    if source == "manual":
        # 不覆盖当前值；启用滑块
        return no_update, False

    if source == "median":
        unit = median_abs_close_delta(prices)
    else:  # "atr"
        try:
            unit = float(k) * atr
        except Exception:
            unit = atr

    # 与 4 位精度和最小步长对齐
    unit = max(1.0 / SCALE, round(float(unit), DEC_PLACES))
    return unit, True

# add base by clicking the chart (now also updates bases-list)
@app.callback(
    Output("bases-store", "data", allow_duplicate=True),
    Output("bases-list", "children", allow_duplicate=True),
    Input("price-graph", "clickData"),
    State("bases-store", "data"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    Input("tf", "value"),
    State("series-store", "data"),
    prevent_initial_call=True
)
def _click_add_base(clickData, bases, start_date, end_date, tf, data):
    if not clickData:
        raise dash.exceptions.PreventUpdate
    x_val = clickData["points"][0].get("x", None)
    if x_val is None:
        raise dash.exceptions.PreventUpdate
    try:
        dt_clicked = pd.Timestamp(x_val)
    except Exception:
        raise dash.exceptions.PreventUpdate
    all_dates_daily = [pd.Timestamp(x) for x in data["dates"]]
    all_prices_daily = np.array(data["prices"], dtype=float)
    all_dates_tf, _ = resample_series_by_tf(all_dates_daily, all_prices_daily, tf)
    start = pd.Timestamp(start_date); end = pd.Timestamp(end_date)
    mask = (all_dates_tf >= start) & (all_dates_tf <= end)
    dates = all_dates_tf[mask]
    if len(dates) == 0:
        raise dash.exceptions.PreventUpdate
    di = pd.DatetimeIndex(dates)
    bar_index = di.get_indexer([dt_clicked], method="nearest")[0]
    dt = dates[bar_index]
    bases = set(bases or [])
    bases.add(str(pd.Timestamp(dt).date()))
    bases = sorted(bases)
    label = "Active base points: " + (", ".join(bases) if bases else "(none)")
    return bases, label


# apply typed bases
@app.callback(
    Output("bases-store", "data"),
    Output("bases-list", "children"),
    Input("apply-bases", "n_clicks"),
    State("base-input", "value"),
    prevent_initial_call=True
)
def _apply_bases(n, text):
    dates_in = str_to_dates_list(text or "")
    bases = []
    for dt in dates_in:
        bases.append(str(pd.Timestamp(dt).date()))
    bases = sorted(set(bases))
    label = "Active base points: " + (", ".join(bases) if bases else "(none)")
    return bases, label

# clear bases
@app.callback(
    Output("bases-store", "data", allow_duplicate=True),
    Output("bases-list", "children", allow_duplicate=True),
    Input("clear-bases", "n_clicks"),
    prevent_initial_call=True
)
def _clear_bases(n):
    return [], "Active base points: (none)"

# show date-range hint
@app.callback(
    Output("range-hint", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("tf", "value")
)
def _range_hint(s, e, tf):
    return f"Selected range: {s} → {e}  |  TF = {tf}"

# main figure render
@app.callback(
    Output("price-graph", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("unit-slider", "value"),
    Input("bases-store", "data"),
    Input("fan-dir", "value"),
    Input("tf", "value"),
    Input("series-store", "data"),
)
def _update_graph(start_date, end_date, unit, bases, fan_dir, tf, data):
    bases = bases or []
    # unpack series
    all_dates_daily = np.array([pd.Timestamp(x) for x in data["dates"]])
    all_prices_daily = np.array(data["prices"], dtype=float)
    all_volumes_daily = np.array(data.get("volumes", [0]*len(all_dates_daily)), dtype=float)
    resampled = resample_series_by_tf(all_dates_daily, all_prices_daily, tf, all_volumes_daily)
    if len(resampled) == 3:
        all_dates, all_prices, all_volumes = resampled
    else:
        all_dates, all_prices = resampled
        all_volumes = np.zeros_like(all_prices)
    return build_figure(start_date, end_date, float(unit), bases,
                        fan_dir=fan_dir,
                        all_dates=all_dates,
                        all_prices=all_prices,
                        all_volumes=all_volumes,
                        default_unit_local=float(data.get("default_unit", DEFAULT_UNIT)))

@app.callback(
    Output("series-store", "data", allow_duplicate=True),
    Input("uploader", "contents"),
    State("uploader", "filename"),
    prevent_initial_call=True
)
# def _on_upload(contents, filename):
#     try:
#         s, du = parse_upload(contents, filename)
#     except Exception as e:
#         raise dash.exceptions.PreventUpdate
#     data = {
#         "dates": [pd.Timestamp(d).isoformat() for d in s.index],
#         "prices": [float(x) for x in s.values],
#         "filename": filename,
#         "default_unit": float(du)
#     }
#     return data
def _on_upload(contents, filename):
    try:
        s, v, du_raw, du_disp = parse_upload(contents, filename)
    except Exception:
        raise dash.exceptions.PreventUpdate
    data = {
        "dates": [pd.Timestamp(d).isoformat() for d in s.index],
        "prices": [float(x) for x in s.values],
        "volumes": [float(x) for x in v.reindex(s.index).values],
        "filename": filename,
        "default_unit_raw": float(du_raw),
        "default_unit": float(du_disp)
    }
    return data

if __name__ == "__main__":
    app.run(debug=True)
