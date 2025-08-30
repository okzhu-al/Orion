import base64
import io
import json
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, exceptions, clientside_callback, ClientsideFunction
from dash_extensions import EventListener
import plotly.graph_objects as go

from app.data.loader import load_price_volume, _extract_price_volume
from app.data.timeframe import resample_series_by_tf, nearest_trading_day_in
from app.models.gann import ANGLES, DEC_PLACES, SCALE, median_abs_close_delta, compute_unit
from app.chart.figure import build_figure
from version import __version__

EXCEL_PATH = "data/399006.xlsx"
SHEET_NAME = 0

GRAPH_EVENTS = [
    {"event": "plotly_mousemove", "props": ["event"]},
    {"event": "plotly_unhover", "props": ["event"]},
]

close_s, vol_s = load_price_volume(EXCEL_PATH, SHEET_NAME)
ALL_DATES = close_s.index
ALL_PRICES = close_s.values.astype(float)
ALL_VOLUMES = vol_s.values.astype(float)

DEFAULT_UNIT_RAW = median_abs_close_delta(ALL_PRICES)
DEFAULT_UNIT = max(1.0 / SCALE, round(DEFAULT_UNIT_RAW, DEC_PLACES))


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
    if not contents or "," not in contents:
        raise ValueError("bad upload")
    header, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    if filename.lower().endswith(".xlsx") or "spreadsheetml" in header:
        s, v = load_price_volume(buf, SHEET_NAME)
    else:
        df = pd.read_csv(buf)
        s, v = _extract_price_volume(df)
    dates = s.index
    prices = s.values.astype(float)
    du_raw = median_abs_close_delta(prices)
    du_disp = max(1.0 / SCALE, round(du_raw, DEC_PLACES))
    return s, v, du_raw, du_disp

app = dash.Dash(__name__)
app.title = f"Orion v{__version__}"

app.layout = html.Div(
    style={"fontFamily":"-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial",
           "padding":"16px", "maxWidth":"1200px", "margin":"0 auto"},
    children=[
        html.H2(f"Orion v{__version__}", style={"margin":"0 0 8px 0"}),
        html.Div(id="source-file", style={"color":"#555", "marginBottom":"10px"}),
        dcc.Upload(
            id="uploader",
            children=html.Div(["拖拽或", html.A("选择CSV/XLSX文件")]),
            multiple=False,
            style={"width":"100%","height":"46px","lineHeight":"46px",
                   "borderWidth":"1px","borderStyle":"dashed","borderRadius":"6px",
                   "textAlign":"center","marginBottom":"16px","background":"#fafafa"}
        ),
        dcc.Store(id="series-store", data=DEFAULT_SERIES),

        html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr 1fr", "gap":"16px"}, children=[
            html.Div([
                html.Label("时间范围"),
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
                html.Label("1×1 单位（每根K线价格变化）"),
                html.Div([
                    dcc.Slider(
                        id="unit-slider",
                        min=DEFAULT_UNIT*0.2,
                        max=DEFAULT_UNIT*5,
                        value=DEFAULT_UNIT,
                        step=1.0/SCALE,
                        marks={DEFAULT_UNIT: "median"},
                        included=False,
                        tooltip={"placement":"bottom","always_visible":False},
                        updatemode="drag",
                        disabled=False,
                        dots=False
                    )
                ], style={"marginTop":"6px"}),
                html.Div(id="unit-text", style={"marginTop":"8px","fontSize":"12px","color":"#333"}),
                html.Div(style={"marginTop":"10px"}, children=[
                    html.Label("单位来源"),
                    dcc.RadioItems(
                        id="unit-source",
                        options=[
                            {"label":"中位数", "value":"median"},
                            {"label":"ATR × k", "value":"atr"},
                            {"label":"手动", "value":"manual"},
                        ],
                        value="median",
                        labelStyle={"display":"inline-block","marginRight":"12px"},
                        inputStyle={"marginRight":"4px"}
                    ),
                    html.Div([
                        html.Label("ATR 的 k 值"),
                        dcc.Slider(
                            id="k-slider",
                            min=0.2, max=3.0, step=0.1, value=1.0,
                            marks={0.2:"0.2", 1.0:"1.0", 3.0:"3.0"},
                            tooltip={"placement":"bottom","always_visible":False},
                        )
                    ], style={"marginTop":"6px"})
                ])
            ]),
            html.Div([
                html.Label("时间框架"),
                dcc.Dropdown(
                    id="tf",
                    options=[
                        {"label": "日线", "value": "D"},
                        {"label": "周线", "value": "W"},
                        {"label": "月线", "value": "M"},
                    ],
                    value="D",
                    clearable=False
                ),
                html.Hr(style={"margin":"10px 0"}),
                html.Label("扇形方向"),
                dcc.Dropdown(
                    id="fan-dir",
                    options=[
                        {"label":"自动（基点推断）", "value":"auto"},
                        {"label":"向上（低点）", "value":"up"},
                        {"label":"向下（高点）", "value":"down"},
                    ],
                    value="auto",
                    clearable=False
                )
            ])
        ]),

        html.Hr(),

        html.Div(style={"display":"grid","gridTemplateColumns":"2fr auto auto auto","gap":"8px","alignItems":"end"}, children=[
            html.Div([
                html.Label("基点（逗号分隔）"),
                dcc.Input(id="base-input", type="text", placeholder="YYYY-MM-DD, YYYY-MM-DD", style={"width":"100%"})
            ]),
            html.Button("应用基点", id="apply-bases", n_clicks=0),
            html.Button("清除基点", id="clear-bases", n_clicks=0, style={"background":"#eee"}),
            html.Div("提示：点击图表可添加最近交易日作为基点", style={"fontSize":"12px","color":"#666","marginBottom":"4px"})
        ]),
        html.Div(id="bases-list", style={"marginTop":"6px","fontSize":"13px","color":"#333"}),
        dcc.Store(id="bases-store", data=[]),

        html.Hr(),
        EventListener(
            id="graph-events",
            events=GRAPH_EVENTS,
            children=dcc.Graph(
                id="price-graph",
                config={"displaylogo": False, "modeBarButtonsToAdd": ["drawopenpath","eraseshape"]},
                style={"height":"620px"}
            ),
        ),
        dcc.Store(id="xrange-store", data=None),
        dcc.Store(id="cursor-y-store", data=None),
    ]
)


def str_to_dates_list(s: str):
    out = []
    for part in (s or "").split(","):
        t = part.strip()
        if not t:
            continue
        try:
            out.append(pd.Timestamp(t))
        except Exception:
            pass
    return out


clientside_callback(
    ClientsideFunction(namespace="orion", function_name="cursorYtoData"),
    Output("cursor-y-store", "data"),
    Input("graph-events", "event"),
    State("price-graph", "figure"),
    prevent_initial_call=True,
)

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
    du_raw = float(data.get("default_unit_raw", DEFAULT_UNIT_RAW))
    default_unit_local = float(data.get("default_unit", DEFAULT_UNIT))
    default_unit_local = round(default_unit_local, DEC_PLACES)
    unit_min = max(1.0/SCALE, round(default_unit_local*0.2, DEC_PLACES))
    unit_max = round(default_unit_local*5.0, DEC_PLACES)
    marks = {unit_min:f"{unit_min:.4f}", default_unit_local:"median", unit_max:f"{unit_max:.4f}"}
    filename = data.get("filename","(上传)")
    return dates[0].date(), dates[-1].date(), dates[0].date(), dates[-1].date(), unit_min, unit_max, default_unit_local, marks, filename


@app.callback(
    Output("unit-slider","value", allow_duplicate=True),
    Output("unit-slider","disabled"),
    Input("unit-source","value"),
    Input("k-slider","value"),
    Input("series-store","data"),
    Input("tf", "value"),
    prevent_initial_call=True
)
def _sync_unit(source, k, data, tf):
    try:
        dates = [pd.Timestamp(x) for x in data["dates"]]
        prices = np.array(data["prices"], dtype=float)
    except Exception:
        return no_update, True
    res = resample_series_by_tf(dates, prices, tf)
    dates_tf, prices_tf = res
    if source == "manual":
        return no_update, False
    if source == "median":
        unit = median_abs_close_delta(prices_tf)
    else:
        diffs = pd.Series(prices_tf).diff().abs()
        atr = diffs.rolling(14).mean().iloc[-1]
        unit = float(k) * (float(atr) if np.isfinite(atr) else 1.0/SCALE)
    unit = max(1.0/SCALE, round(float(unit), DEC_PLACES))
    return unit, True


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
        src_txt = "中位数"
    elif source == "atr":
        src_txt = f"ATR×k (k={float(k):.1f})"
    else:
        src_txt = "手动"
    return f"1×1 = {float(unit):.4f} | 基准 = {default_unit_local:.4f} | 来源 = {src_txt}"


@app.callback(
    Output("bases-store", "data"),
    Output("bases-list", "children"),
    Input("apply-bases", "n_clicks"),
    State("base-input", "value"),
    prevent_initial_call=True
)
def _apply_bases(n, text):
    dates_in = str_to_dates_list(text or "")
    bases = [str(pd.Timestamp(dt).date()) for dt in dates_in]
    bases = sorted(set(bases))
    label = "基点: " + ("，".join(bases) if bases else "无")
    return bases, label


@app.callback(
    Output("bases-store", "data", allow_duplicate=True),
    Output("bases-list", "children", allow_duplicate=True),
    Input("clear-bases", "n_clicks"),
    prevent_initial_call=True
)
def _clear_bases(n):
    return [], "基点: 无"


@app.callback(
    Output("range-hint", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("tf", "value")
)
def _range_hint(s, e, tf):
    return f"范围: {s} → {e} | TF = {tf}"


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
    all_dates_daily = np.array([pd.Timestamp(x) for x in data["dates"]])
    all_prices_daily = np.array(data["prices"], dtype=float)
    all_volumes_daily = np.array(data.get("volumes", [0]*len(all_dates_daily)), dtype=float)
    resampled = resample_series_by_tf(all_dates_daily, all_prices_daily, tf, all_volumes_daily)
    if len(resampled) == 3:
        all_dates, all_prices, all_volumes = resampled
    else:
        all_dates, all_prices = resampled
        all_volumes = np.zeros_like(all_prices)
    return build_figure(start_date, end_date, float(unit), bases, fan_dir,
                        all_dates, all_prices, all_volumes,
                        default_unit_local=float(data.get("default_unit", DEFAULT_UNIT)))


@app.callback(
    Output("series-store", "data", allow_duplicate=True),
    Input("uploader", "contents"),
    State("uploader", "filename"),
    prevent_initial_call=True
)
def _on_upload(contents, filename):
    try:
        s, v, du_raw, du_disp = parse_upload(contents, filename)
    except Exception:
        raise exceptions.PreventUpdate
    data = {
        "dates": [pd.Timestamp(d).isoformat() for d in s.index],
        "prices": [float(x) for x in s.values],
        "volumes": [float(x) for x in v.reindex(s.index).values],
        "filename": filename,
        "default_unit_raw": float(du_raw),
        "default_unit": float(du_disp)
    }
    return data


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
        raise exceptions.PreventUpdate
    x_val = clickData["points"][0].get("x")
    if x_val is None:
        raise exceptions.PreventUpdate
    dt_clicked = pd.Timestamp(x_val)
    all_dates_daily = [pd.Timestamp(x) for x in data["dates"]]
    all_prices_daily = np.array(data["prices"], dtype=float)
    all_dates_tf, _ = resample_series_by_tf(all_dates_daily, all_prices_daily, tf)
    start = pd.Timestamp(start_date); end = pd.Timestamp(end_date)
    mask = (all_dates_tf >= start) & (all_dates_tf <= end)
    dates = all_dates_tf[mask]
    if len(dates) == 0:
        raise exceptions.PreventUpdate
    di = pd.DatetimeIndex(dates)
    bar_index = di.get_indexer([dt_clicked], method="nearest")[0]
    dt = dates[bar_index]
    bases = set(bases or [])
    bases.add(str(pd.Timestamp(dt).date()))
    bases = sorted(bases)
    label = "基点: " + ("，".join(bases) if bases else "无")
    return bases, label


@app.callback(
    Output("price-graph", "figure", allow_duplicate=True),
    Input("cursor-y-store", "data"),
    State("price-graph", "figure"),
    prevent_initial_call=True,
)
def _update_price_label(dataY, fig_json):
    fig = go.Figure(fig_json)

    anns = list(fig.layout.annotations) if fig.layout.annotations else []
    anns = [a for a in anns if getattr(a, "name", None) != "__y_price__"]

    if dataY is not None:
        anns.append(dict(
            x=1.0, xref="paper", xanchor="left",
            y=float(dataY), yref="y", yanchor="middle",
            text=f"{float(dataY):.4f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#aaa", borderwidth=1,
            font=dict(size=12),
            name="__y_price__",
        ))

    fig.update_layout(annotations=anns)
    m = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    if m.get("r", 40) < 80:
        fig.update_layout(margin=dict(l=m.get("l", 20), r=80, t=m.get("t", 60), b=m.get("b", 40)))

    return fig
