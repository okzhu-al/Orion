import numpy as np
import pandas as pd

from typing import Iterable, Optional

from .loader import _find_col


_COL_CANDIDATES = {
    "date": ["date", "日期", "交易日期", "time", "时间"],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价", "收盘价格", "收盘价(元)"],
    "volume": ["volume", "成交量", "vol", "VOL"],
}


def resample_ohlcv_daily_to_tf(df_daily: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample daily OHLCV data to a target timeframe.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily data containing at least a date column and close prices.
    tf : str
        Target timeframe: ``"D"``, ``"W"`` or ``"M"``.

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV with index at period end. Missing OHLC or volume
        columns are proxied from close or filled with zeros.
    """

    tf = tf.upper()
    if tf not in {"D", "W", "M"}:
        raise ValueError("tf must be one of 'D', 'W', or 'M'")

    if df_daily.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    cols = {}
    for key, cands in _COL_CANDIDATES.items():
        try:
            cols[key] = _find_col(cands, df_daily.columns)
        except ValueError:
            pass

    if "date" not in cols or "close" not in cols:
        raise ValueError("Input data must contain date and close columns")

    df = df_daily.rename(columns={cols[k]: k for k in cols})

    if "volume" not in df:
        df["volume"] = 0.0

    for col in ["open", "high", "low"]:
        if col not in df:
            df[col] = df["close"]

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df = df.groupby("date").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    if tf == "D":
        out = df
    else:
        freq = "W-FRI" if tf == "W" else "ME"
        out = df.resample(freq, label="right", closed="right").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["close"])
        if len(out) > 0:
            last_orig = df.index[-1]
            if out.index[-1] > last_orig:
                idx = out.index.to_numpy()
                idx[-1] = last_orig.to_datetime64()
                out.index = pd.DatetimeIndex(idx)

    out = out[["open", "high", "low", "close", "volume"]]
    out.index = pd.DatetimeIndex(out.index)
    return out


def resample_series_by_tf(dates: Iterable, prices: Iterable, tf: str,
                          volumes: Optional[Iterable] = None):
    s = pd.Series(np.asarray(prices, dtype=float), index=pd.DatetimeIndex(dates))
    v = None
    if volumes is not None:
        v = pd.Series(np.asarray(volumes, dtype=float), index=s.index)

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
    dates_tf = sr.index.to_numpy()
    if tf in ("W", "M") and len(dates_tf) > 0:
        last_orig = pd.DatetimeIndex(dates)[-1]
        if dates_tf[-1] > last_orig.to_datetime64():
            dates_tf[-1] = last_orig.to_datetime64()
    prices_tf = sr.values.astype(float)
    if v is not None:
        return dates_tf, prices_tf, vr.values.astype(float)
    return dates_tf, prices_tf


def nearest_trading_day_in(dates_index: Iterable, ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    di = pd.DatetimeIndex(dates_index)
    if ts in di:
        return ts
    idx = di.get_indexer([ts], method="nearest")[0]
    return di[idx]
