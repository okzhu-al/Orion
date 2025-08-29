import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional


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
