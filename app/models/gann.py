import numpy as np
import pandas as pd
from typing import Iterable

ANGLES = [1/8, 1/4, 1/3, 1/2, 1, 2, 3, 4, 8]
DEC_PLACES = 4
SCALE = 10 ** DEC_PLACES


def median_abs_close_delta(prices: Iterable[float]) -> float:
    m = pd.Series(prices).diff().abs().median()
    if not np.isfinite(m) or m == 0:
        m = max(1.0, float(np.std(prices) / 100.0))
    return float(m)


def compute_unit(prices: Iterable[float], mode: str = "median", k: float = 1.0,
                 dec_places: int = DEC_PLACES) -> float:
    prices = np.asarray(prices, dtype=float)
    if mode == "median":
        base = median_abs_close_delta(prices)
    else:  # atr
        diffs = pd.Series(prices).diff().abs()
        base = diffs.rolling(14).mean().iloc[-1]
        base = float(base) * k
    base = round(base, dec_places)
    return max(1.0 / SCALE, base)


def infer_direction_around(prices: Iterable[float], idx: int, window: int = 5) -> int:
    prices = np.asarray(prices, dtype=float)
    i0 = max(0, idx - window)
    i1 = min(len(prices), idx + window + 1)
    segment = prices[i0:i1]
    if len(segment) == 0:
        return +1
    if prices[idx] >= np.max(segment):
        return -1
    if prices[idx] <= np.min(segment):
        return +1
    left = prices[idx-1] if idx-1 >= 0 else prices[idx]
    right = prices[idx+1] if idx+1 < len(prices) else prices[idx]
    return -1 if prices[idx] > left and prices[idx] > right else +1
