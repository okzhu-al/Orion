import pandas as pd
from typing import Tuple, Iterable


def _find_col(candidates: Iterable[str], cols: Iterable[str]) -> str:
    lower = {str(c).lower(): c for c in cols}
    for k in candidates:
        if k in cols:
            return k
        lk = str(k).lower()
        if lk in lower:
            return lower[lk]
    for c in cols:
        for k in candidates:
            if str(k).lower() in str(c).lower():
                return c
    raise ValueError(f"Cannot find any of {list(candidates)} in columns: {list(cols)}")


def _extract_price_volume(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    dcol = _find_col(["date", "日期", "时间", "time"], df.columns)
    ccol = _find_col(["close", "收盘", "收盘价"], df.columns)
    try:
        vcol = _find_col(["volume", "成交量", "vol"], df.columns)
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


def load_price_volume(path_or_buf, sheet=0):
    if hasattr(path_or_buf, "read"):
        df = pd.read_excel(path_or_buf, sheet_name=sheet)
    else:
        df = pd.read_excel(path_or_buf, sheet_name=sheet)
    return _extract_price_volume(df)
