from typing import Iterable


def compute_unit(prices: Iterable[float], mode: str = "median", k: float = 1.0,
                 dec_places: int = 4) -> float:
    raise NotImplementedError


def infer_direction(prices: Iterable[float], idx: int, mode: str = "auto") -> int:
    raise NotImplementedError
