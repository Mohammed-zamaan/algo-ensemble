"""Indicators (Wilder RMA, ATR, ADX) from StageB notebook [file:35]."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder RMA via EMA(alpha=1/length)."""
    return series.ewm(alpha=1.0 / float(length), adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    return rma(true_range(high, low, close), length)


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, di_len: int, adx_smooth: int) -> pd.Series:
    up = high.diff()
    dn = -low.diff()

    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr = true_range(high, low, close)
    tr_rma = rma(tr, di_len)

    plus_rma = rma(pd.Series(plus_dm, index=high.index), di_len)
    minus_rma = rma(pd.Series(minus_dm, index=high.index), di_len)

    plus_di = 100.0 * (plus_rma / tr_rma)
    minus_di = 100.0 * (minus_rma / tr_rma)

    dx = 100.0 * (plus_di.sub(minus_di).abs() / plus_di.add(minus_di))
    adx = rma(dx, adx_smooth)
    return adx
