import pandas as pd
from typing import Tuple, List

from .technical import sma, ema, macd, rsi, roc, rolling_volatility, bollinger_bands


def build_feature_panel(
    log_prices: pd.DataFrame,
    sma_windows: Tuple[int, ...] = (20,),
    ema_spans: Tuple[int, ...] = (20,),
    rsi_windows: Tuple[int, ...] = (14,),
    roc_windows: Tuple[int, ...] = (10,),
    vol_windows: Tuple[int, ...] = (20,),
    bollinger_windows: Tuple[int, ...] = (20,)
) -> pd.DataFrame:
    """
    Returns a wide feature panel with MultiIndex columns:
      level0 = feature_name
      level1 = symbol

    This is convenient for later correlation/distance/clustering pipelines.
    """
    parts = []  # type: List[pd.DataFrame]

    # SMA
    for w in sma_windows:
        df = sma(log_prices, window=w)
        df.columns = pd.MultiIndex.from_product(([f"SMA_{w}"], df.columns))
        parts.append(df)

    # EMA
    for s in ema_spans:
        df = ema(log_prices, span=s)
        df.columns = pd.MultiIndex.from_product(([f"EMA_{s}"], df.columns))
        parts.append(df)

    # RSI
    for w in rsi_windows:
        df = rsi(log_prices, window=w)
        df.columns = pd.MultiIndex.from_product(([f"RSI_{w}"], df.columns))
        parts.append(df)

    # ROC
    for w in roc_windows:
        df = roc(log_prices, window=w)
        df.columns = pd.MultiIndex.from_product(([f"ROC_{w}"], df.columns))
        parts.append(df)

    # Volatility
    for w in vol_windows:
        df = rolling_volatility(log_prices, window=w)
        df.columns = pd.MultiIndex.from_product(([f"VOL_{w}"], df.columns))
        parts.append(df)

    # Bollinger: use band width as a single feature (upper - lower)
    for w in bollinger_windows:
        mid, upper, lower = bollinger_bands(log_prices, window=w)
        bw = (upper - lower)
        bw.columns = pd.MultiIndex.from_product(([f"BB_WIDTH_{w}"], bw.columns))
        parts.append(bw)

    # MACD defaults – 3 blocks
    macd_line, signal_line, hist = macd(log_prices)
    macd_line.columns = pd.MultiIndex.from_product((["MACD_LINE"], macd_line.columns))
    signal_line.columns = pd.MultiIndex.from_product((["MACD_SIGNAL"], signal_line.columns))
    hist.columns = pd.MultiIndex.from_product((["MACD_HIST"], hist.columns))
    parts.extend([macd_line, signal_line, hist])

    feature_panel = pd.concat(parts, axis=1).sort_index(axis=1)
    return feature_panel
