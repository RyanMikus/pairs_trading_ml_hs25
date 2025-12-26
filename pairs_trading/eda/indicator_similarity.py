import numpy as np
import pandas as pd

from pairs_trading import features


def symbols_from_pairs(pairs_df, col1="symbol1", col2="symbol2"):
    """
    Extract unique symbols from a pairs dataframe.
    Expected columns: col1, col2 (default: symbol1, symbol2)
    """
    if col1 not in pairs_df.columns or col2 not in pairs_df.columns:
        raise KeyError(f"pairs_df must contain columns '{col1}' and '{col2}'")

    syms = sorted(set(pairs_df[col1].dropna()).union(set(pairs_df[col2].dropna())))
    return syms


def indicator_similarity_for_pairs(
    pairs_df,
    prices_wide,
    log_prices,
    top_n=50
):
    """
    Compute indicator-based correlation and distance matrices for the symbol universe
    induced by the top-N pairs.

    - SMA/EMA/RSI computed on prices_wide
    - Volatility computed on log_prices (log returns)
    """
    # 1) select top pairs
    pairs_top = pairs_df.head(top_n)[["symbol1", "symbol2"]].copy()

    # 2) derive symbols and validate existence in prices_wide
    symbols = symbols_from_pairs(pairs_top)
    symbols = [s for s in symbols if s in prices_wide.columns]

    if len(symbols) < 2:
        raise ValueError("Not enough symbols found in prices_wide for correlation analysis.")

    prices_sub = prices_wide[symbols]
    log_sub = log_prices[symbols]

    # 3) indicators
    sma_20 = features.sma(prices_sub, window=20)
    ema_20 = features.ema(prices_sub, span=20)
    rsi_14 = features.rsi(prices_sub, window=14)
    vol_20 = features.rolling_volatility(log_sub, window=20)

    # 4) align start date (avoid initial NaN regions)
    valid_start = max(
        sma_20.dropna().index.min(),
        ema_20.dropna().index.min(),
        rsi_14.dropna().index.min(),
        vol_20.dropna().index.min(),
    )

    sma_20 = sma_20.loc[valid_start:]
    ema_20 = ema_20.loc[valid_start:]
    rsi_14 = rsi_14.loc[valid_start:]
    vol_20 = vol_20.loc[valid_start:]

    # 5) correlations
    corr = {
        "SMA_20": sma_20.corr(),
        "EMA_20": ema_20.corr(),
        "RSI_14": rsi_14.corr(),
        "VOL_20": vol_20.corr(),
    }

    # 6) distances
    dist = {k: (1.0 - v) for k, v in corr.items()}

    meta = {
        "top_n": int(top_n),
        "n_pairs": int(len(pairs_top)),
        "n_symbols": int(len(symbols)),
        "symbols": symbols,
        "valid_start": valid_start,
        "pairs_top": pairs_top,
    }

    return meta, corr, dist


def pair_corr_summary(pairs_df, corr_matrix, top_n=20):
    """
    For the top-N pairs, attach the indicator correlation corr(symbol1, symbol2).
    """
    df = pairs_df.head(top_n)[["symbol1", "symbol2"]].copy()
    df["indicator_corr"] = [
        corr_matrix.loc[a, b] if (a in corr_matrix.index and b in corr_matrix.columns) else np.nan
        for a, b in df[["symbol1", "symbol2"]].itertuples(index=False, name=None)
    ]
    return df
