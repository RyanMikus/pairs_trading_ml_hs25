import numpy as np
import pandas as pd

from ..features.technical import sma, ema, rsi, rolling_volatility


def symbols_from_pairs(pairs_df, top_n=50):
    """
    Public helper: derive (pairs_top, symbols) from a top-N slice of a pair ranking.
    """
    pairs_top = pairs_df.head(top_n)[["symbol1", "symbol2"]].copy()
    symbols = sorted(set(pairs_top["symbol1"]).union(set(pairs_top["symbol2"])))
    return pairs_top, symbols


def _valid_start_from_frames(frames):
    starts = []
    for df in frames:
        if df is None or df.empty:
            continue
        s = df.dropna().index.min()
        if pd.notna(s):
            starts.append(s)
    return max(starts) if starts else None


def indicator_similarity_for_pairs(
    pairs_df,
    prices_wide,
    log_prices,
    top_n=50,
    sma_window=20,
    ema_span=20,
    rsi_window=14,
    vol_window=20
):
    """
    Build indicator-based similarity matrices on the symbol universe induced by top-N pairs.

    Returns:
      meta: dict
      corr: dict[str -> DataFrame]   (symbol x symbol correlation)
      dist: dict[str -> DataFrame]   (symbol x symbol distance = 1 - corr)
    """
    pairs_top, symbols = symbols_from_pairs(pairs_df, top_n)

    prices_sub = prices_wide[symbols].copy()
    log_prices_sub = log_prices[symbols].copy()

    # Indicators
    sma_df = sma(prices_sub, window=sma_window)
    ema_df = ema(prices_sub, span=ema_span)
    rsi_df = rsi(prices_sub, window=rsi_window)
    vol_df = rolling_volatility(log_prices_sub, window=vol_window)

    # Align to common valid start (avoid early NaNs driving corr inconsistently)
    valid_start = _valid_start_from_frames([sma_df, ema_df, rsi_df, vol_df])
    if valid_start is not None:
        sma_df = sma_df.loc[valid_start:]
        ema_df = ema_df.loc[valid_start:]
        rsi_df = rsi_df.loc[valid_start:]
        vol_df = vol_df.loc[valid_start:]

    corr = {
        "SMA_%d" % sma_window: sma_df.corr(),
        "EMA_%d" % ema_span: ema_df.corr(),
        "RSI_%d" % rsi_window: rsi_df.corr(),
        "VOL_%d" % vol_window: vol_df.corr(),
    }
    dist = {k: (1.0 - v) for k, v in corr.items()}

    meta = {
        "top_n": int(top_n),
        "n_pairs": int(len(pairs_top)),
        "pairs_top": pairs_top,
        "symbols": symbols,
        "n_symbols": int(len(symbols)),
        "valid_start": valid_start,
        "params": {
            "sma_window": int(sma_window),
            "ema_span": int(ema_span),
            "rsi_window": int(rsi_window),
            "vol_window": int(vol_window),
        }
    }
    return meta, corr, dist


def pair_corr_summary(pairs_df, corr_df, top_n=10):
    """
    Attach correlation (corr_df.loc[symbol1, symbol2]) for the top-N pairs.
    """
    pairs_top = pairs_df.head(top_n)[["symbol1", "symbol2"]].copy()

    vals = []
    for a, b in pairs_top[["symbol1", "symbol2"]].itertuples(index=False, name=None):
        if (a in corr_df.index) and (b in corr_df.columns):
            vals.append(float(corr_df.loc[a, b]))
        else:
            vals.append(np.nan)

    out = pairs_top.copy()
    out["indicator_corr"] = vals
    return out