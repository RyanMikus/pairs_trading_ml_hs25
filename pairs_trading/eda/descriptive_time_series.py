# descriptive_time_series.py
# Minimal descriptive EDA for intraday equity time series (prices & returns)
# - Built for many symbols (~500): cross-sectional plots + representative drill-down
# - Keeps plotting inside this module (as requested)
# - Assumes long df with at least: ['timestamp', 'symbol', <price_col>]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew as _skew
from scipy.stats import kurtosis as _kurtosis


# =========================
# Helpers
# =========================

def _ensure_datetime(df, timestamp_col):
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=False, errors="coerce")
    out = out.dropna(subset=[timestamp_col])
    return out


def _pivot_prices_long_to_wide(df, timestamp_col, symbol_col, price_col, freq=None):
    """
    Long -> wide: index=timestamp, columns=symbol, values=price.
    If freq is provided, resample by last observation within each interval.
    """
    tmp = df[[timestamp_col, symbol_col, price_col]].copy()
    tmp = tmp.dropna(subset=[price_col, symbol_col, timestamp_col])
    tmp = tmp.sort_values([symbol_col, timestamp_col])

    wide = tmp.pivot(index=timestamp_col, columns=symbol_col, values=price_col).sort_index()

    if freq is not None:
        wide = wide.resample(freq).last()

    return wide


def compute_returns(prices_wide, log_return=True):
    """
    Compute (log) returns from wide price panel.
    """
    if log_return:
        return np.log(prices_wide).diff()
    return prices_wide.pct_change()


def _skew_series(x):
    x = x.dropna()
    if len(x) < 3:
        return np.nan
    return float(_skew(x.values, bias=False))


def _kurtosis_series(x):
    x = x.dropna()
    if len(x) < 4:
        return np.nan
    # Pearson kurtosis (normal => 3.0)
    return float(_kurtosis(x.values, fisher=False, bias=False))


def _safe_quantile(s, q):
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.quantile(q))


def _count_zeros(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    return float((s == 0).mean())


def _intraday_profile_index(dt_index):
    return pd.Index(dt_index.time, name="time")


# =========================
# Core stats
# =========================

def cross_section_return_stats(returns_wide):
    """
    Per-symbol summary of return distribution & a simple microstructure proxy.
    Minimal but useful set.
    """
    stats = pd.DataFrame(index=returns_wide.columns)

    stats["n_obs"] = returns_wide.count()
    stats["mean"] = returns_wide.mean()
    stats["std"] = returns_wide.std()
    stats["q01"] = returns_wide.apply(lambda s: _safe_quantile(s, 0.01))
    stats["q05"] = returns_wide.apply(lambda s: _safe_quantile(s, 0.05))
    stats["median"] = returns_wide.median()
    stats["q95"] = returns_wide.apply(lambda s: _safe_quantile(s, 0.95))
    stats["q99"] = returns_wide.apply(lambda s: _safe_quantile(s, 0.99))
    stats["skew"] = returns_wide.apply(_skew_series)
    stats["kurtosis"] = returns_wide.apply(_kurtosis_series)  # normal ≈ 3
    stats["pct_zero_returns"] = returns_wide.apply(_count_zeros)

    abs_r = returns_wide.abs()
    stats["mean_abs_return"] = abs_r.mean()
    stats["abs_q99"] = abs_r.apply(lambda s: _safe_quantile(s, 0.99))

    return stats.sort_values("std", ascending=False)


def rolling_volatility(returns_wide, window):
    """
    Rolling volatility (std) per symbol. For intraday, window is number of bars.
    """
    return returns_wide.rolling(window).std()


def intraday_volatility_profile(returns_wide):
    """
    Mean absolute return per time-of-day.
    Output: index=time-of-day, columns=symbol.
    """
    if not isinstance(returns_wide.index, pd.DatetimeIndex):
        raise TypeError("returns_wide index must be DatetimeIndex for intraday profile.")

    times = _intraday_profile_index(returns_wide.index)
    return returns_wide.abs().groupby(times).mean()


# =========================
# Representative selection (for 500 symbols)
# =========================

def select_representative_symbols(stats, n_top_vol=5, n_bottom_vol=5, n_median_vol=5):
    """
    Pick a small set of symbols for drill-down plots:
    - Top vol
    - Bottom vol
    - Around median vol
    """
    s = stats.dropna(subset=["std"]).copy()
    if s.empty:
        return []

    s_sorted = s.sort_values("std", ascending=False)
    top = list(s_sorted.head(n_top_vol).index)
    bottom = list(s_sorted.tail(n_bottom_vol).index)

    med = float(s["std"].median())
    s["dist_to_median"] = (s["std"] - med).abs()
    median_syms = list(s.sort_values("dist_to_median").head(n_median_vol).index)

    out = []
    for sym in top + median_syms + bottom:
        if sym not in out:
            out.append(sym)
    return out


# =========================
# Plotting (in-module)
# =========================

def plot_cross_section_hist(stats, col, bins=50, title=None):
    s = stats[col].dropna()
    if s.empty:
        print("[plot] No data for histogram:", col)
        return
    s.hist(bins=bins)
    plt.title(title or "Cross-section distribution: {}".format(col))
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_cross_section_box(stats, cols, title=None):
    data = [stats[c].dropna().values for c in cols if c in stats.columns]
    labels = [c for c in cols if c in stats.columns]
    if not data:
        print("[plot] No data for boxplot.")
        return
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title or "Cross-section boxplot")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_intraday_profile(profile, title=None):
    """
    Plots the cross-section mean intraday profile (best default for ~500 symbols).
    """
    if profile is None or profile.empty:
        print("[plot] No data for intraday profile.")
        return

    profile.mean(axis=1).plot()
    plt.title(title or "Intraday volatility profile (mean |return| across symbols)")
    plt.xlabel("time-of-day")
    plt.ylabel("mean |return|")
    plt.tight_layout()
    plt.show()


def plot_representative_time_series(returns_wide, symbols, kind="returns", title=None):
    symbols = [s for s in symbols if s in returns_wide.columns]
    if not symbols:
        print("[plot] No representative symbols available to plot.")
        return

    n = len(symbols)
    fig_h = min(2.2 * n + 1.5, 30)
    returns_wide[symbols].plot(subplots=True, figsize=(12, fig_h), legend=False)
    plt.suptitle(title or "Representative {} time series ({} symbols)".format(kind, n))
    plt.tight_layout()
    plt.show()


# =========================
# End-to-end EDA step
# =========================

def run_descriptive_eda(
    df,
    timestamp_col="timestamp",
    symbol_col="symbol",
    price_col="price",
    freq=None,
    log_return=True,
    drop_nonpositive_prices=True,
    plot=True,
    n_top_vol=5,
    n_bottom_vol=5,
    n_median_vol=5,
    max_representative_plots=15
):
    """
    Minimal descriptive EDA pipeline step.

    Returns dict with:
      - prices_wide
      - returns_wide
      - stats
      - representative_symbols
      - intraday_profile
    """
    df2 = _ensure_datetime(df, timestamp_col)

    if drop_nonpositive_prices:
        df2 = df2[df2[price_col] > 0].copy()

    prices_wide = _pivot_prices_long_to_wide(
        df2,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        price_col=price_col,
        freq=freq
    )

    returns_wide = compute_returns(prices_wide, log_return=log_return).dropna(how="all")

    stats = cross_section_return_stats(returns_wide)

    reps = select_representative_symbols(
        stats,
        n_top_vol=n_top_vol,
        n_bottom_vol=n_bottom_vol,
        n_median_vol=n_median_vol
    )

    intraday_prof = None
    if isinstance(returns_wide.index, pd.DatetimeIndex):
        intraday_prof = intraday_volatility_profile(returns_wide)

    if plot:
        plot_cross_section_hist(stats, "std", title="Cross-section: return volatility (std)")
        plot_cross_section_hist(stats, "mean_abs_return", title="Cross-section: mean absolute return")
        plot_cross_section_hist(stats, "pct_zero_returns", title="Cross-section: % zero returns (liquidity proxy)")
        plot_cross_section_box(stats, ["mean", "std", "skew", "kurtosis"], title="Cross-section summary (boxplots)")

        if intraday_prof is not None:
            plot_intraday_profile(intraday_prof)

        reps_to_plot = reps[:max_representative_plots]
        plot_representative_time_series(returns_wide, reps_to_plot, kind="log-returns")

    return {
        "prices_wide": prices_wide,
        "returns_wide": returns_wide,
        "stats": stats,
        "representative_symbols": reps,
        "intraday_profile": intraday_prof,
    }