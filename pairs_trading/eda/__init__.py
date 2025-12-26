from .data_loading import fetch_data

from .data_quality import (
    analyze_missing_values,
    analyze_missing_coverage,
    plot_intraday_missing_distribution,
    analyze_nan_gaps_global,
    gap_len_counts,
    long_gap_summary,
    gap_stats_stocks_with_min_gap,
    plot_gap_length_hist,
    plot_stocks_vs_gap_threshold,
)

from .data_cleaning import (
    filter_symbols_by_full_coverage,
    find_symbols_with_long_gaps,
    drop_symbols,
    ensure_datetime_and_sort,
    drop_duplicate_rows,          # ← NEU
    fill_close_gaps_per_symbol,
    symbol_lengths,
    expected_length_mode,
    symbols_with_unexpected_length,
    removed_symbols_between,
)

from .descriptive_time_series import (
    compute_returns,
    cross_section_return_stats,
    rolling_volatility,
    intraday_volatility_profile,
    select_representative_symbols,
    plot_cross_section_hist,
    plot_cross_section_box,
    plot_intraday_profile,
    plot_representative_time_series,
    run_descriptive_eda,
)

from .correlation import correlation_analysis

# --- NEW: cointegration ---
from .cointegration import (
    compute_daily_cointegration,
    compute_pair_stats,
    rank_pairs,
    plot_daily_pvalues,
    plot_coint_network,
    compute_daily_spread_stats,
    plot_daily_mean_spreads,
    plot_intraday_prices_last_day,
    plot_intraday_spreads_last_day,
)

from .indicator_similarity import (
    symbols_from_pairs,
    indicator_similarity_for_pairs,
    pair_corr_summary,
)

__all__ = [
    "fetch_data",

    # data_quality
    "analyze_missing_values",
    "analyze_missing_coverage",
    "plot_intraday_missing_distribution",
    "analyze_nan_gaps_global",
    "gap_len_counts",
    "long_gap_summary",
    "gap_stats_stocks_with_min_gap",
    "plot_gap_length_hist",
    "plot_stocks_vs_gap_threshold",

    # data_cleaning
    "filter_symbols_by_full_coverage",
    "find_symbols_with_long_gaps",
    "drop_symbols",
    "ensure_datetime_and_sort",
    "drop_duplicate_rows",
    "fill_close_gaps_per_symbol",
    "symbol_lengths",
    "expected_length_mode",
    "symbols_with_unexpected_length",
    "removed_symbols_between",

    # descriptive_time_series
    "compute_returns",
    "cross_section_return_stats",
    "rolling_volatility",
    "intraday_volatility_profile",
    "select_representative_symbols",
    "plot_cross_section_hist",
    "plot_cross_section_box",
    "plot_intraday_profile",
    "plot_representative_time_series",
    "run_descriptive_eda",

    # correlation
    "correlation_analysis",

    # cointegration
    "compute_daily_cointegration",
    "compute_pair_stats",
    "rank_pairs",
    "plot_daily_pvalues",
    "plot_coint_network",
    "compute_daily_spread_stats",
    "plot_daily_mean_spreads",
    "plot_intraday_prices_last_day",
    "plot_intraday_spreads_last_day",

    # indicator similarity
    "symbols_from_pairs",
    "indicator_similarity_for_pairs",
    "pair_corr_summary",
]