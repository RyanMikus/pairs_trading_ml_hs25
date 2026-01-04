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

from .indicator_clustering import (
    agglom_labels_from_dist,
    cluster_count_table,
    pair_proximity_table,
    plot_corr_heatmap,
)

__all__ = [
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

    # indicator clustering (NEW)
    "agglom_labels_from_dist",
    "cluster_count_table",
    "pair_proximity_table",
    "plot_corr_heatmap",
]