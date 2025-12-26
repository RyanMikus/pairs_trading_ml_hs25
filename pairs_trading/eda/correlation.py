import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_analysis(
    prices_wide,
    method="pearson",
    strong_threshold=0.97,
    top_n_heatmap=19,
    plot=True,
    title=None,
):
    """
    Explorative Korrelationsanalyse für Wide-Format Preis- oder Logpreis-Zeitreihen.

    Parameters
    ----------
    prices_wide : pd.DataFrame
        Index = datetime, Columns = Symbole, Values = Preise oder Logpreise
    method : str
        'pearson' oder 'spearman'
    strong_threshold : float
        Schwelle für stark korrelierte Paare (nur obere Dreiecksmatrix)
    top_n_heatmap : int
        Anzahl repräsentativer Symbole für Heatmap
    plot : bool
        Heatmap plotten
    title : str or None
        Plot-Titel

    Returns
    -------
    dict with keys:
        corr
        strong_pairs
        representative_symbols
        vmin_offdiag
        vmax_offdiag
        corr_sub
    """

    if prices_wide.empty:
        raise ValueError("prices_wide ist leer.")

    # 1) Korrelationsmatrix
    corr = prices_wide.corr(method=method)

    # 2) Stark korrelierte Paare (nur obere Dreiecksmatrix)
    pairs = []
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and val > strong_threshold:
                pairs.append((cols[i], cols[j], float(val)))

    strong_pairs = (
        pd.DataFrame(pairs, columns=["symbol_1", "symbol_2", "correlation"])
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )

    # 3) Repräsentative Symbole (nach mittlerer Korrelation)
    mean_corr = corr.mean().sort_values(ascending=False)
    representative_symbols = mean_corr.head(top_n_heatmap).index

    # 4) Submatrix + Off-Diagonal-Skalierung
    corr_sub = corr.loc[representative_symbols, representative_symbols]

    mask = np.eye(len(corr_sub), dtype=bool)
    offdiag = corr_sub.where(
        ~pd.DataFrame(mask, index=corr_sub.index, columns=corr_sub.columns)
    )

    vmin = offdiag.min().min()
    vmax = offdiag.max().max()

    # 5) Plot
    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_sub,
            mask=mask,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            linewidths=0.2,
            annot_kws={"size": 8},
        )
        plt.title(title or f"Korrelations-Heatmap der Top {top_n_heatmap} Aktien (reskaliert)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    return {
        "corr": corr,
        "strong_pairs": strong_pairs,
        "representative_symbols": representative_symbols,
        "vmin_offdiag": vmin,
        "vmax_offdiag": vmax,
        "corr_sub": corr_sub,
    }
