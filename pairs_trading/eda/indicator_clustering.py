import numpy as np
import pandas as pd


def agglom_labels_from_dist(dist_df, n_clusters, linkage="average"):
    """
    dist_df: square DataFrame (symbols x symbols), e.g. dist = 1 - corr
    returns: pd.Series cluster labels indexed by symbol
    """
    dist_df = dist_df.copy()
    dist_df = dist_df.loc[dist_df.index, dist_df.index]

    dist_vals = dist_df.values.astype(float)

    if np.isnan(dist_vals).any():
        max_val = np.nanmax(dist_vals)
        dist_vals = np.where(np.isnan(dist_vals), max_val, dist_vals)

    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage
    )
    labels = model.fit_predict(dist_vals)
    return pd.Series(labels, index=dist_df.index, name="cluster")


def cluster_count_table(label_dict):
    """
    label_dict: {name: pd.Series(symbol -> cluster)}
    returns: DataFrame with cluster counts per method
    """
    counts = []
    for name, labels in label_dict.items():
        vc = labels.value_counts().sort_index()
        counts.append(vc.to_frame(name))
    return pd.concat(counts, axis=1).fillna(0).astype(int)


def pair_proximity_table(pairs_df, corr_df, labels, top_n=10, corr_col_name="indicator_corr", same_cluster_col="same_cluster"):
    """
    For top-N pairs: add correlation and whether both symbols fall into same cluster.
    """
    pairs_top = pairs_df.head(top_n)[["symbol1", "symbol2"]].copy()

    corr_vals = []
    same = []
    for a, b in pairs_top[["symbol1", "symbol2"]].itertuples(index=False, name=None):
        if (a in corr_df.index) and (b in corr_df.columns):
            corr_vals.append(float(corr_df.loc[a, b]))
        else:
            corr_vals.append(np.nan)

        if (a in labels.index) and (b in labels.index):
            same.append(bool(labels.loc[a] == labels.loc[b]))
        else:
            same.append(np.nan)

    out = pairs_top.copy()
    out[corr_col_name] = corr_vals
    out[same_cluster_col] = same
    return out


def plot_corr_heatmap(corr_df, title, figsize=(12, 10)):
    """
    Minimal report-friendly heatmap with diagonal masked and vmin/vmax based on off-diagonal.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr_df = corr_df.copy()

    mask = np.eye(corr_df.shape[0], dtype=bool)
    offdiag = corr_df.where(~pd.DataFrame(mask, index=corr_df.index, columns=corr_df.columns))

    vmin = float(offdiag.min().min())
    vmax = float(offdiag.max().max())

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        annot=False,
        linewidths=0.2
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()