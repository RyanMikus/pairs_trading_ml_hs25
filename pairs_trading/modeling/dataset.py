"""
pairs_trading.modeling.dataset

Utilities zur Konstruktion von Trainings- und Test-Datasets aus Feature-Panels.

Wichtiger Hinweis:
- Diese Funktionen definieren die Zielvariable (Label) explizit.
- Refactorings dürfen Feature-Selektion, Label-Definition und Rückgaben nicht verändern.
"""

import pandas as pd


# ============================================================================
# Dataset-Konstruktion
# ============================================================================

def build_Xy_from_feature_panel(
    panel: pd.DataFrame,
    k: int,
    rsi_w: int,
    z_w: int,
    vol_w: int,
    return_meta: bool = False,
):
    """
    Erzeugt Feature-Matrix X und Zielvariable y aus einem Feature-Panel.

    Zielvariable:
    - y = 1, falls |Z_{t+k}| < |Z_t| (Mean-Reversion-Indikator)

    Rückgabe:
    - X : Feature-Matrix
    - y : Zielvariable
    - meta (optional): DataFrame mit ['datetime', 'pair', 'z', 'spread_ret_k', 'spread'].
      'z', 'spread_ret_k' und 'spread' werden hier (und nur hier) berechnet und
      sind garantiert zeilenweise exakt auf X/y ausgerichtet, da sie aus demselben
      maskierten long_df stammen. Eine fruehere, unabhaengige Rekonstruktion
      dieser Werte in walkforward.py (via panel.stack()) verwendete eine
      andere Zeilenreihenfolge und ordnete z/spread_ret_k dadurch teils
      falschen (datetime, pair)-Kombinationen zu. Siehe SESSION_CHANGES.md,
      Abschnitt 7.5. 'spread' (rohes Spread-Niveau, log-Preis-Einheiten) wurde
      ergaenzt, um in 06_Backtest.ipynb den tatsaechlich realisierten Ertrag
      ueber die variable Haltedauer einer Position berechnen zu koennen, siehe
      SESSION_CHANGES.md, Abschnitt 8.4.
    """
    pair_names = panel.columns.get_level_values(1).unique().tolist()

    blocks = []
    for p in pair_names:
        dfp = panel.xs(p, axis=1, level=1).copy()
        dfp["pair"] = p
        blocks.append(dfp)

    long_df = (
        pd.concat(blocks, axis=0)
        .reset_index()
        .rename(columns={"index": "datetime"})
    ).sort_values(["pair", "datetime"])

    z_col = f"Z_{z_w}_spread"
    if z_col not in long_df.columns:
        raise ValueError(f"Missing column: {z_col}")
    if "spread" not in long_df.columns:
        raise ValueError("Missing column: spread")

    # Gruppierung zusaetzlich nach Handelstag: ein reiner groupby("pair")-Shift
    # wuerde bei den letzten k Zeilen eines Tages die "Zukunft" aus dem
    # naechsten Handelstag ziehen (18+ Stunden statt k Minuten voraus).
    # Siehe SESSION_CHANGES.md, Abschnitt 7.3.
    day = long_df["datetime"].dt.normalize()

    z_now = long_df[z_col].astype(float)
    z_fut = long_df.groupby(["pair", day])[z_col].shift(-k).astype(float)

    # spread_ret_k (fuer die Trading-Inspektion) auf Basis des ROHEN Spreads,
    # nicht der Z-Score-Differenz -- siehe SESSION_CHANGES.md, Abschnitt 7.1.
    spread_now = long_df["spread"].astype(float)
    spread_fut = long_df.groupby(["pair", day])["spread"].shift(-k).astype(float)

    y = (z_fut.abs() < z_now.abs()).astype(int)

    X = long_df[
        [
            f"RSI_{rsi_w}_spread",
            f"Z_{z_w}_spread",
            f"VOL_{vol_w}_spread",
            "return_pct_spread",
        ]
    ].copy()

    # z_fut.notna() statt y.notna(): der Vergleich oben macht y nie selbst zu
    # NaN (NaN < x wird in pandas zu False, nicht NaN), daher wuerde
    # 'y.notna()' nichts herausfiltern und Zeilen ohne definierte Zukunft
    # wuerden faelschlich als y=0 gewertet. Siehe SESSION_CHANGES.md, Abschnitt 7.2.
    valid = X.notna().all(axis=1) & z_fut.notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    meta = None
    if return_meta:
        meta = long_df.loc[valid, ["datetime", "pair"]].reset_index(drop=True)
        meta["z"] = z_now.loc[valid].reset_index(drop=True)
        meta["spread_ret_k"] = (spread_fut - spread_now).loc[valid].reset_index(drop=True)
        # Rohes Spread-Niveau (log-Preis-Einheiten), zusaetzlich zur bereits
        # vorhandenen k-Schritte-Differenz spread_ret_k: wird fuer 06_Backtest.ipynb
        # benoetigt, um den tatsaechlich realisierten Ertrag ueber die (variable)
        # Haltedauer einer Position zu berechnen (exit_spread - entry_spread), statt
        # den fixen k-Minuten-Horizont von spread_ret_k als Naeherung zu missbrauchen.
        meta["spread"] = spread_now.loc[valid].reset_index(drop=True)

    return X, y, meta