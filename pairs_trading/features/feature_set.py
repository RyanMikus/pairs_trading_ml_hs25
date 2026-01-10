"""
ML Feature Panel Construction (Spread-basiert, rollenorientiert).

Scope:
- Erzeugt kompakte, modell-taugliche Feature-Panels
- Fokus auf Stabilität, Klarheit und Wiederverwendbarkeit
- Keine Visualisierung, keine Notebook-Abhängigkeiten

Dieses Modul ist die Brücke zwischen EDA und Modellierung.
"""

import pandas as pd
import numpy as np

from .technical import (
    rsi,
    roc,
    momentum,
    rolling_volatility,
    bollinger_bands,
)


def build_feature_panel(
    log_prices,
    pairs,
    rsi_window=14,
    z_window=60,
    vol_window=20,
):
    """
    Build a compact ML feature panel for pair trading based on SPREADS.

    Features (per pair):
      - RSI_spread        (Signal)
      - ZScore_spread     (State)
      - VOL_spread        (Regime)
      - return_spread     (Timing)

    Output:
      DataFrame with MultiIndex columns:
        level 0: feature name
        level 1: pair name (e.g. AAPL_MSFT)
    """
    features = []

    for s1, s2 in pairs:
        if s1 not in log_prices.columns or s2 not in log_prices.columns:
            continue

        pair_name = f"{s1}_{s2}"
        prices = log_prices[[s1, s2]].dropna()
        if len(prices) < max(z_window, vol_window):
            continue

        # ------------------------
        # Spread construction
        # ------------------------
        beta, _ = np.polyfit(prices[s2].values, prices[s1].values, 1)
        spread = prices[s1] - beta * prices[s2]

        df = pd.DataFrame(index=spread.index)
        df["spread"] = spread

        spread_wide = pd.concat([spread, spread], axis=1)
        spread_wide.columns = ["SPREAD", "SPREAD_DUMMY"]

        # ------------------------
        # Features
        # ------------------------

        df[f"RSI_{rsi_window}"] = rsi(spread_wide, window=rsi_window)["SPREAD"]

        mean = spread.rolling(z_window).mean()
        std = spread.rolling(z_window).std().replace(0, np.nan)
        df[f"Z_{z_window}"] = (spread - mean) / std

        df[f"VOL_{vol_window}"] = rolling_volatility(
            spread_wide, window=vol_window
        )["SPREAD"]

        df["RETURN"] = spread.diff()

        # ------------------------
        # Formatting
        # ------------------------
        df = df.dropna()
        df.columns = pd.MultiIndex.from_product(
            (df.columns, [pair_name])
        )

        features.append(df)

    if not features:
        raise ValueError("No valid pair features could be constructed.")

    return pd.concat(features, axis=1).sort_index(axis=1)