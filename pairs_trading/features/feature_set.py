"""
ML Feature Panel Construction (Spread-basiert, rollenorientiert).

Finales Feature-Set für intraday pairs trading mit Rolling-Window-Training.

Features (pro Pair):
- RSI_14_spread        (Signal)
- Z_60_spread          (State)
- VOL_20_spread        (Regime)
- return_pct_spread   (Timing)

Design:
- Leak-free (keine Zukunftsinformation)
- Skalenstabil über Paare
- RF- und XGBoost-kompatibel
- Keine Rolling-Window-Logik hier
"""

import numpy as np
import pandas as pd

from .technical import (
    rsi,
    rolling_volatility,
)


def build_feature_panel(
    log_prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    rsi_window: int = 14,
    z_window: int = 60,
    vol_window: int = 20,
    beta_fit_end=None,
) -> pd.DataFrame:
    """
    Build a compact ML-ready feature panel on SPREAD level.

    Parameters
    ----------
    log_prices : pd.DataFrame
        Wide-format DataFrame of log prices (DatetimeIndex, symbols as columns).
        Expected to be built from close_filled.
    pairs : list of tuple(str, str)
        List of symbol pairs, e.g. [("AAPL", "MSFT"), ...]
    rsi_window : int
    z_window : int
    vol_window : int
    beta_fit_end : optional Timestamp
        Falls gesetzt, wird der Hedge-Ratio (beta) je Paar NUR auf Zeilen mit
        Index <= beta_fit_end gefittet (typischerweise das Ende des
        Trainingsfensters), aber auf den gesamten uebergebenen Zeitraum
        angewendet. Ohne dieses Argument (Default None) wird beta wie zuvor
        auf den gesamten uebergebenen 'log_prices'-Zeitraum gefittet.

        Wird dies weggelassen und 'log_prices' isoliert nur den Testtag
        enthalten (wie es die aeltere Walk-Forward-Logik tat), fliesst
        Information aus dem gesamten Testtag -- auch aus zukuenftigen Minuten
        relativ zu jedem einzelnen Zeitpunkt -- in die Spread-Definition ein
        (Look-ahead-Bias). Siehe SESSION_CHANGES.md, Abschnitt 7.4.

    Returns
    -------
    pd.DataFrame
        Feature panel with MultiIndex columns:
        level 0: feature name
        level 1: pair name (e.g. AAPL_MSFT)
    """

    feature_blocks = []

    for s1, s2 in pairs:
        if s1 not in log_prices.columns or s2 not in log_prices.columns:
            continue

        pair_name = f"{s1}_{s2}"
        prices = log_prices[[s1, s2]].dropna()

        if len(prices) < max(rsi_window, z_window, vol_window):
            continue

        # -------------------------------------------------
        # Spread construction (static hedge ratio)
        # -------------------------------------------------
        if beta_fit_end is not None:
            fit_prices = prices.loc[prices.index <= beta_fit_end]
        else:
            fit_prices = prices

        if len(fit_prices) < 2:
            continue

        beta, _ = np.polyfit(fit_prices[s2].values, fit_prices[s1].values, 1)
        spread = prices[s1] - beta * prices[s2]

        df = pd.DataFrame(index=spread.index)
        df["spread"] = spread

        # Wide-format requirement for technical indicators
        spread_wide = pd.concat([spread, spread], axis=1)
        spread_wide.columns = ["SPREAD", "SPREAD_DUMMY"]

        # -------------------------------------------------
        # Features
        # -------------------------------------------------

        # Signal
        df[f"RSI_{rsi_window}_spread"] = rsi(
            spread_wide, window=rsi_window
        )["SPREAD"]

        # State
        mean = spread.rolling(z_window).mean()
        std = spread.rolling(z_window).std().replace(0, np.nan)
        df[f"Z_{z_window}_spread"] = (spread - mean) / std

        # Regime
        df[f"VOL_{vol_window}_spread"] = rolling_volatility(
            spread_wide, window=vol_window
        )["SPREAD"]

        # Timing / Magnitude (skaleninvariant)
        prev_abs = spread.shift(1).abs().replace(0, np.nan)
        df["return_pct_spread"] = (
            (spread - spread.shift(1)) / prev_abs * 100.0
        )

        # -------------------------------------------------
        # Final formatting
        # -------------------------------------------------
        df = df.dropna()

        df.columns = pd.MultiIndex.from_product(
            (df.columns, [pair_name])
        )

        feature_blocks.append(df)

    if not feature_blocks:
        raise ValueError("No valid pair features could be constructed.")

    return (
        pd.concat(feature_blocks, axis=1)
        .sort_index(axis=1)
        .sort_index()
    )


def build_train_test_feature_panels(
    log_prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    train_days,
    test_day,
    rsi_window: int = 14,
    z_window: int = 60,
    vol_window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baut Trainings- und Test-Feature-Panels in einem gemeinsamen Aufruf.

    Der Hedge-Ratio (beta) je Paar wird ausschliesslich auf dem Trainings-
    fenster gefittet und auf den gesamten Zeitraum (Training + Testtag)
    angewendet. Das vermeidet zwei Probleme einer getrennten Berechnung
    (siehe SESSION_CHANGES.md, Abschnitt 7.4):

    - Look-ahead: ein separat auf dem isolierten Testtag gefitteter
      Hedge-Ratio wuerde Information aus dem gesamten Testtag -- auch aus
      zukuenftigen Minuten relativ zu jedem Zeitpunkt -- in die
      Spread-Definition einfliessen lassen.
    - Kuenstlich abgeschnittene rollierende Fenster (z. B. der 60-Minuten
      Z-Score): isoliert betrachtet haette der Testtag zu Beginn keinen
      Vorlauf, wodurch die ersten rsi/z/vol_window Minuten verworfen wuerden.
      Mit einem gemeinsamen Panel schliesst der Testtag nahtlos an das
      Trainingsfenster an.

    Parameters
    ----------
    log_prices : pd.DataFrame
        Wide-format Log-Preise (DatetimeIndex, Symbole als Spalten).
    pairs : list of tuple(str, str)
    train_days : pd.Index
        Handelstage des Trainingsfensters (ohne den Testtag).
    test_day : Timestamp-artig
        Der Testtag.
    rsi_window, z_window, vol_window : int

    Returns
    -------
    (panel_train, panel_test) : Tuple[pd.DataFrame, pd.DataFrame]
    """
    combined_days = train_days.append(pd.Index([test_day]))
    mask = log_prices.index.normalize().isin(combined_days)
    lp_combined = log_prices.loc[mask].copy()

    beta_fit_end = pd.Timestamp(train_days.max())

    panel = build_feature_panel(
        lp_combined, pairs, rsi_window, z_window, vol_window, beta_fit_end=beta_fit_end
    )

    train_mask = panel.index.normalize().isin(train_days)
    test_mask = panel.index.normalize() == pd.Timestamp(test_day)

    return panel.loc[train_mask], panel.loc[test_mask]