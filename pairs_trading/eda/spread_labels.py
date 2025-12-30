
import numpy as np
import pandas as pd
from IPython.display import display
from pairs_trading.features.technical import (
    sma,
    ema,
    rsi,
    macd,
)
from typing import Optional, List, Dict


import numpy as np
import pandas as pd

from pairs_trading.eda.data_loading import fetch_data
from pairs_trading.eda.correlation import correlation_analysis

def get_log_prices(
    data_dir: str = "Daten",
    pattern: str = "sp500_1min_*.csv",
) -> pd.DataFrame:
    """
    Lade Intraday-Daten des letzten Handelstags und berechne log-Preise.

    Rueckgabe:
        DataFrame mit log-Preisen (Wide-Format: Index = datetime, Spalten = Symbole)
    """

    # ---------------------------
    # 1) Rohdaten laden
    # ---------------------------
    df_raw = fetch_data(data_dir=data_dir, pattern=pattern)

    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], errors="coerce")
    df_raw = df_raw.dropna(subset=["datetime"])
    df_raw = df_raw.sort_values("datetime")

    # ---------------------------
    # 2) Nur letzter Handelstag
    # ---------------------------
    last_day = df_raw["datetime"].dt.normalize().max()
    df_day = df_raw[df_raw["datetime"].dt.normalize() == last_day]

    if df_day.empty:
        raise ValueError("Keine Daten fuer den letzten Handelstag gefunden.")

    # ---------------------------
    # 3) Wide-Format + log prices
    # ---------------------------
    prices_wide = df_day.pivot_table(
        index="datetime",
        columns="symbol",
        values="close"
    )

    log_prices = np.log(prices_wide)

    return log_prices

def get_top_pairs_today(
    data_dir: str = "Daten",
    pattern: str = "sp500_1min_*.csv",
    top_n: int = 5,
    method: str = "pearson",
    strong_threshold: float = 0.97,
) -> pd.DataFrame:
    """
    Ermittelt die Top-N Aktienpaare basierend auf Intraday-Daten
    des letzten Handelstags.

    Rueckgabe:
        DataFrame mit Spalten:
            - symbol1
            - symbol2
            - (optional) corr
    """

    # ---------------------------
    # 1) Rohdaten laden
    # ---------------------------
    df_raw = fetch_data(data_dir=data_dir, pattern=pattern)

    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], errors="coerce")
    df_raw = df_raw.dropna(subset=["datetime"])
    df_raw = df_raw.sort_values("datetime")

    # ---------------------------
    # 2) Nur letzter Handelstag
    # ---------------------------
    last_day = df_raw["datetime"].dt.normalize().max()
    df_day = df_raw[df_raw["datetime"].dt.normalize() == last_day]

    if df_day.empty:
        raise ValueError("Keine Daten fuer den letzten Handelstag gefunden.")

    # ---------------------------
    # 3) Wide-Format + log prices
    # ---------------------------
    prices_wide = df_day.pivot_table(
        index="datetime",
        columns="symbol",
        values="close"
    )

    log_prices = np.log(prices_wide)

    # ---------------------------
    # 4) Korrelationsanalyse
    # ---------------------------
    corr_out = correlation_analysis(
        prices_wide=log_prices,
        method=method,
        strong_threshold=strong_threshold,
        plot=False
    )

    strong_pairs = corr_out.get("strong_pairs")
    if strong_pairs is None or strong_pairs.empty:
        raise ValueError("Keine starken Paare fuer heute gefunden.")

    # ---------------------------
    # 5) Normiertes Rueckgabeformat
    # ---------------------------
    top_pairs = (
        strong_pairs
        .rename(columns={"symbol_1": "symbol1", "symbol_2": "symbol2"})
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_pairs
print(get_top_pairs_today())




def classify_top_pair_spread(
    pair_rank: int = 0,
    lookahead: int = 2,
    min_obs: int = 60,
    feature_specs: Optional[List[Dict]] = None
) -> pd.DataFrame:

    """
    Klassifiziert die zukuenftige Spread-Bewegung fuer EIN Top-Paar
    aus einer Ranking-Tabelle (top_pairs).

    Klassen (target_4c):
        0 = Spread schliesst sich, Aktie symbol1 dominiert
        1 = Spread schliesst sich, Aktie symbol2 dominiert
        2 = Spread vergroessert sich, Aktie symbol1 dominiert
        3 = Spread vergroessert sich, Aktie symbol2 dominiert

    Parameter:
        pair_rank : Rang des Paars in top_pairs (0 = bestes Paar)
        lookahead : Zeitschritte in die Zukunft
        min_obs   : Mindestanzahl Intraday-Punkte
        feature_specs : Liste von Feature-Spezifikationen fuer Spread-Features.
            Beispiel:
            [
                {"name": "SMA", "params": {"window": 20}},
                {"name": "RSI", "params": {"window": 14}},
                {"name": "MACD_HIST", "params": {"fast": 12, "slow": 26, "signal": 9}},
            ]
            Erlaubte Feature-Namen: "SMA", "EMA", "RSI", "MACD_HIST"

    Rueckgabe:
        DataFrame mit:
            - spread
            - future_spread
            - spread_change
            - spread_change_pct
            - dominant_stock
            - target_4c
    """

    # ---------------------------
    # Paar aus Ranking auswaehlen
    # ---------------------------
    top_pairs= get_top_pairs_today()
    try:
        s1 = top_pairs.iloc[pair_rank]["symbol1"]
        s2 = top_pairs.iloc[pair_rank]["symbol2"]
    except Exception:
        raise ValueError("pair_rank ist ungueltig oder top_pairs leer.")

    # ---------------------------
    # Intraday-Daten (letzter Handelstag)
    # ---------------------------
    log_prices = get_log_prices()
    last_day = log_prices.index.normalize().max()
    day_mask = log_prices.index.normalize() == last_day
    prices = log_prices.loc[day_mask, [s1, s2]].dropna()

    if prices.empty or len(prices) < min_obs:
        raise ValueError("Nicht genuegend Intraday-Daten fuer das Paar.")

    # ---------------------------
    # Spread via OLS
    # ---------------------------
    beta, _ = np.polyfit(prices[s2].values, prices[s1].values, 1)
    spread = prices[s1] - beta * prices[s2]

    df = pd.DataFrame(index=spread.index)
    df["spread"] = spread

    # ---------------------------
    # FEATURES AUF SPREAD (t)
    # ---------------------------
    if feature_specs:
        spread_df = df["spread"].to_frame("SPREAD")

        for spec in feature_specs:
            name = spec["name"]
            params = spec.get("params", {})

            if name == "SMA":
                df[f"SMA_{params['window']}"] = sma(spread_df, **params)["SPREAD"]

            elif name == "EMA":
                df[f"EMA_{params['span']}"] = ema(spread_df, **params)["SPREAD"]

            elif name == "RSI":
                df[f"RSI_{params['window']}"] = rsi(spread_df, **params)["SPREAD"]

            elif name == "MACD_HIST":
                _, _, hist = macd(spread_df, **params)
                p = params
                df[f"MACD_HIST_{p['fast']}_{p['slow']}_{p['signal']}"] = hist["SPREAD"]

            else:
                raise ValueError(f"Unbekanntes Feature: {name}")


    # ---------------------------
    # Zukunftsbewegung
    # ---------------------------
    df["future_spread"] = df["spread"].shift(-lookahead)
    df["spread_change"] = df["future_spread"] - df["spread"]

    denom = df["spread"].abs().replace(0, np.nan)
    df["spread_change_pct"] = df["spread_change"] / denom * 100

    # ---------------------------
    # Relative Aktienbewegung
    # ---------------------------
    ret_s1 = prices[s1].shift(-lookahead) - prices[s1]
    ret_s2 = prices[s2].shift(-lookahead) - prices[s2]

    df["dominant_stock"] = np.where(ret_s1 > ret_s2, s1, s2)

    # ---------------------------
    # Spread schliesst / oeffnet
    # ---------------------------
    spread_closes = df["spread_change"] * df["spread"] < 0
    spread_opens = df["spread_change"] * df["spread"] > 0

    df["target_4c"] = np.select(
        [
            spread_closes & (df["dominant_stock"] == s1),
            spread_closes & (df["dominant_stock"] == s2),
            spread_opens & (df["dominant_stock"] == s1),
            spread_opens & (df["dominant_stock"] == s2),
        ],
        [0, 1, 2, 3],
        default=np.nan,
    )

    # ---------------------------
    # Cleanup
    # ---------------------------
    df = df.dropna(subset=["future_spread", "target_4c"])
    df["target_4c"] = df["target_4c"].astype(int)

    
    return df