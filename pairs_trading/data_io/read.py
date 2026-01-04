from pathlib import Path
import json
import pandas as pd

from .paths import (
    as_path,
    unprocessed_files,
    parquet_path,
    metadata_path,
)


def _show_head(df: pd.DataFrame, n: int = 5) -> None:
    try:
        display(df.head(n))
    except NameError:
        print(df.head(n))


def load_unprocessed_data(
    data_dir: str = "Daten",
    pattern: str = "sp500_1min_*.csv",
) -> pd.DataFrame:
    data_dir_p = as_path(data_dir)
    files = unprocessed_files(data_dir_p, pattern)

    if not files:
        print("Keine unprocessed Daten gefunden.")
        print(f"Ordner: {data_dir_p} | Pattern: {pattern}")
        print('Bitte zuerst das Notebook "02_Datensammlung.ipynb" ausführen.')
        return pd.DataFrame()

    dfs = []
    for file in files:
        df_day = pd.read_csv(file, parse_dates=["datetime"])
        df_day["date"] = df_day["datetime"].dt.date
        dfs.append(df_day)

    df = (
        pd.concat(dfs, ignore_index=True)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"Anzahl Zeilen gesamt: {len(df):,}")
    if "symbol" in df.columns:
        print(f"Anzahl Ticker gesamt: {df['symbol'].nunique()}")
    if "date" in df.columns:
        print(f"Anzahl Handelstage: {df['date'].nunique()}")

    _show_head(df)
    return df


def load_preprocessed_data(
    data_dir: str = "Daten",
    basename: str = "prices_long_cleaned",
):
    data_dir_p = as_path(data_dir)

    parquet_fp = parquet_path(data_dir_p, basename)
    meta_fp = metadata_path(data_dir_p, basename)

    if (not parquet_fp.exists()) or (not meta_fp.exists()):
        print("Keine preprocessed Daten gefunden.")
        print(f"Erwartet: {parquet_fp.name} und {meta_fp.name} in {data_dir_p}")
        print('Bitte zuerst das Notebook "03_Preprocessing.ipynb" ausführen.')
        return pd.DataFrame(), {}

    df = pd.read_parquet(parquet_fp)
    meta = json.loads(meta_fp.read_text(encoding="utf-8"))

    required_cols = {"datetime", "symbol"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame fehlt/n Spalte/n: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")

    print("OK: Daten geladen via io.load_preprocessed_data")
    print("Rows:", len(df), "| Symbols:", df["symbol"].nunique())
    print("Meta keys:", list(meta.keys()))

    _show_head(df)
    return df, meta
