import pandas as pd
from typing import Optional, List

from .paths import unprocessed_files, parquet_path


def _print_df_overview(df: pd.DataFrame, title: str) -> None:
    print(f"\n[{title}]")
    print("Shape:", df.shape)
    print("Spalten:", list(df.columns))
    print("\nHead:")
    print(df.head())


def load_unprocessed_data(
    *,
    data_dir: str,
    pattern: str,
    parse_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    files = unprocessed_files(data_dir, pattern)
    if not files:
        raise FileNotFoundError(f"Keine Dateien gefunden für Pattern '{pattern}'")

    print(f"\n[Unprocessed Data Load]")
    print("Verzeichnis:", data_dir)
    print("Pattern:", pattern)
    print("Gefundene Dateien:", len(files))

    dfs = [
        pd.read_csv(fp, parse_dates=parse_dates)
        for fp in files
    ]
    df = pd.concat(dfs, ignore_index=True)

    _print_df_overview(df, "Unprocessed Data")

    return df


def load_preprocessed_data(
    *,
    data_dir: str,
    basename: str,
) -> pd.DataFrame:
    fp = parquet_path(data_dir, basename)

    print(f"\n[Preprocessed Data Load]")
    print("Pfad:", fp)

    df = pd.read_parquet(fp)

    _print_df_overview(df, "Preprocessed Data")

    return df
