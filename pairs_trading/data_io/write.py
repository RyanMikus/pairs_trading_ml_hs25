from pathlib import Path
import json
import pandas as pd

from .paths import (
    as_path,
    parquet_path,
    metadata_path,
)


def _show_head(df: pd.DataFrame, n: int = 5) -> None:
    try:
        display(df.head(n))
    except NameError:
        print(df.head(n))


def save_preprocessed_data(
    df_final: pd.DataFrame,
    output_dir: str = "Daten",
    basename: str = "prices_long_cleaned",
) -> None:
    if not isinstance(df_final, pd.DataFrame):
        raise TypeError("df_final must be a pandas DataFrame")

    for col in ("datetime", "symbol"):
        if col not in df_final.columns:
            raise ValueError(f"df_final missing required column: '{col}'")

    if df_final.empty:
        print("Warnung: df_final ist leer – es werden leere Dateien geschrieben.")

    df_out = df_final.copy()
    df_out["datetime"] = pd.to_datetime(df_out["datetime"], errors="raise")

    output_dir_p = as_path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    parquet_fp = parquet_path(output_dir_p, basename)
    meta_fp = metadata_path(output_dir_p, basename)

    meta = {
        "n_rows": int(len(df_out)),
        "n_symbols": int(df_out["symbol"].nunique()),
        "min_datetime": str(df_out["datetime"].min()),
        "max_datetime": str(df_out["datetime"].max()),
        "columns": list(df_out.columns),
        "dtypes": {c: str(t) for c, t in df_out.dtypes.items()},
        "format": "parquet",
        "engine": "pyarrow",
    }

    df_out.to_parquet(
        parquet_fp,
        engine="pyarrow",
        index=False,
    )

    meta_fp.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("Preprocessed Daten gespeichert")
    print(f"Parquet : {parquet_fp}")
    print(f"Metadata: {meta_fp}")
    print(f"Rows: {len(df_out)} | Symbols: {df_out['symbol'].nunique()}")

    _show_head(df_out)
