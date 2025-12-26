from pathlib import Path
import pandas as pd


def fetch_data(data_dir: str = "Daten", pattern: str = "sp500_1min_*.csv") -> pd.DataFrame:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Keine Dateien gefunden in '{data_dir}' mit Pattern '{pattern}'"
        )

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
    print(f"Anzahl Ticker gesamt: {df['symbol'].nunique()}")
    print(f"Anzahl Handelstage: {df['date'].nunique()}")

    return df
