"""
pairs_trading.preprocessing.data_cleaning

Bereinigungs- und Transformationsfunktionen für das Preprocessing.

Wichtiger Hinweis:
- Print-Ausgaben (falls vorhanden) und Rückgaben sind Teil des beobachtbaren Verhaltens.
- Refactorings dürfen Funktionalität nicht verändern.
- Dieses Modul ist bewusst "funktional" gehalten (DataFrame rein → DataFrame raus).
"""

import numpy as np
import pandas as pd


# ============================================================================
# Coverage / universe filtering
# ============================================================================

def filter_symbols_by_full_coverage(
    df: pd.DataFrame,
    time_col: str = "datetime",
    symbol_col: str = "symbol",
    return_removed: bool = True,
):
    """
    Filtert Symbole, die über den gesamten Zeitraum des Datensatzes präsent sind.

    Ein Symbol gilt als vollständig, wenn es:
      - am globalen Startzeitpunkt vorkommt
      - am globalen Endzeitpunkt vorkommt

    Entfernt damit z.B. neu aufgenommene oder ausgeschlossene Index-Mitglieder.

    Returns:
      - gefiltertes DataFrame
      - entfernte Symbole (optional)
    """
    if time_col not in df.columns:
        raise KeyError(f"Spalte '{time_col}' fehlt im DataFrame")
    if symbol_col not in df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt im DataFrame")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    start = df[time_col].min()
    end = df[time_col].max()

    symbols_start = set(df.loc[df[time_col] == start, symbol_col].unique())
    symbols_end = set(df.loc[df[time_col] == end, symbol_col].unique())

    valid_symbols = symbols_start & symbols_end
    all_symbols = set(df[symbol_col].unique())

    removed_symbols = sorted(all_symbols - valid_symbols)

    filtered_df = df[df[symbol_col].isin(valid_symbols)].reset_index(drop=True)

    if return_removed:
        return filtered_df, pd.Index(removed_symbols, name=symbol_col)

    return filtered_df


def find_symbols_with_long_gaps(
    nan_gaps_all: pd.DataFrame,
    min_gap_len: int = 6,
    symbol_col: str = "symbol",
    gap_len_col: str = "gap_len",
) -> pd.Index:
    """
    Liefert Symbole, deren maximale Gap-Länge >= min_gap_len ist.

    Erwartet nan_gaps_all mit Spalten: symbol_col, gap_len_col
    """
    if nan_gaps_all is None or nan_gaps_all.empty:
        return pd.Index([], name=symbol_col)

    missing = {symbol_col, gap_len_col} - set(nan_gaps_all.columns)
    if missing:
        raise KeyError(f"nan_gaps_all fehlt Spalten: {sorted(missing)}")

    max_gap = nan_gaps_all.groupby(symbol_col)[gap_len_col].max()
    return pd.Index(max_gap[max_gap >= min_gap_len].index, name=symbol_col)


def drop_symbols(
    df: pd.DataFrame,
    symbols_to_drop,
    symbol_col: str = "symbol",
    sort_by=None,
    reset_index: bool = True,
):
    """
    Entfernt alle Zeilen mit Symbolen in symbols_to_drop.
    Optional: sort_by, reset_index.

    Returns:
      - df_new
    """
    if symbol_col not in df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt im DataFrame")

    symbols_to_drop = pd.Index(symbols_to_drop).astype(str)
    df_new = df.loc[~df[symbol_col].isin(symbols_to_drop)].copy()

    if sort_by is not None:
        df_new = df_new.sort_values(list(sort_by)).copy()
    if reset_index:
        df_new = df_new.reset_index(drop=True)

    return df_new


# ============================================================================
# Time handling, sorting, duplicates
# ============================================================================

def ensure_datetime_and_sort(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    time_col: str = "datetime",
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    Konvertiert time_col nach datetime und sortiert nach (symbol_col, time_col).
    """
    if symbol_col not in df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt im DataFrame")
    if time_col not in df.columns:
        raise KeyError(f"Spalte '{time_col}' fehlt im DataFrame")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors=errors)
    df = df.sort_values([symbol_col, time_col]).copy()
    return df


def drop_duplicate_rows(
    df,
    subset=None,
    keep="first",
    return_duplicates=True,
    sort_by=None,
    reset_index=True,
):
    """
    Findet und entfernt doppelte Zeilen.

    Standardmäßig werden Duplikate anhand von (datetime, symbol) entfernt.

    Returns:
      - df_clean
      - duplicates_removed (optional, wenn return_duplicates=True)
    """
    if subset is None:
        subset = ["datetime", "symbol"]

    missing = set(subset) - set(df.columns)
    if missing:
        raise KeyError(f"df fehlt Spalten für subset: {sorted(missing)}")

    df_work = df.copy()

    if sort_by is not None:
        df_work = df_work.sort_values(list(sort_by)).copy()

    dup_mask = df_work.duplicated(subset=subset, keep=keep)

    duplicates_removed = df_work.loc[dup_mask].copy()
    df_clean = df_work.loc[~dup_mask].copy()

    if reset_index:
        df_clean = df_clean.reset_index(drop=True)
        duplicates_removed = duplicates_removed.reset_index(drop=True)

    if return_duplicates:
        return df_clean, duplicates_removed

    return df_clean


# ============================================================================
# Preis-Ausreisser ("Bad Ticks")
# ============================================================================

def flag_price_spikes(
    df: pd.DataFrame,
    price_col: str = "close",
    symbol_col: str = "symbol",
    time_col: str = "datetime",
    threshold: float = 0.5,
    snapback_frac: float = 0.3,
) -> pd.Series:
    """
    Markiert einzelne Bar-Ausreisser ("Bad Ticks") je Symbol.

    Ein Punkt gilt als Ausreisser, wenn der log-Return zur Vorgängerzeile
    betragsmässig über 'threshold' liegt UND der log-Return zur Folgezeile
    das Gegenvorzeichen hat und den Ausschlag zu mindestens (1 - snapback_frac)
    wieder rückgängig macht (V-förmiger Snap-back).

    Das unterscheidet einzelne Fehlmessungen (z. B. ein fehlerhafter
    "Official Close"-Print) von echten, dauerhaften Kurssprüngen
    (z. B. ein Übernacht-Gap durch Earnings), die absichtlich NICHT
    markiert werden sollen.

    Randfall: Für die letzte Beobachtung je Symbol existiert keine Folgezeile,
    der Snap-back kann also nicht geprüft werden. Dort wird stattdessen
    zusätzlich verlangt, dass der Wert auch vom Median des jeweiligen Tages
    stark abweicht -- das erkennt z. B. einen fehlerhaften "Official Close"
    am letzten Handelstag des gesamten Datensatzes, für den es keine
    Folgeminute mehr gibt.

    Returns:
      - boolean Series (gleicher Index wie df), True = Ausreisser
    """
    missing = {symbol_col, time_col, price_col} - set(df.columns)
    if missing:
        raise KeyError(f"df fehlt Spalten: {sorted(missing)}")

    df_sorted = df.sort_values([symbol_col, time_col])
    log_p = np.log(df_sorted[price_col])
    g = df_sorted[symbol_col]

    ret_in = log_p.groupby(g).diff()
    ret_out = log_p.groupby(g).diff().shift(-1)

    is_snapback_spike = (
        (ret_in.abs() > threshold)
        & (np.sign(ret_in) == -np.sign(ret_out))
        & ((ret_in + ret_out).abs() < snapback_frac * ret_in.abs())
    )

    day = df_sorted[time_col].dt.date
    daily_median_log_p = log_p.groupby([g, day]).transform("median")
    dev_from_day_median = (log_p - daily_median_log_p).abs()
    is_last_per_symbol = ret_out.isna() & log_p.notna()
    is_boundary_spike = (
        is_last_per_symbol
        & (ret_in.abs() > threshold)
        & (dev_from_day_median > threshold)
    )

    is_spike = is_snapback_spike | is_boundary_spike

    return is_spike.reindex(df.index).fillna(False)


# ============================================================================
# Preisniveau-Brüche (Aktiensplits / Reverse-Splits)
# ============================================================================

def detect_price_level_breaks(
    df: pd.DataFrame,
    price_col: str = "close",
    symbol_col: str = "symbol",
    time_col: str = "datetime",
    lower: float = 0.6,
    upper: float = 1.6,
) -> pd.DataFrame:
    """
    Findet permanente Preisniveau-Brüche je Symbol (typischerweise
    Aktiensplits oder Reverse-Splits), die in den Rohdaten nicht bereinigt
    wurden.

    Berechnet den täglichen Median-Preis je Symbol und markiert Tage, an
    denen das Verhältnis zum Vortag ausserhalb von [lower, upper] liegt. Im
    Gegensatz zu flag_price_spikes() wird hier NICHT auf einen Snap-back
    geprüft -- der Bruch soll ja gerade dauerhaft sein, nicht isoliert.

    Returns:
      - DataFrame mit Spalten [symbol_col, 'date', 'ratio'], eine Zeile pro
        erkanntem Bruch
    """
    missing = {symbol_col, time_col, price_col} - set(df.columns)
    if missing:
        raise KeyError(f"df fehlt Spalten: {sorted(missing)}")

    tmp = df[[symbol_col, time_col, price_col]].copy()
    tmp["date"] = tmp[time_col].dt.date

    daily = tmp.groupby([symbol_col, "date"])[price_col].median().reset_index()
    daily = daily.sort_values([symbol_col, "date"])
    daily["ratio"] = daily.groupby(symbol_col)[price_col].pct_change() + 1

    breaks = daily[(daily["ratio"] < lower) | (daily["ratio"] > upper)].copy()
    return breaks[[symbol_col, "date", "ratio"]].reset_index(drop=True)


def adjust_for_price_level_breaks(
    df: pd.DataFrame,
    breaks: pd.DataFrame,
    price_cols=("close",),
    symbol_col: str = "symbol",
    time_col: str = "datetime",
) -> pd.DataFrame:
    """
    Passt historische Preise rückwirkend an einen erkannten Preisniveau-Bruch
    an (Split-Adjustierung), damit die Zeitreihe wieder durchgängig
    vergleichbar ist.

    Für jede Zeile in 'breaks' (symbol_col, 'date', 'ratio') werden alle
    Preise des betroffenen Symbols VOR 'date' mit 'ratio' multipliziert.
    Mehrere Brüche pro Symbol werden in chronologischer Reihenfolge
    angewendet, sodass sich die Anpassungen korrekt kumulieren.

    Returns:
      - df (Kopie) mit angepassten price_cols
    """
    missing = {symbol_col, time_col} - set(df.columns)
    if missing:
        raise KeyError(f"df fehlt Spalten: {sorted(missing)}")

    df = df.copy()
    breaks = breaks.sort_values([symbol_col, "date"])

    for _, row in breaks.iterrows():
        sym, brk_date, ratio = row[symbol_col], row["date"], row["ratio"]
        mask = (df[symbol_col] == sym) & (df[time_col].dt.date < brk_date)
        for col in price_cols:
            df.loc[mask, col] = df.loc[mask, col] * ratio

    return df


# ============================================================================
# Gap filling / interpolation
# ============================================================================

def fill_close_gaps_per_symbol(
    df: pd.DataFrame,
    value_col: str = "close",
    out_col: str = "close_filled",
    symbol_col: str = "symbol",
    time_col: str = "datetime",
    ensure_sorted: bool = True,
) -> pd.DataFrame:
    """
    Interpoliert fehlende Close-Werte pro Symbol zeitbasiert und füllt Ränder:
      1) interpolate(method="time", limit_area="inside")
      2) bfill() / ffill() für Randbereiche

    Returns:
      - df mit neuer Spalte out_col
    """
    missing = {symbol_col, time_col, value_col} - set(df.columns)
    if missing:
        raise KeyError(f"df fehlt Spalten: {sorted(missing)}")

    df_work = ensure_datetime_and_sort(df, symbol_col=symbol_col, time_col=time_col) if ensure_sorted else df.copy()

    def _fill_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).set_index(time_col)
        s = g[value_col]

        s_interp = s.interpolate(method="time", limit_area="inside")
        s_filled = s_interp.bfill().ffill()

        g[out_col] = s_filled
        return g.reset_index()

    return (
        df_work.groupby(symbol_col, group_keys=True)
        .apply(_fill_group)
        .reset_index(level=0)
        .reset_index(drop=True)
    )


# ============================================================================
# Length diagnostics / reporting
# ============================================================================

def symbol_lengths(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.Series:
    """
    Anzahl Zeilen pro Symbol (aufsteigend sortiert).
    """
    if symbol_col not in df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt im DataFrame")
    return df.groupby(symbol_col).size().sort_values()


def expected_length_mode(lengths: pd.Series) -> int:
    """
    Häufigste Länge als Referenz (Mode).
    """
    if lengths is None or lengths.empty:
        raise ValueError("lengths ist leer")
    return int(lengths.mode().iloc[0])


def symbols_with_unexpected_length(
    lengths: pd.Series,
    expected_length: int,
    tolerance: float = 0.0,
) -> pd.Series:
    """
    Liefert Series der Symbole, deren Länge um mehr als 'tolerance'
    (relativer Anteil von expected_length) von expected_length abweicht.

    tolerance=0.0 (Default) entspricht einem exakten Abgleich
    (lengths != expected_length).
    """
    lower = expected_length * (1 - tolerance)
    upper = expected_length * (1 + tolerance)
    return lengths[(lengths < lower) | (lengths > upper)]


def removed_symbols_between(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    symbol_col: str = "symbol",
) -> pd.Index:
    """
    Liefert Symbole, die in df_before vorkommen, aber nicht mehr in df_after.
    """
    if symbol_col not in df_before.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt in df_before")
    if symbol_col not in df_after.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt in df_after")

    before = pd.Index(df_before[symbol_col].dropna().unique()).astype(str)
    after = pd.Index(df_after[symbol_col].dropna().unique()).astype(str)
    return before.difference(after)
