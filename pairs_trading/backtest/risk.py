"""
pairs_trading.backtest.risk

Risikomanagement auf Portfolioebene für den Backtest.

Wichtiger Hinweis:
- Definiert Risiko-Limits auf Portfolioebene (z. B. maximale gleichzeitige
  Exponierung, Stop-Loss- oder Max-Drawdown-Regeln je Paar/Portfolio), um
  das in der Trading-Inspektion (05_Modellierung.ipynb, 5.6) beobachtete
  unkontrollierte Verhalten (Drawdown ohne jegliche Risikosteuerung)
  einzugrenzen.
"""

import pandas as pd


def cap_concurrent_exposure(
    positions_df: pd.DataFrame,
    *,
    max_exposure: float = 1.0,
    day_col: str = "day",
    entry_col: str = "entry_time",
    exit_col: str = "exit_time",
    size_col: str = "position_size",
) -> pd.Series:
    """
    Begrenzt die gleichzeitige Gesamtexponierung über alle offenen Positionen
    hinweg auf max_exposure (Vielfaches des Kapitals, Default 1.0 = 100%) —
    die konkrete Antwort auf den in 06_Backtest.ipynb §6.5.2 gemessenen
    Befund (Exponierung > 100% des Kapitals an 40% der Handelstage).

    Funktionsweise (kausal, keine rückwirkende Anpassung bereits offener
    Positionen): Positionen werden chronologisch nach Eröffnungszeitpunkt
    verarbeitet. Beim Eröffnen einer neuen Position wird geprüft, wie viel
    Kapazität (max_exposure minus die durch bereits offene Positionen
    gebundene Exponierung) noch verfügbar ist; die neue Position wird auf
    diese verfügbare Kapazität begrenzt (ggf. auf 0, wenn keine Kapazität
    mehr frei ist). Eine bereits eröffnete Position wird durch später
    eintreffende Positionen NICHT nachträglich verkleinert — nur neue
    Positionen werden an die aktuelle Kapazität angepasst.

    Da jede Position innerhalb genau eines Handelstages beginnt und endet
    (siehe build_positions), wird pro Tag unabhängig verarbeitet.

    Parameter:
    - positions_df: DataFrame mit day_col, entry_col, exit_col, size_col.
    - max_exposure: maximal zulässige Summe gleichzeitig offener
      position_size (Vielfaches des Kapitals).

    Rückgabe:
    - pd.Series (gleicher Index wie positions_df): die ggf. reduzierte
      position_size je Position.
    """
    capped = pd.Series(index=positions_df.index, dtype=float)

    for _, day_positions in positions_df.groupby(day_col):
        order = day_positions.sort_values(entry_col).index
        current_exposure = 0.0
        open_positions = []  # Liste von (exit_time, zugewiesene Grösse, Index)

        for idx in order:
            entry_t = day_positions.loc[idx, entry_col]
            exit_t = day_positions.loc[idx, exit_col]
            original_size = day_positions.loc[idx, size_col]

            # Bereits geschlossene Positionen freigeben, bevor die Kapazität
            # für die neue Position geprüft wird.
            still_open = []
            for (o_exit, o_size, o_idx) in open_positions:
                if o_exit <= entry_t:
                    current_exposure -= o_size
                else:
                    still_open.append((o_exit, o_size, o_idx))
            open_positions = still_open

            available = max(max_exposure - current_exposure, 0.0)
            new_size = min(original_size, available)

            capped.loc[idx] = new_size
            current_exposure += new_size
            open_positions.append((exit_t, new_size, idx))

    return capped
