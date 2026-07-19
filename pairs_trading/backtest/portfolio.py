"""
pairs_trading.backtest.portfolio

Portfolio-Aggregation und Transaktionskosten für den Backtest.

Wichtiger Hinweis:
- Aggregiert einzelne Paar-Positionen (inkl. Positionsgrösse aus
  `pairs_trading.backtest.sizing` und Transaktionskosten je Trade) zu einer
  konsistenten Portfolio-Equity-Kurve, inkl. Kapitalbindung über mehrere
  gleichzeitig offene Paare hinweg.
- Modelliert realistische Transaktionskosten (Spread/Slippage, Kommission)
  in Abhängigkeit von Positionsgrösse und Anzahl Round-Trips.
"""

import pandas as pd


def transaction_cost_per_position(
    positions_df: pd.DataFrame,
    *,
    cost_bps_per_leg: float,
    legs_per_round_trip: int = 4,
) -> pd.Series:
    """
    Transaktionskosten je Position, als Fraktion des Kapitals (dieselbe
    Grössenordnung wie position_size * realized_return, der
    Portfolio-Beitrag einer Position).

        Kosten = position_size * legs_per_round_trip * (cost_bps_per_leg / 10000)

    Ein Pairs-Trade hat zwei Beine (Long + Short); Eröffnen und Schliessen
    sind je zwei Ausführungen -> 4 Ausführungen je Round-Trip (Default).

    cost_bps_per_leg ist bewusst eine Annahme (Geld-Brief-Spanne +
    Kommission je Ausführung) und -- anders als jeder andere Parameter in
    diesem Projekt -- nicht aus den vorhandenen Daten kalibrierbar: es
    liegen nur Minuten-Schlusskurse vor, keine Geld-/Brief-Kurse oder
    Orderbuchtiefe. Diese Einschränkung wird im Notebook offen benannt statt
    stillschweigend übergangen.

    Parameter:
    - positions_df: DataFrame mit Spalte 'position_size'.
    - cost_bps_per_leg: Kosten je Ausführung in Basispunkten (1 bp = 0.0001).
    - legs_per_round_trip: Anzahl Ausführungen je Position (Default 4).

    Rückgabe:
    - pd.Series (gleicher Index wie positions_df), Transaktionskosten je
      Position (positiver Wert = Kosten, wird vom Bruttoertrag abgezogen).
    """
    return positions_df["position_size"] * legs_per_round_trip * (cost_bps_per_leg / 10000)


def aggregate_daily_portfolio(
    positions_df: pd.DataFrame,
    *,
    day_col: str = "day",
    gross_col: str = "gross_contribution",
    net_col: str = "net_contribution",
) -> pd.DataFrame:
    """
    Aggregiert Positionen zu einer täglichen Portfolio-PnL-Serie und einer
    kumulierten Equity-Kurve.

    Kumulierung additiv (cumsum), nicht multiplikativ verzinst — dieselbe
    Konvention wie in 05_Modellierung.ipynb §5.6 ('daily["equity"] =
    daily["pnl"].cumsum()'): gross_contribution/net_contribution sind aus
    Log-Preis-Differenzen abgeleitet und damit näherungsweise additive
    Log-Renditen, kein einfacher (multiplikativ zu verzinsender) Return.

    Da jede Position innerhalb genau eines Handelstages beginnt und endet
    (siehe build_positions — keine Positionen über Nacht), ist die
    Zuordnung Position -> Tag eindeutig; es wird nicht über offene
    Positionen an Tagesgrenzen interpoliert.

    Parameter:
    - positions_df: DataFrame mit day_col, gross_col, net_col.

    Rückgabe:
    - DataFrame mit day, n_positions, gross_pnl, net_pnl, equity_gross,
      equity_net, drawdown_net (Drawdown relativ zum bisherigen Höchststand
      der Netto-Equity).
    """
    daily = (
        positions_df.groupby(day_col)
        .agg(
            n_positions=(net_col, "size"),
            gross_pnl=(gross_col, "sum"),
            net_pnl=(net_col, "sum"),
        )
        .reset_index()
        .sort_values(day_col)
        .reset_index(drop=True)
    )
    daily["equity_gross"] = daily["gross_pnl"].cumsum()
    daily["equity_net"] = daily["net_pnl"].cumsum()
    daily["drawdown_net"] = daily["equity_net"] - daily["equity_net"].cummax()
    return daily


def concurrent_exposure(
    positions_df: pd.DataFrame,
    *,
    day_col: str = "day",
    entry_col: str = "entry_time",
    exit_col: str = "exit_time",
    size_col: str = "position_size",
) -> pd.DataFrame:
    """
    Maximale gleichzeitige Kapitalbindung (Summe von position_size über alle
    zum selben Zeitpunkt offenen Positionen, über alle Paare hinweg) je
    Handelstag.

    Kelly-Positionsgrössen (6.3) wurden je Position unabhängig berechnet,
    ohne Rücksicht darauf, wie viele andere Positionen gleichzeitig offen
    sind (bewusste Vereinfachung, siehe 6.3.6 — kein portfolio-optimales
    Multi-Asset-Kelly). Diese Funktion macht messbar, ob diese Vereinfachung
    unrealistische Kapitalanforderungen erzeugt (Summe der gleichzeitig
    offenen position_size > 1.0 = mehr als 100% des Kapitals gleichzeitig
    gebunden).

    Da jede Position innerhalb genau eines Handelstages beginnt und endet,
    wird pro Tag unabhängig ausgewertet (Ereignis-Zeitreihe: +position_size
    bei Eröffnung, -position_size bei Schliessung, kumulativ aufsummiert).

    Wichtig bei Ereignissen zur exakt gleichen Minute: eine Schliessung wird
    IMMER vor einer Eröffnung zur selben Minute verarbeitet (Sortierung nach
    ['t', '_order'], Schliessungen mit _order=0, Eröffnungen mit _order=1) —
    eine zur Minute T geschlossene Position gibt ihre Kapazität rechtzeitig
    für eine zur selben Minute T eröffnete neue Position frei. Ohne diese
    explizite Tie-Break-Regel hängt die Reihenfolge gleichzeitiger Ereignisse
    von der (nicht garantiert stabilen) Sortierung allein nach 't' ab, was zu
    einer künstlich überhöhten, nie tatsächlich gleichzeitig bestehenden
    Exponierung führen kann (in einer früheren Version dieser Funktion
    tatsächlich beobachtet — siehe SESSION_CHANGES.md).

    Rückgabe:
    - DataFrame mit day, max_exposure (höchste innerhalb des Tages
      beobachtete Summe gleichzeitig offener position_size).
    """
    entries = (
        positions_df[[day_col, entry_col, size_col]]
        .rename(columns={entry_col: "t"})
        .assign(delta=positions_df[size_col], _order=1)
    )
    exits = (
        positions_df[[day_col, exit_col, size_col]]
        .rename(columns={exit_col: "t"})
        .assign(delta=-positions_df[size_col], _order=0)
    )
    events = pd.concat([entries, exits])

    rows = []
    for day, g in events.groupby(day_col):
        g = g.sort_values(["t", "_order"])
        running = g["delta"].cumsum()
        rows.append({day_col: day, "max_exposure": running.max()})

    return pd.DataFrame(rows).sort_values(day_col).reset_index(drop=True)
