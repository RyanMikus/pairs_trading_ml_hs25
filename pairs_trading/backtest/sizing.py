"""
pairs_trading.backtest.sizing

Positionsgrössen-Logik (Position Sizing) für den Backtest.

Wichtiger Hinweis:
- Legt fest, wie aus Modell-Wahrscheinlichkeit (proba) und/oder Signalstärke
  (Z-Score) eine tatsächliche Positionsgrösse abgeleitet wird (z. B.
  fixed-fractional, volatilitätsskaliert, Kelly-basiert).
- Umfasst auch die Positions-Design-Entscheidung, wie mit mehreren
  gleichzeitigen Signalen im selben Paar umgegangen wird (überlappende
  Positionen vs. Ignorieren neuer Signale bei bereits offener Position) und
  wann eine Position wieder geschlossen wird. Diese Entscheidungen sind als
  explizite Zustandsmaschine (PositionState) in build_positions() umgesetzt,
  siehe dort.
"""

from enum import Enum

import numpy as np
import pandas as pd


class PositionState(Enum):
    """
    Zustände der Positions-Zustandsmaschine (pro Paar, pro Handelstag).

    FLAT        - keine offene Position.
    IN_POSITION - eine offene Position läuft.
    """

    FLAT = "FLAT"
    IN_POSITION = "IN_POSITION"


def _run_position_state_machine(
    datetimes,
    probas,
    zs,
    spreads,
    entry_signal,
    cap_minutes,
):
    """
    Explizite Zustandsmaschine für EIN (pair, day) — läuft chronologisch
    Minute für Minute.

    Zustände: siehe PositionState.

    Übergänge:
    - FLAT -> IN_POSITION:
        sobald 'entry_signal' erstmals True ist (Einstieg; 'entry_signal' ist
        proba >= threshold UND genug Restlaufzeit bis Handelsschluss, siehe
        build_positions). Die Richtung wird beim Einstieg einmalig festgelegt:
        direction = -1, falls entry_z > 0 (Wette auf fallenden Spread),
        sonst +1.
    - IN_POSITION -> FLAT, bei der jeweils ZUERST eintretenden Bedingung:
        (a) 'z' erreicht/kreuzt die Null-Linie in Richtung von 'direction'
            (vollständige Reversion — das eigentliche Handelsziel, nicht nur
            die fortlaufende Modell-Konfidenz), oder
        (b) seit Einstieg sind 'cap_minutes' Minuten vergangen (Kappung der
            Haltedauer; None = keine Kappung), oder
        (c) das Ende des Handelstages ist erreicht (erzwungener Ausstieg —
            Positionen werden nie über Nacht gehalten).
        Der Exit wird in jedem Fall an der Minute festgemacht, in der die
        Bedingung beobachtet wird — nie rückwirkend an einer früheren Minute
        (kausal exakt umsetzbar, kein Hindsight).

    Rückgabe:
    - Liste von dicts, je eine geschlossene Position.
    """
    n = len(zs)
    positions = []

    state = PositionState.FLAT
    entry_idx = None
    direction = None

    for i in range(n):
        if state is PositionState.FLAT:
            if entry_signal[i]:
                state = PositionState.IN_POSITION
                entry_idx = i
                direction = -1.0 if zs[i] > 0 else 1.0

        elif state is PositionState.IN_POSITION:
            reverted = (zs[i] <= 0) if direction == -1.0 else (zs[i] >= 0)
            elapsed_min = (datetimes[i] - datetimes[entry_idx]) / np.timedelta64(1, "m")
            capped = cap_minutes is not None and elapsed_min >= cap_minutes

            if reverted or capped:
                exit_reason = "z_cross" if reverted else "cap"
                positions.append(
                    _close_position(datetimes, probas, zs, spreads, entry_idx, i, exit_reason)
                )
                state = PositionState.FLAT
                entry_idx = None
                direction = None

    # Tagesende erreicht, aber noch IN_POSITION -> erzwungener Ausstieg.
    if state is PositionState.IN_POSITION:
        positions.append(
            _close_position(datetimes, probas, zs, spreads, entry_idx, n - 1, "day_end")
        )

    return positions


def _close_position(datetimes, probas, zs, spreads, entry_idx, exit_idx, exit_reason: str):
    """
    Baut das Ergebnis-dict für eine geschlossene Position (Hilfsfunktion).

    'realized_return' ist der tatsächlich über die (variable) Haltedauer
    realisierte Ertrag in log-Preis-Einheiten: direction * (exit_spread -
    entry_spread), mit derselben direction-Konvention wie in 5.6
    (direction = -1 falls entry_z > 0, sonst +1 -- Wette auf Reversion des
    Spreads Richtung 0). Bewusst NICHT spread_ret_k verwendet: das ist ein
    fixer k-Minuten-Horizont-Ertrag aus dem Modell-Label in Kapitel 5, der
    bei einer Position mit abweichender (durch die Zustandsmaschine
    bestimmter) Haltedauer nicht dem tatsaechlich Realisierten entspricht.
    """
    entry_z = zs[entry_idx]
    direction = -1.0 if entry_z > 0 else 1.0
    realized_return = direction * (spreads[exit_idx] - spreads[entry_idx])
    return {
        "entry_time": datetimes[entry_idx],
        "exit_time": datetimes[exit_idx],
        "entry_proba": probas[entry_idx],
        "exit_proba": probas[exit_idx],
        "entry_z": entry_z,
        "exit_z": zs[exit_idx],
        "entry_spread": spreads[entry_idx],
        "exit_spread": spreads[exit_idx],
        "direction": direction,
        "realized_return": realized_return,
        "duration_min": exit_idx - entry_idx + 1,
        "exit_reason": exit_reason,
    }


def build_positions(
    trade_df: pd.DataFrame,
    *,
    threshold: float,
    min_minutes_to_close: int = 0,
    cap_minutes: int = None,
) -> pd.DataFrame:
    """
    Verwandelt die minütlichen Walk-Forward-Signale (proba/z je (pair,
    datetime)) in eine Liste diskreter Positionen.

    Zustandsmaschine (siehe PositionState / _run_position_state_machine):

    Einstieg — die Positions-Design-Entscheidung aus 6.2: EIN Signal, das
    über mehrere Minuten aktiv bleibt, ist EINE Position, keine pro Minute
    neue, sich aufaddierende Position. Eine neue Position wird nur eröffnet,
    wenn zusätzlich mindestens min_minutes_to_close Minuten Daten im
    laufenden Handelstag verbleiben — ohne diese Regel eröffnen manche
    Signale erst im letzten verfügbaren Moment des Tages und werden Sekunden
    später zwangsweise wieder geschlossen (siehe 06_Backtest.ipynb §6.2,
    Diagnose 3a). Bereits offene Positionen sind von dieser Regel nicht
    betroffen. Absichtlich KEIN symmetrischer Cutoff am Tagesanfang: eine zu
    Handelsbeginn eröffnete Position hat den ganzen Tag Zeit, sich zu
    entwickeln, und die realisierte Edge ist in den ersten Handelsminuten
    empirisch nicht schlechter (§6.2, Diagnose 3b) — nur die Varianz ist
    höher, was eine Positionsgrössen- (6.3), keine Positions-
    Zulässigkeits-Frage ist.

    Ausstieg — ursprünglich (erste Version dieser Funktion) wurde eine
    Position geschlossen, sobald das Signal unter die Schwelle fiel
    ("Exit bei Signal-Off"). Der Vergleich in 06_Backtest.ipynb §6.2.1 zeigt,
    dass diese Regel die Edge des Modells zerstört (deutlich niedrigere
    Trefferquote und negativer mittlerer Ertrag als jede getestete
    Alternative, bei identischen Einstiegen) — das Modell hört auf,
    Reversion zu "erwarten", oft nachdem der Spread bereits wieder in die
    falsche Richtung gelaufen ist, nicht nachdem die Reversion tatsächlich
    stattgefunden hat. Eine Position wird daher stattdessen geschlossen,
    sobald 'z' die Null-Linie in Richtung von 'direction' erreicht (volle
    Reversion — das eigentliche Handelsziel), optional gekappt bei
    cap_minutes, spätestens aber am Handelsende (keine Positionen über
    Nacht). Die volle, ungekappte Reversionsregel hat die stärkste Edge,
    aber auch die längste und breiteste Haltedauer-Verteilung; cap_minutes
    (kalibriert auf tuning_days als datengetriebenes Perzentil dieser
    Verteilung, siehe §6.2.1) behält den grössten Teil von Edge und
    Trefferquote, reduziert aber Haltedauer und Streuung spürbar. Das ist
    KEIN Wiederaufleben des schon einmal verworfenen best_k-basierten
    Max-Hold-Caps (§6.2, Diagnose 1b): dort ging es um einen willkürlichen
    Cap auf eine Regel ohne eigenes Zielkriterium (Signal-Off), hier um eine
    datengetriebene Kappung des langen rechten Randes einer Regel, die
    bereits ein klares ökonomisches Ziel verfolgt (volle Reversion).

    Parameter:
    - trade_df: DataFrame mit Spalten 'pair', 'day', 'datetime', 'proba',
      'z', 'spread' (z. B. trade_df_bt aus 6.1).
    - threshold: Schwelle für 'proba', ab der ein Signal als aktiv gilt.
    - min_minutes_to_close: Mindestanzahl an Minuten, die im (pair, day) nach
      der aktuellen Minute noch an Daten verfügbar sein muss, damit an dieser
      Minute eine NEUE Position eröffnet werden darf (0 = keine Einschränkung).
    - cap_minutes: maximale Haltedauer in Minuten (None = keine Kappung,
      Position läuft bis zur vollen Reversion oder Handelsende).

    Rückgabe:
    - DataFrame, eine Zeile je Position: pair, day, entry_time, exit_time,
      entry_proba, exit_proba, entry_z, exit_z, entry_spread, exit_spread,
      direction, realized_return, duration_min, exit_reason ('z_cross',
      'cap' oder 'day_end'). 'realized_return' ist der tatsächlich
      realisierte Ertrag über die Haltedauer (siehe _close_position) —
      Grundlage für Position Sizing in 6.3.
    """
    df = trade_df.sort_values(["pair", "day", "datetime"]).copy()
    df["active"] = df["proba"] >= threshold

    all_positions = []
    for (pair, day), group in df.groupby(["pair", "day"], sort=False):
        datetimes = group["datetime"].to_numpy()
        probas = group["proba"].to_numpy()
        zs = group["z"].to_numpy()
        spreads = group["spread"].to_numpy()
        active = group["active"].to_numpy()

        minutes_to_close = (datetimes[-1] - datetimes) / np.timedelta64(1, "m")
        entry_signal = active & (minutes_to_close >= min_minutes_to_close)

        for pos in _run_position_state_machine(datetimes, probas, zs, spreads, entry_signal, cap_minutes):
            pos["pair"] = pair
            pos["day"] = day
            all_positions.append(pos)

    cols = [
        "pair", "day", "entry_time", "exit_time",
        "entry_proba", "exit_proba", "entry_z", "exit_z",
        "entry_spread", "exit_spread", "direction", "realized_return",
        "duration_min", "exit_reason",
    ]
    positions_df = pd.DataFrame(all_positions, columns=cols)
    return positions_df.sort_values(["pair", "entry_time"]).reset_index(drop=True)


def kelly_fraction(p, b):
    """
    Klassische Kelly-Formel für eine Wette mit Gewinnwahrscheinlichkeit p und
    Auszahlungsverhältnis b (mittlerer Gewinn / |mittlerer Verlust|):

        f* = p - (1 - p) / b

    p und b werden hier bewusst nicht selbst geschätzt (Kalibrierung von proba,
    Schätzung von b) — das erfordert Trainings-/Holdout-Disziplin (nur auf
    tuning_days) und passiert in 06_Backtest.ipynb §6.3. Diese Funktion ist
    absichtlich eine reine, zustandslose Formel, die auf jedes (p, b)-Paar
    angewendet werden kann.

    Rückgabe:
    - f* geclippt auf [0, 1] (keine negativen Fraktionen, keine Hebelung über
      den vollen Kapitaleinsatz hinaus).
    """
    f = p - (1.0 - p) / b
    return np.clip(f, 0.0, 1.0)
