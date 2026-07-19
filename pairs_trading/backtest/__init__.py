"""
pairs_trading.backtest

Public API (Re-Exports) für Backtest und Performance-Analyse im pairs_trading Projekt.

Wichtiger Hinweis:
- Dieses Subpackage baut auf den in `pairs_trading.modeling` erzeugten
  Walk-Forward-Artefakten (`trade_df`) auf und ergänzt Position Sizing,
  Transaktionskosten, Portfolio-Aggregation, Risikomanagement, Performance-
  Kennzahlen und statistische Signifikanzprüfung.
- Diese Datei definiert die nach außen sichtbare Oberfläche des Subpackages,
  analog zu `pairs_trading.modeling` (flache Funktions-Exports statt
  Submodul-Importe).
- Performance-Kennzahlen (Abschnitt 6.7) wurden direkt im Notebook
  implementiert (kein eigenes `metrics`-Modul nötig, da die Berechnung
  ausschliesslich bereits vorhandene Spalten aggregiert); `metrics` und
  `plotting` bleiben unimplementierte Platzhalter.
"""

from .sizing import (
    PositionState,
    build_positions,
    kelly_fraction,
)

from .portfolio import (
    transaction_cost_per_position,
    aggregate_daily_portfolio,
    concurrent_exposure,
)

from .risk import (
    cap_concurrent_exposure,
)

from .significance import (
    block_bootstrap_mean_test,
    newey_west_mean_test,
    null_calibration_check,
)

__all__ = [
    # sizing
    "PositionState",
    "build_positions",
    "kelly_fraction",

    # portfolio
    "transaction_cost_per_position",
    "aggregate_daily_portfolio",
    "concurrent_exposure",

    # risk
    "cap_concurrent_exposure",

    # significance
    "block_bootstrap_mean_test",
    "newey_west_mean_test",
    "null_calibration_check",
]
