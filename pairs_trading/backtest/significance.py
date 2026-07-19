"""
pairs_trading.backtest.significance

Statistische Signifikanzprüfung für die Backtest-Performance.

Wichtiger Hinweis:
- Prüft, ob die beobachtete Performance statistisch von Null unterscheidbar
  ist. Da Trades autokorreliert und nicht unabhängig sind, wird bewusst kein
  naiver t-Test auf Einzel-Trade-Ebene verwendet, sondern ein Block-Bootstrap
  bzw. eine Newey-West-korrigierte Schätzung auf Basis der täglich
  aggregierten PnL.
- Stellt zusätzlich eine wiederverwendbare Null-Kalibrierungs-Kontrolle
  bereit, im selben Geist wie der in 05_Modellierung.ipynb etablierte
  Random-Walk-Check (dort: prüft, ob eine PnL-Kennzahl auf zwei unabhängigen
  Random Walks korrekt ~50% Trefferquote / ~0 mittleren PnL anzeigt statt
  ein mechanisches Artefakt zu melden; hier: prüft, ob ein Signifikanztest
  auf einer künstlichen Serie mit garantiert wahrem Mittelwert von Null
  nicht öfter ablehnt, als sein Alpha-Niveau erlaubt).
"""

import numpy as np


def block_bootstrap_mean_test(
    values,
    *,
    block_length: int,
    n_bootstrap: int = 10000,
    random_state=None,
) -> dict:
    """
    Moving-Block-Bootstrap-Test von H0: mean(values) == 0.

    Autokorrelierte Werte (z. B. tägliche Portfolio-PnL, da benachbarte Tage
    demselben Walk-Forward-Modell und Marktregime ausgesetzt sind) verletzen
    die i.i.d.-Annahme eines naiven t-Tests. Das Resampling zusammenhängender
    Blöcke der Länge block_length (mit Zurücklegen, aneinandergehängt bis zur
    Ausgangslänge) erhält die kurzfristige Autokorrelationsstruktur innerhalb
    jedes Blocks, während trotzdem randomisiert wird, welcher Zeitabschnitt
    der Stichprobe ins Resample einfliesst.

    Parameter:
    - values: 1D array-artig, tägliche PnL-Beobachtungen.
    - block_length: Länge je zusammenhängendem Block (>= 1), aus der
      Autokorrelationsstruktur der Daten abgeleitet, nicht willkürlich fest
      gewählt.
    - n_bootstrap: Anzahl Bootstrap-Resamples.
    - random_state: Seed oder np.random.Generator für Reproduzierbarkeit.

    Rückgabe:
    - dict mit observed_mean, ci_low, ci_high (95%-Perzentil-KI der
      Bootstrap-Mittelwertverteilung), p_value (zweiseitig, aus dem Anteil
      der Bootstrap-Mittelwerte auf jeder Seite von Null) und
      bootstrap_means (das volle Array).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    block_length = max(1, min(block_length, n))
    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length

    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([values[s : s + block_length] for s in starts])[:n]
        bootstrap_means[i] = sample.mean()

    observed_mean = values.mean()
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    share_below = (bootstrap_means <= 0).mean()
    share_above = (bootstrap_means >= 0).mean()
    p_value = min(2 * min(share_below, share_above), 1.0)

    return {
        "observed_mean": observed_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "bootstrap_means": bootstrap_means,
    }


def newey_west_mean_test(values, *, max_lags: int | None = None) -> dict:
    """
    Newey-West-(HAC-)robuster Test von H0: mean(values) == 0.

    Schätzt die Long-Run-Varianz des Mittelwerts über einen gewichteten
    Bartlett-Kernel-Zuschlag der Autokovarianzen bis max_lags, statt die
    naive i.i.d.-Standardabweichung zu verwenden -- robust gegenüber genau
    der Art von Autokorrelation, die block_bootstrap_mean_test über
    Block-Resampling abfängt. Beide Verfahren sollten bei echtem Signal zu
    einer übereinstimmenden Schlussfolgerung kommen; eine Abweichung
    zwischen beiden ist selbst ein Diagnosesignal.

    Parameter:
    - values: 1D array-artig, tägliche PnL-Beobachtungen.
    - max_lags: Anzahl Lags im Bartlett-Kernel. Falls None, automatische
      Newey-West-Bandbreite floor(4 * (n/100)^(2/9)).

    Rückgabe:
    - dict mit observed_mean, se (HAC-robuster Standardfehler), t_stat,
      p_value (zweiseitig, asymptotisch normalverteilt), max_lags.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if max_lags is None:
        max_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    max_lags = max(0, min(max_lags, n - 1))

    mean = values.mean()
    centered = values - mean

    long_run_var = np.dot(centered, centered) / n
    for lag in range(1, max_lags + 1):
        weight = 1 - lag / (max_lags + 1)  # Bartlett-Kernel
        gamma_lag = np.dot(centered[lag:], centered[:-lag]) / n
        long_run_var += 2 * weight * gamma_lag

    se = np.sqrt(max(long_run_var, 0.0) / n)
    t_stat = mean / se if se > 0 else np.nan

    from scipy.stats import norm

    p_value = 2 * (1 - norm.cdf(abs(t_stat))) if np.isfinite(t_stat) else np.nan

    return {
        "observed_mean": mean,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "max_lags": max_lags,
    }


def null_calibration_check(
    values,
    test_fn,
    *,
    n_trials: int = 500,
    alpha: float = 0.05,
    random_state=None,
) -> dict:
    """
    Validiert eine Signifikanztest-Funktion an einer künstlichen Nullserie
    mit garantiert wahrem Mittelwert von Null -- im selben Geist wie der
    Random-Walk-Check aus 05_Modellierung.ipynb (dort: zwei unabhängige
    Random Walks ohne echte Beziehung; hier: eine Serie, deren wahrer
    Mittelwert durch Konstruktion exakt Null ist, aber deren
    Autokorrelations-/Varianzstruktur aus der echten Serie übernommen wird).

    Konstruiert die Nullserien durch Demeaning der beobachteten Serie
    (erzwingt einen wahren Mittelwert von exakt 0) und zieht davon je Trial
    ein Moving-Block-Resample (erhält die echte Autokorrelations-/
    Varianzstruktur, randomisiert alles andere). test_fn wird auf jede
    Nullserie angewendet; die empirische Ablehnungsrate bei alpha sollte
    nahe alpha liegen, wenn der Test korrekt kalibriert ist -- eine
    Ablehnungsrate deutlich über alpha würde bedeuten, dass test_fn zu oft
    "Signifikanz" meldet, wo per Konstruktion keine ist.

    Parameter:
    - values: die echte beobachtete Serie (dient nur dazu, die
      Autokorrelations-/Varianzstruktur über Demeaning + Block-Resampling
      abzuleiten).
    - test_fn: callable(values) -> dict mit Schlüssel 'p_value' (z. B.
      block_bootstrap_mean_test oder newey_west_mean_test, mit
      functools.partial vorbelegt).
    - n_trials: Anzahl künstlicher Nullserien-Trials.
    - alpha: Signifikanzniveau, gegen das die Kalibrierung geprüft wird.

    Rückgabe:
    - dict mit empirical_rejection_rate, n_trials, alpha.
    """
    values = np.asarray(values, dtype=float)
    demeaned = values - values.mean()
    n = len(demeaned)
    block_length = max(1, int(round(np.sqrt(n))))
    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length

    rejections = 0
    for _ in range(n_trials):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        synthetic = np.concatenate([demeaned[s : s + block_length] for s in starts])[:n]
        result = test_fn(synthetic)
        if result["p_value"] < alpha:
            rejections += 1

    return {
        "empirical_rejection_rate": rejections / n_trials,
        "n_trials": n_trials,
        "alpha": alpha,
    }
