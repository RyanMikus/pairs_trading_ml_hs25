import time
import sys

import pandas as pd
import requests


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_universe(
    *,
    replace_dot: bool = True,
    user_agent: str = "Mozilla/5.0",
    verbose: bool = True,
) -> list[str]:
    headers = {"User-Agent": user_agent}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    table = next((t for t in tables if "Symbol" in t.columns), None)

    if table is None:
        raise RuntimeError("Keine 'Symbol'-Spalte auf Wikipedia gefunden")

    tickers = table["Symbol"].astype(str).str.strip().tolist()

    if replace_dot:
        tickers = [t.replace(".", "-") for t in tickers]

    if verbose:
        print("Aktienuniversum erfolgreich geladen")
        print("Anzahl Aktien:", len(tickers))
        print("Beispiele:", tickers[:10])

    return tickers


def _fetch_yahoo_intraday(
    ticker: str,
    *,
    interval: str,
    range_: str,
    session: requests.Session,
    timeout_s: int,
) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": interval, "range": range_, "region": "US", "lang": "en-US"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }

    r = session.get(url, headers=headers, params=params, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"{ticker}: HTTP {r.status_code}")

    data = r.json().get("chart", {}).get("result")
    if not data:
        raise RuntimeError(f"{ticker}: keine Daten")

    result = data[0]
    timestamps = result.get("timestamp")
    quote = result.get("indicators", {}).get("quote", [{}])[0]

    if not timestamps or "close" not in quote:
        raise RuntimeError(f"{ticker}: unvollständige Daten")

    df = pd.DataFrame({
        "symbol": ticker,
        "datetime": pd.to_datetime(timestamps, unit="s"),
        "open": quote.get("open"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "close": quote.get("close"),
        "volume": quote.get("volume"),
    })

    df = df.loc[~df[["open", "high", "low", "close"]].isna().all(axis=1)]
    if df.empty:
        raise RuntimeError(f"{ticker}: nur leere Kerzen")

    return df


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _print_progress(
    *,
    done: int,
    total: int,
    start_time: float,
    last_label: str = "",
    bar_width: int = 30,
) -> None:
    frac = done / total if total else 1.0
    filled = int(bar_width * frac)
    bar = "#" * filled + "-" * (bar_width - filled)

    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else float("inf")

    pct = frac * 100.0
    eta = _format_eta(remaining)

    msg = f"\r[{bar}] {done}/{total} ({pct:5.1f}%) ETA {eta}"
    if last_label:
        msg += f" | {last_label}"

    sys.stdout.write(msg)
    sys.stdout.flush()

    if done == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def collect_sp500_intraday_data(
    *,
    universe: list[str],
    date: str,
    interval: str = "1m",
    range_: str = "1d",
    sleep_s: float = 0.1,
    timeout_s: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print("Lade Intraday-Daten für", len(universe), "Aktien")
        print("Datum:", date, "| Interval:", interval)

    all_data: list[pd.DataFrame] = []
    total = len(universe)
    start = time.time()

    # Throttle für Progress-Updates
    last_update = 0.0
    update_every_s = 0.2

    with requests.Session() as session:
        for i, ticker in enumerate(universe, start=1):
            df_t = _fetch_yahoo_intraday(
                ticker,
                interval=interval,
                range_=range_,
                session=session,
                timeout_s=timeout_s,
            )
            all_data.append(df_t)

            if verbose:
                now = time.time()
                if (now - last_update) >= update_every_s or i == total:
                    _print_progress(
                        done=i,
                        total=total,
                        start_time=start,
                        last_label=ticker,
                    )
                    last_update = now

            if sleep_s > 0:
                time.sleep(sleep_s)

    df_all = pd.concat(all_data, ignore_index=True)

    if verbose:
        print("\nDatensatz erstellt")
        print("Zeilen:", len(df_all))
        print("Ticker:", df_all["symbol"].nunique())
        print("Zeitbereich:", df_all["datetime"].min(), " bis ", df_all["datetime"].max())

    return df_all
