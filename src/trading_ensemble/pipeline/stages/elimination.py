from __future__ import annotations

import re

import pandas as pd
import yfinance as yf
from comet_ml import API

from ..engine import PipelineStage


TOP_N = 10
FINAL_CANDIDATES = 5
MIN_COMPOSITE_SCORE = 50.0
MAX_VOLATILITY_SCORE = 80.0
MIN_VOLUME_RATIO = 1.0
DONCHIAN_PERIOD = 20


def extract_symbol(name: str) -> str:
    match = re.match(r"^([A-Z0-9]+-EQ)", name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return name.split("_")[0].upper()


def get_all_logged_values(exp) -> dict:
    values = {}
    try:
        for m in exp.get_metrics_summary():
            values[m["name"]] = m["valueCurrent"]
    except Exception:
        pass
    try:
        for p in exp.get_parameters_summary():
            values[p["name"]] = p["valueCurrent"]
    except Exception:
        pass
    try:
        for o in exp.get_others_summary():
            values[o["name"]] = o["valueCurrent"]
    except Exception:
        pass
    return values


def fetch_top_n_from_comet(api_key: str, workspace: str, project: str, top_n: int) -> pd.DataFrame:
    api = API(api_key=api_key)
    experiments = list(api.get_experiments(workspace, project))

    records = []
    for exp in experiments:
        try:
            values = get_all_logged_values(exp)
            composite = float(values.get("composite_score", 0))
            if composite <= 0:
                continue

            symbol = extract_symbol(exp.name)
            records.append(
                {
                    "experiment_name": exp.name,
                    "symbol": symbol,
                    "COMPOSITE_SCORE": composite,
                    "SENTIMENT_SCORE": float(values.get("sentiment_score", 0)),
                    "VOLATILITY_SCORE": float(values.get("volatility_score", 0)),
                    "RETURN_20C_PCT": float(values.get("return_20c_pct", 0)),
                    "HIGH_52W": float(values.get("52w_high", 0)),
                }
            )
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.sort_values("COMPOSITE_SCORE", ascending=False, inplace=True)
    df.drop_duplicates(subset="symbol", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df.head(top_n)


def get_donchian_and_volume(symbol_eq: str, period: int = 20):
    ticker = symbol_eq.replace("-EQ", ".NS")
    try:
        df = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty or len(df) < period:
            return None, None, None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        high = df["High"].squeeze().astype(float)
        close = df["Close"].squeeze().astype(float)
        volume = df["Volume"].squeeze().astype(float)

        donchian_upper = float(high.rolling(period).max().iloc[-2])
        ltp = float(close.iloc[-1])
        avg_vol = float(volume.iloc[-period:-1].mean())
        cur_vol = float(volume.iloc[-1])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0.0

        return donchian_upper, ltp, vol_ratio
    except Exception:
        return None, None, None


def apply_elimination_filter(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, row in df.iterrows():
        symbol = row["symbol"]
        composite_score = float(row["COMPOSITE_SCORE"])
        volatility_score = float(row["VOLATILITY_SCORE"])

        eliminated = False

        if composite_score < MIN_COMPOSITE_SCORE:
            eliminated = True
        if volatility_score > MAX_VOLATILITY_SCORE:
            eliminated = True

        donchian_upper, ltp, vol_ratio = get_donchian_and_volume(symbol, DONCHIAN_PERIOD)

        if vol_ratio is None or vol_ratio < MIN_VOLUME_RATIO:
            eliminated = True

        breakout = False
        if donchian_upper is not None and ltp is not None:
            breakout = ltp >= donchian_upper
            if not breakout:
                eliminated = True
        else:
            eliminated = True

        if not eliminated:
            results.append(
                {
                    **row.to_dict(),
                    "LTP": ltp,
                    "DONCHIAN_UPPER": donchian_upper,
                    "VOLUME_RATIO": vol_ratio,
                    "BREAKOUT": breakout,
                }
            )

    candidates_df = pd.DataFrame(results)
    if not candidates_df.empty:
        candidates_df.sort_values("COMPOSITE_SCORE", ascending=False, inplace=True)
        candidates_df = candidates_df.head(FINAL_CANDIDATES)

    return candidates_df


class EliminationStage(PipelineStage):
    name = "elimination"

    def run(self, context):
        settings = context["settings"]

        if not settings.comet_api_key:
            print("Comet API key missing - elimination stage skipped")
            context["candidates_df"] = pd.DataFrame()
            return

        top_df = fetch_top_n_from_comet(
            api_key=settings.comet_api_key,
            workspace=settings.comet_workspace,
            project=settings.comet_project_name,
            top_n=TOP_N,
        )

        if top_df.empty:
            print("No ranked Comet candidates found")
            context["candidates_df"] = pd.DataFrame()
            return

        candidates_df = apply_elimination_filter(top_df)
        context["candidates_df"] = candidates_df

        store = context["store"]
        run_id = context["run_id"]

        for _, row in candidates_df.iterrows():
            store.insert_candidate(
                run_id=run_id,
                symbol=str(row["symbol"]),
                composite_score=float(row.get("COMPOSITE_SCORE", 0.0)),
                volatility_score=float(row.get("VOLATILITY_SCORE", 0.0)),
                ltp=float(row.get("LTP", 0.0)) if pd.notna(row.get("LTP")) else None,
                donchian_upper=float(row.get("DONCHIAN_UPPER", 0.0)) if pd.notna(row.get("DONCHIAN_UPPER")) else None,
                volume_ratio=float(row.get("VOLUME_RATIO", 0.0)) if pd.notna(row.get("VOLUME_RATIO")) else None,
                breakout=bool(row.get("BREAKOUT", False)),
            )

        candidates_df.to_csv(settings.trade_candidates_path, index=False)
        print(f"Saved {len(candidates_df)} candidates -> {settings.trade_candidates_path}")