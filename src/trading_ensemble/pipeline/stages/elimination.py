from __future__ import annotations

import re

import pandas as pd
from comet_ml import API

from trading_ensemble.data.market_providers import ExecutionMarketDataProvider
from trading_ensemble.data.sheets_output import maybe_write_output
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


def get_donchian_and_volume(symbol_eq: str, provider: ExecutionMarketDataProvider, period: int = 20):
    try:
        df = provider.fetch_mode_candles(symbol_eq, mode="SWING")
        if df is None or df.empty or len(df) < period + 1:
            return None, None, None

        high = df["high"].squeeze().astype(float)
        close = df["close"].squeeze().astype(float)
        volume = df["volume"].squeeze().astype(float)

        donchian_upper = float(high.rolling(period).max().iloc[-2])
        ltp = float(close.iloc[-1])
        avg_vol = float(volume.iloc[-period:-1].mean())
        cur_vol = float(volume.iloc[-1])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0.0

        return donchian_upper, ltp, vol_ratio
    except Exception:
        return None, None, None


def apply_elimination_filter(df: pd.DataFrame, provider: ExecutionMarketDataProvider) -> pd.DataFrame:
    results = []
    rejected_score = 0
    rejected_volatility = 0
    rejected_data = 0
    breakout_not_ready = 0
    volume_not_ready = 0

    for _, row in df.iterrows():
        symbol = row["symbol"]
        composite_score = float(row["COMPOSITE_SCORE"])
        volatility_score = float(row["VOLATILITY_SCORE"])

        eliminated = False

        if composite_score < MIN_COMPOSITE_SCORE:
            rejected_score += 1
            eliminated = True

        if volatility_score > MAX_VOLATILITY_SCORE:
            rejected_volatility += 1
            eliminated = True

        donchian_upper, ltp, vol_ratio = get_donchian_and_volume(symbol, provider, DONCHIAN_PERIOD)

        if donchian_upper is None or ltp is None:
            rejected_data += 1
            eliminated = True
            breakout = False
            breakout_ready = False
            volume_ready = False
        else:
            breakout = ltp >= donchian_upper
            breakout_ready = breakout
            volume_ready = vol_ratio is not None and vol_ratio >= MIN_VOLUME_RATIO

            if not breakout_ready:
                breakout_not_ready += 1
            if not volume_ready:
                volume_not_ready += 1

        if not eliminated:
            results.append(
                {
                    **row.to_dict(),
                    "LTP": ltp,
                    "DONCHIAN_UPPER": donchian_upper,
                    "VOLUME_RATIO": vol_ratio,
                    "BREAKOUT": breakout,
                    "BREAKOUT_READY": breakout_ready,
                    "VOLUME_READY": volume_ready,
                }
            )

    candidates_df = pd.DataFrame(results)
    if not candidates_df.empty:
        candidates_df.sort_values("COMPOSITE_SCORE", ascending=False, inplace=True)
        candidates_df = candidates_df.head(FINAL_CANDIDATES)

    print("Elimination diagnostics:")
    print(f"  rejected_score      = {rejected_score}")
    print(f"  rejected_volatility = {rejected_volatility}")
    print(f"  rejected_data       = {rejected_data}")
    print(f"  breakout_not_ready  = {breakout_not_ready}")
    print(f"  volume_not_ready    = {volume_not_ready}")
    print(f"  final_candidates    = {len(candidates_df)}")

    return candidates_df


class EliminationStage(PipelineStage):
    name = "elimination"

    def run(self, context):
        settings = context["settings"]
        control_panel = context.get("control_panel")

        if not settings.comet_api_key:
            print("Comet API key missing - elimination stage skipped")
            context["candidates_df"] = pd.DataFrame()
            maybe_write_output(settings, control_panel, "SelectedCandidates", pd.DataFrame())
            return

        top_df = fetch_top_n_from_comet(
            api_key=settings.comet_api_key,
            workspace=settings.comet_workspace,
            project=settings.comet_project_name,
            top_n=TOP_N,
        )

        print(f"Comet ranked symbols fetched: {len(top_df)}")
        context["comet_ranked_count"] = len(top_df)

        if top_df.empty:
            print("No ranked Comet candidates found")
            context["candidates_df"] = pd.DataFrame()
            maybe_write_output(settings, control_panel, "SelectedCandidates", pd.DataFrame())
            return

        watchlist_df = context.get("watchlist_df", pd.DataFrame())
        if not watchlist_df.empty:
            allowed_symbols = set(watchlist_df["symbol"].astype(str).str.upper())
            before = len(top_df)
            top_df = top_df[top_df["symbol"].astype(str).str.upper().isin(allowed_symbols)].copy()
            print(f"After watchlist filter: {len(top_df)} / {before}")

        if top_df.empty:
            print("No Comet-ranked symbols matched the active watchlist")
            context["candidates_df"] = pd.DataFrame()
            maybe_write_output(settings, control_panel, "SelectedCandidates", pd.DataFrame())
            return

        try:
            market_provider = ExecutionMarketDataProvider.from_env()
        except Exception as exc:
            print(f"Execution market data provider unavailable: {exc}")
            context["candidates_df"] = pd.DataFrame()
            maybe_write_output(settings, control_panel, "SelectedCandidates", pd.DataFrame())
            return

        candidates_df = apply_elimination_filter(top_df, market_provider)
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
        maybe_write_output(settings, control_panel, "SelectedCandidates", candidates_df)

        print(f"Saved {len(candidates_df)} candidates -> {settings.trade_candidates_path}")
