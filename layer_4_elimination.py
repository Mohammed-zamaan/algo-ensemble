# layer_4_elimination.py
# LAYER 4 — Elimination Filter (Pre-Trade Gate)

import os
import re
import pandas as pd
import yfinance as yf
from comet_ml import API
from dotenv import load_dotenv
from src.trading_ensemble.data.comet_screener import get_top_stocks
from src.trading_ensemble.data.dual_source import to_yf_ticker

load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TOP_N                = 10
FINAL_CANDIDATES     = 5
MIN_COMPOSITE_SCORE  = 50.0
MAX_VOLATILITY_SCORE = 80.0
MIN_VOLUME_RATIO     = 1.0
DONCHIAN_PERIOD      = 20
# ─────────────────────────────────────────────


def extract_symbol(name: str) -> str:
    """Extract clean NSE symbol from any experiment name."""
    match = re.match(r'^([A-Z0-9]+-EQ)', name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return name.split("_")[0].upper()


def get_all_logged_values(exp) -> dict:
    """
    Comet ML stores values in metrics, parameters, and 'others'.
    Try all three and merge into one flat dict.
    """
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


def fetch_top_n_from_comet(top_n: int) -> pd.DataFrame:
    api = API(api_key=COMET_API_KEY)
    experiments = api.get_experiments(COMET_WORKSPACE, COMET_PROJECT)
    exp_list = list(experiments)
    print(f"  [INFO] Total experiments in {COMET_PROJECT}: {len(exp_list)}")
    records = []
    for exp in exp_list:
        try:
            v = get_all_logged_values(exp)
            composite = float(v.get("composite_score", 0))
            if composite <= 0:
                continue
            symbol = extract_symbol(exp.name)
            records.append({
                "experiment_name":  exp.name,
                "symbol":           symbol,
                "COMPOSITE_SCORE":  composite,
                "SENTIMENT_SCORE":  float(v.get("sentiment_score",  0)),
                "VOLATILITY_SCORE": float(v.get("volatility_score", 0)),
                "RETURN_20C_PCT":   float(v.get("return_20c_pct",   0)),
                "HIGH_52W":         float(v.get("52w_high",         0)),
            })
        except Exception as e:
            print(f"  [WARN] Could not read {exp.name}: {e}")
    if not records:
        print("  [ERROR] No experiments with composite_score > 0 found.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df.sort_values("COMPOSITE_SCORE", ascending=False, inplace=True)
    df.drop_duplicates(subset="symbol", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  [INFO] {len(df)} unique symbols found.")
    return df.head(top_n)


def get_donchian_and_volume(symbol_eq: str, period: int = 20):
    ticker = symbol_eq.replace("-EQ", ".NS")
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
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
    except Exception as e:
        print(f"  [WARN] yfinance error {ticker}: {e}")
        return None, None, None
        donchian_upper = float(df["High"].rolling(period).max().iloc[-2])
        ltp            = float(df["Close"].iloc[-1])
        avg_vol        = float(df["Volume"].iloc[-period:-1].mean())
        cur_vol        = float(df["Volume"].iloc[-1])
        vol_ratio      = cur_vol / avg_vol if avg_vol > 0 else 0.0
        return donchian_upper, ltp, vol_ratio
    except Exception as e:
        print(f"  [WARN] yfinance error {ticker}: {e}")
        return None, None, None


def apply_elimination_filter(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    print("\n" + "="*65)
    print("  LAYER 4 — ELIMINATION FILTER")
    print("="*65)
    print(f"  Input: {len(df)} stocks")
    print("-"*65)

    for _, row in df.iterrows():
        symbol  = row["symbol"]
        name    = row["experiment_name"]
        c_score = row["COMPOSITE_SCORE"]
        v_score = row["VOLATILITY_SCORE"]
        eliminated = False
        reason = []

        if c_score < MIN_COMPOSITE_SCORE:
            eliminated = True
            reason.append(f"COMPOSITE_SCORE {c_score:.1f} < {MIN_COMPOSITE_SCORE}")

        if v_score > MAX_VOLATILITY_SCORE:
            eliminated = True
            reason.append(f"VOLATILITY_SCORE {v_score:.1f} > {MAX_VOLATILITY_SCORE}")

        donchian_upper, ltp, vol_ratio = get_donchian_and_volume(symbol, DONCHIAN_PERIOD)

        if vol_ratio is not None and vol_ratio < MIN_VOLUME_RATIO:
            eliminated = True
            reason.append(f"Volume ratio {vol_ratio:.2f} < {MIN_VOLUME_RATIO}")

        breakout = False
        if donchian_upper is not None and ltp is not None:
            breakout = ltp >= donchian_upper
            if not breakout:
                eliminated = True
                reason.append(f"No breakout: LTP {ltp:.2f} < Donchian {donchian_upper:.2f}")
        elif ltp is None:
            eliminated = True
            reason.append("Could not fetch market data")

        vr_str = f"{vol_ratio:.2f}" if vol_ratio else "N/A"
        marker = "X" if eliminated else "OK"
        status = "ELIMINATED" if eliminated else "CANDIDATE"
        print(f"  [{marker}] {status:<12} {symbol:<20} Score={c_score:.1f}  "
              f"Vola={v_score:.1f}  LTP={ltp}  Don={donchian_upper}  VolR={vr_str}")
        if reason:
            print(f"       Reason: {' | '.join(reason)}")

        if not eliminated:
            results.append({**row.to_dict(),
                             "LTP": ltp, "DONCHIAN_UPPER": donchian_upper,
                             "VOLUME_RATIO": vol_ratio, "BREAKOUT": breakout})

    candidates_df = pd.DataFrame(results)
    if not candidates_df.empty:
        candidates_df.sort_values("COMPOSITE_SCORE", ascending=False, inplace=True)
        candidates_df = candidates_df.head(FINAL_CANDIDATES)

    print("-"*65)
    print(f"  Result: {len(candidates_df)} TRADE CANDIDATE(s) passed all filters")
    print("="*65)
    return candidates_df


def save_candidates(df: pd.DataFrame, path: str = "trade_candidates.csv"):
    if df.empty:
        print("\n  [INFO] No candidates today — no trades to place.")
        return
    df.to_csv(path, index=False)
    print(f"\n  [SAVED] {len(df)} trade candidates -> {path}")
    cols = ["symbol", "COMPOSITE_SCORE", "VOLATILITY_SCORE",
            "LTP", "DONCHIAN_UPPER", "VOLUME_RATIO", "BREAKOUT"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    print(f"\n[LAYER 4] Connecting to Comet ML: {COMET_WORKSPACE}/{COMET_PROJECT}")
    top_df = fetch_top_n_from_comet(TOP_N)
    if top_df.empty:
        print("[LAYER 4] Exiting — re-run run_screener_v2.py first.")
    else:
        print(f"[LAYER 4] Fetched {len(top_df)} stocks. Running elimination filter...")
        candidates = apply_elimination_filter(top_df)
        save_candidates(candidates, "trade_candidates.csv")
