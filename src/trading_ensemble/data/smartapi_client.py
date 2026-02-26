"""
SmartAPI data connector extracted from StageA notebook. [file:33]

Design goal: keep all broker/API I/O here so strategy/backtest code stays pure.
Later deployment can replace/extend these functions to stream live data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import os
import time
import random

import pandas as pd
import pyotp
from SmartApi import SmartConnect

SCRIPMASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

@dataclass
class SmartAPISession:
    smart: Any
    jwt_token: str
    refresh_token: str
    feed_token: str
    profile: Dict[str, Any]

def login_from_env() -> SmartAPISession:
    """
    Env vars expected (set in Codespaces secrets or local .env):
    ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PIN, ANGEL_TOTP_SECRET [file:33]
    """
    api_key = os.getenv("ANGEL_API_KEY")
    client_code = os.getenv("ANGEL_CLIENT_CODE")
    pin = os.getenv("ANGEL_PIN")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET")

    missing = [k for k, v in {
        "ANGEL_API_KEY": api_key,
        "ANGEL_CLIENT_CODE": client_code,
        "ANGEL_PIN": pin,
        "ANGEL_TOTP_SECRET": totp_secret,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")

    smart = SmartConnect(api_key=api_key)
    totp_code = pyotp.TOTP(totp_secret).now()
    profile = smart.generateSession(client_code, pin, totp_code)
    if not profile or profile.get("status") is False:
        raise RuntimeError(f"generateSession failed: {profile}")

    jwt_token = profile["data"]["jwtToken"]
    refresh_token = profile["data"]["refreshToken"]
    feed_token = smart.getfeedToken()

    # Optional calls used in the repo flow (safe to ignore failures)
    try:
        smart.getProfile(refresh_token)
        smart.generateToken(refresh_token)
    except Exception:
        pass

    return SmartAPISession(
        smart=smart,
        jwt_token=jwt_token,
        refresh_token=refresh_token,
        feed_token=feed_token,
        profile=profile,
    )

def _split_date_ranges(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if end <= start:
        raise ValueError("end must be after start")
    ranges = []
    cur = start
    step = pd.Timedelta(days=int(chunk_days))
    while cur < end:
        nxt = min(cur + step, end)
        ranges.append((cur, nxt))
        cur = nxt
    return ranges

def _candles_to_df(raw: Any) -> pd.DataFrame:
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    if not isinstance(raw, dict):
        return pd.DataFrame(columns=cols)
    data = raw.get("data") or []
    if not data:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(data, columns=cols)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).reset_index(drop=True)
    return df

def fetch_candles_chunked(
    smart: Any,
    *,
    exchange: str,
    symbol_token: str,
    interval: str,
    start: str,
    end: str,
    chunk_days: int = 60,
    sleep_seconds: float = 1.5,
    max_retries: int = 12,
) -> pd.DataFrame:
    """
    Rate-limit-resilient historical candles fetch, based on StageA. [file:33]
    start/end format example: '2025-08-25 0915'
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    pieces: List[pd.DataFrame] = []
    for a, b in _split_date_ranges(start_dt, end_dt, chunk_days=chunk_days):
        from_dt = a.strftime("%Y-%m-%d %H%M")
        to_dt = b.strftime("%Y-%m-%d %H%M")

        time.sleep(sleep_seconds + random.uniform(0, 0.35))

        attempt = 0
        while True:
            attempt += 1
            try:
                params = {
                    "exchange": exchange,
                    "symboltoken": str(symbol_token),
                    "interval": interval,
                    "fromdate": from_dt,
                    "todate": to_dt,
                }
                raw = smart.getCandleData(params)
                part = _candles_to_df(raw)
                if not part.empty:
                    pieces.append(part)
                break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(min(60.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 1.0))
                    continue
                raise RuntimeError(f"getCandleData failed after retries: {e}") from e

    if not pieces:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    out = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    return out

def load_scrip_master(url: str = SCRIPMASTER_URL) -> pd.DataFrame:
    """Offline symbol->token map as in StageA. [file:33]"""
    df = pd.read_json(url)
    df["exch_seg"] = df["exch_seg"].astype(str).str.upper()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["token"] = df["token"].astype(str)
    return df

def resolve_symbol_to_token_offline(symbol: str, *, exchange: str = "NSE", scrip_df: Optional[pd.DataFrame] = None) -> Tuple[str, str]:
    """
    Returns (trading_symbol, token). Normalizes to '*-EQ' when needed. [file:33]
    """
    df = scrip_df if scrip_df is not None else load_scrip_master()
    df_ex = df[df["exch_seg"].str.upper() == exchange.upper()]

    sym_in = symbol.strip().upper()
    if "-" not in sym_in:
        sym_key = f"{sym_in}-EQ"
        base = sym_in
    else:
        sym_key = sym_in
        base = sym_in.split("-", 1)[0]

    exact = df_ex[df_ex["symbol"] == sym_key]
    if len(exact) == 1:
        row = exact.iloc[0]
        return row["symbol"], str(row["token"])

    cand = df_ex[df_ex["symbol"].str.startswith(base) & df_ex["symbol"].str.endswith("-EQ")]
    if len(cand) == 1:
        row = cand.iloc[0]
        return row["symbol"], str(row["token"])

    raise RuntimeError(f"Token not found or ambiguous for: {symbol}")
