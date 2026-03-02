#!/usr/bin/env bash
set -e
echo "=== algo-ensemble auto-patcher ==="

echo "[1/9] Fixing requirements.txt..."
python3 -c "
open('requirements.txt','w').write('comet-ml\nyfinance\npandas\nnumpy\npython-dotenv\npyotp\nsmartapi-python\npycryptodome\nlogzero\nwebsocket-client\nrequests\nbeautifulsoup4\n')
print('  Done.')
"

echo "[2/9] Creating comet_screener.py..."
python3 -c "
content = open('/dev/stdin').read()
open('src/trading_ensemble/data/comet_screener.py','w').write(content)
print('  Done.')
" << 'PYEOF'
from __future__ import annotations
import re
import pandas as pd
from comet_ml.api import API

COMET_WORKSPACE = "zamaan"
COMET_PROJECT   = "stock-screener"

def get_top_stocks(min_score: float = 60) -> pd.DataFrame:
    api  = API()
    exps = api.get_experiments(COMET_WORKSPACE, COMET_PROJECT)
    records = []
    for exp in exps:
        name = exp.name or ""
        m = re.match(r"^([A-Z&]+)-EQ", name)
        if not m:
            continue
        symbol  = m.group(1)
        metrics = {mt["name"]: mt["valueCurrent"]
                   for mt in exp.get_metrics_summary() or []}
        score   = metrics.get("composite_score", 0)
        if score >= min_score:
            records.append({"symbol": symbol + ".NS", "composite_score": score})
    return pd.DataFrame(records).sort_values("composite_score", ascending=False)
PYEOF

echo "[3/9] Removing duplicated get_top_stocks() from layer files..."
python3 << 'PYEOF'
import re, os
files = [
    'layer_4_elimination.py','swing_layer_4_elimination.py',
    'swing_layer_9_backtest.py','positional_layer_4_elimination.py',
    'positional_layer_9_backtest.py','run_swing_backtest.py',
]
import_line = 'from src.trading_ensemble.data.comet_screener import get_top_stocks\n'
for f in files:
    if not os.path.exists(f): print(f'  Skip: {f}'); continue
    src = open(f).read()
    src = re.sub(r'^COMET_API_KEY\s*=.*\n','',src,flags=re.MULTILINE)
    src = re.sub(r'^COMET_WORKSPACE\s*=.*\n','',src,flags=re.MULTILINE)
    src = re.sub(r'^COMET_PROJECT\s*=.*\n','',src,flags=re.MULTILINE)
    src = re.sub(r'def get_top_stocks\(\):.*?(?=\ndef |\Z)','',src,flags=re.DOTALL)
    src = re.sub(r'^from comet_ml\.api import API\n','',src,flags=re.MULTILINE)
    if import_line not in src:
        src = re.sub(r'((?:^(?:import|from) .+\n)+)', lambda m: m.group(0)+import_line, src, count=1, flags=re.MULTILINE)
    open(f,'w').write(src)
    print(f'  Patched: {f}')
PYEOF

echo "[4/9] Fixing to_yf_ticker duplication in layer files..."
python3 << 'PYEOF'
import re, os
files = [
    'layer_4_elimination.py','swing_layer_4_elimination.py',
    'swing_layer_9_backtest.py','positional_layer_4_elimination.py',
    'positional_layer_9_backtest.py','run_swing_backtest.py',
]
import_line = 'from src.trading_ensemble.data.dual_source import to_yf_ticker\n'
for f in files:
    if not os.path.exists(f): continue
    src = open(f).read()
    src = re.sub(r'symbol\s*\+\s*["\']\.NS["\']','to_yf_ticker(symbol)',src)
    if import_line not in src:
        src = re.sub(r'((?:^(?:import|from) .+\n)+)', lambda m: m.group(0)+import_line, src, count=1, flags=re.MULTILINE)
    open(f,'w').write(src)
    print(f'  Patched: {f}')
PYEOF

echo "[5/9] Fixing env var names in dual_source.py..."
sed -i \
    -e 's/os\.getenv("SMARTAPI_KEY")/os.getenv("ANGEL_API_KEY")/g' \
    -e 's/os\.getenv("SMARTAPI_TOTP_SECRET")/os.getenv("ANGEL_TOTP_SECRET")/g' \
    -e 's/os\.getenv("SMARTAPI_CLIENT_ID")/os.getenv("ANGEL_CLIENT_CODE")/g' \
    -e 's/os\.getenv("SMARTAPI_PASSWORD")/os.getenv("ANGEL_PIN")/g' \
    src/trading_ensemble/data/dual_source.py
echo "  Done."

echo "[6/9] Removing inline SmartAPI login from dual_source.py..."
python3 << 'PYEOF'
import re
filepath = 'src/trading_ensemble/data/dual_source.py'
src = open(filepath).read()
new_fn = '''def fetch_smartapi_candles(symbol_token: str, interval: str = "FIFTEEN_MINUTE", days: int = 30):
    """Delegates to smartapi_client — single session, chunked, retry-safe."""
    from src.trading_ensemble.data.smartapi_client import login_from_env, fetch_candles_chunked
    from datetime import datetime, timedelta
    session  = login_from_env()
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    return fetch_candles_chunked(
        session.smart, exchange="NSE", symbol_token=str(symbol_token),
        interval=interval,
        start=start_dt.strftime("%Y-%m-%d 0915"),
        end=end_dt.strftime("%Y-%m-%d 1530"),
        chunk_days=60,
    )
'''
src = re.sub(r'def fetch_smartapi_candles\(.*?(?=\ndef |\Z)', new_fn, src, flags=re.DOTALL)
open(filepath,'w').write(src)
print('  Done.')
PYEOF

echo "[7/9] Adding fetch_candles_live() to smartapi_client.py..."
python3 << 'PYEOF'
filepath = 'src/trading_ensemble/data/smartapi_client.py'
src = open(filepath).read()
live_fn = '''
def fetch_candles_live(smart, *, exchange, symbol_token, interval, start, end):
    """Live-safe fetch: max 3 retries, 0.5s sleep — will not block live orders."""
    return fetch_candles_chunked(
        smart, exchange=exchange, symbol_token=symbol_token,
        interval=interval, start=start, end=end,
        chunk_days=1, sleep_seconds=0.5, max_retries=3,
    )
'''
if 'def fetch_candles_live' not in src:
    src += live_fn
    open(filepath,'w').write(src)
    print('  Done.')
else:
    print('  Already exists, skipped.')
PYEOF

echo "[8/9] Vectorising compute_dynamic_donchian() in engine.py..."
python3 << 'PYEOF'
import re
filepath = 'src/trading_ensemble/core/engine.py'
src = open(filepath).read()
new_fn = '''def compute_dynamic_donchian(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Vectorised Donchian with regime-adaptive windows. Replaces Python for-loop."""
    out = df.copy()
    out["_up_lo"]   = out["high"].shift(1).rolling(p.don_entry_lo,   min_periods=p.don_entry_lo).max()
    out["_up_base"] = out["high"].shift(1).rolling(p.don_entry_base, min_periods=p.don_entry_base).max()
    out["_up_hi"]   = out["high"].shift(1).rolling(p.don_entry_hi,   min_periods=p.don_entry_hi).max()
    out["_dn_lo"]   = out["low"].shift(1).rolling(p.don_exit_lo,     min_periods=p.don_exit_lo).min()
    out["_dn_base"] = out["low"].shift(1).rolling(p.don_exit_base,   min_periods=p.don_exit_base).min()
    out["_dn_hi"]   = out["low"].shift(1).rolling(p.don_exit_hi,     min_periods=p.don_exit_hi).min()
    is_hi = out["atrp_sm"] >= p.atrp_hi
    is_lo = out["atrp_sm"] <= p.atrp_lo
    out["don_up_entry"]   = np.where(is_hi, out["_up_hi"], np.where(is_lo, out["_up_lo"], out["_up_base"]))
    out["don_dn_exit"]    = np.where(is_hi, out["_dn_hi"], np.where(is_lo, out["_dn_lo"], out["_dn_base"]))
    out["trail_mult_eff"] = np.where(is_hi, p.trail_mult_hi, p.trail_mult_lo)
    out.drop(columns=[c for c in out.columns if c.startswith("_up_") or c.startswith("_dn_")], inplace=True)
    return out

'''
src = re.sub(
    r'def compute_dynamic_donchian\(df: pd\.DataFrame, p: StrategyParams\) -> pd\.DataFrame:.*?(?=\ndef |\Z)',
    new_fn, src, flags=re.DOTALL
)
open(filepath,'w').write(src)
print('  Done.')
PYEOF

echo "[9/9] Creating pre_market_fetch.py..."
python3 << 'PYEOF'
content = '''"""
pre_market_fetch.py — Run at 08:45 AM before market open.
Fetches sentiment + fundamentals for all Comet stocks and caches to JSON.
Live pipeline reads from cache instead of calling yfinance at runtime.
"""
import json
from src.trading_ensemble.data.comet_screener import get_top_stocks
from src.trading_ensemble.data.dual_source import fetch_yf_sentiment, fetch_yf_fundamentals

CACHE_FILE = "data/pre_market_cache.json"

def main():
    stocks = get_top_stocks()
    cache  = {}
    total  = len(stocks)
    for i, (_, row) in enumerate(stocks.iterrows(), 1):
        symbol = row["symbol"].replace(".NS", "")
        print(f"  [{i}/{total}] Fetching {symbol}...")
        cache[symbol] = {
            **fetch_yf_sentiment(symbol),
            **fetch_yf_fundamentals(symbol),
        }
    import os; os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Cache saved -> {CACHE_FILE}")

if __name__ == "__main__":
    main()
'''
open('pre_market_fetch.py','w').write(content)
print('  Done.')
PYEOF

echo ""
echo "=== All 9 patches applied ==="
echo "Next: pip install -r requirements.txt"
echo "      python pre_market_fetch.py   (run at 08:45 AM daily)"
echo "      python run_swing_backtest.py swing"
