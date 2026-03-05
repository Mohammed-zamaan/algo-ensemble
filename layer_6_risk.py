# layer_6_risk.py
# HASH-004: Conviction overlay | HASH-005: LIMIT orders | HASH-006: Drawdown throttle
import os
import pandas as pd
from dotenv import load_dotenv
from state_manager import load_state, get_drawdown_risk_multiplier, log_equity_snapshot
from market_regime import detect_regime, regime_multiplier
from src.trading_ensemble.data.sheets_client import read_master_universe, get_conviction_for_symbol, write_layer_output

load_dotenv()

TOTAL_CAPITAL         = float(os.getenv("TOTAL_CAPITAL", 500000))
RISK_PER_TRADE_PCT    = 0.02
MAX_CAPITAL_PER_TRADE = 0.20
MAX_POSITIONS         = 5
MAX_DAILY_RISK_PCT    = 0.06
MIN_RR_RATIO          = 1.5
ONLY_CONFIRMED        = True
LIMIT_BUFFER_PCT      = 0.001
PAPER_TRADE           = os.getenv("PAPER_TRADE", "true").lower() == "true"
CONVICTION_MULT       = {1: 0.9, 2: 1.0, 3: 1.1}
SIGNALS_FILE          = "trade_signals.csv"
ORDERS_FILE           = "trade_orders.csv"


def calculate_position_size(entry, stop_loss, capital, c_mult, dd_mult):
    combined_mult  = c_mult * dd_mult
    risk_amount    = capital * RISK_PER_TRADE_PCT * combined_mult
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return {"quantity": 0, "reason": "Invalid SL"}
    qty_by_risk = int(risk_amount / risk_per_share)
    qty_by_cap  = int((capital * MAX_CAPITAL_PER_TRADE) / entry)
    quantity    = min(qty_by_risk, qty_by_cap)
    if quantity <= 0:
        return {"quantity": 0, "reason": "Position rounds to 0"}
    capital_used = round(quantity * entry, 2)
    max_loss     = round(quantity * risk_per_share, 2)
    return {
        "quantity": quantity, "capital_used": capital_used,
        "capital_pct": round(capital_used/capital*100, 2),
        "max_loss": max_loss, "max_loss_pct": round(max_loss/capital*100, 2),
        "reason": "OK",
    }


def apply_risk_filters(df, dd_mult):
    print(f"  Input: {len(df)} signal(s)")
    if ONLY_CONFIRMED:
        before = len(df)
        df = df[df["BREAKOUT_15M"] == True].copy()
        print(f"    - {before-len(df)} skipped (no 15m confirmation)")
    before = len(df)
    df = df[df["RR_RATIO"] >= MIN_RR_RATIO].copy()
    print(f"    - {before-len(df)} skipped (RR < {MIN_RR_RATIO})")
    if dd_mult == 0.0:
        print("    STOP: Drawdown > 15% — all trades blocked")
        return pd.DataFrame()
    if len(df) > MAX_POSITIONS:
        df = df.nlargest(MAX_POSITIONS, "COMPOSITE_SCORE")
        print(f"    - Capped at {MAX_POSITIONS} positions")
    print(f"  After filters: {len(df)} signal(s)")
    return df


def build_orders(df, universe_df, dd_mult):
    orders = []
    total_capital_used = 0.0
    total_risk_used    = 0.0
    print("\n" + "-"*65)
    print(f"  CAPITAL  : INR {TOTAL_CAPITAL:,.0f}")
    print(f"  DD MULT  : {dd_mult:.2f}x")
    print(f"  MODE     : {'PAPER' if PAPER_TRADE else 'LIVE'}")
    print("-"*65)
    for _, row in df.iterrows():
        symbol = row["symbol"]
        entry  = float(row["ENTRY_PRICE"])
        target = float(row["TARGET_PRICE"])
        sl     = float(row["STOP_LOSS"])
        rr     = float(row["RR_RATIO"])
        source_type, conviction, c_mult = get_conviction_for_symbol(symbol, universe_df)
        limit_price = round(entry * (1 + LIMIT_BUFFER_PCT), 2)
        sizing = calculate_position_size(limit_price, sl, TOTAL_CAPITAL, c_mult, dd_mult)
        if sizing["quantity"] == 0:
            print(f"  [SKIP] {symbol:<20} {sizing['reason']}")
            continue
        if total_risk_used + sizing["max_loss"] > TOTAL_CAPITAL * MAX_DAILY_RISK_PCT:
            print(f"  [SKIP] {symbol:<20} Daily risk cap reached")
            continue
        total_capital_used += sizing["capital_used"]
        total_risk_used    += sizing["max_loss"]
        orders.append({
            "symbol": symbol, "MODE": row.get("MODE","INTRADAY"),
            "COMPOSITE_SCORE": float(row.get("COMPOSITE_SCORE",0)),
            "ENTRY_PRICE": entry, "LIMIT_PRICE": limit_price,
            "TARGET_PRICE": target, "STOP_LOSS": sl,
            "QUANTITY": sizing["quantity"], "CAPITAL_USED": sizing["capital_used"],
            "CAPITAL_PCT": sizing["capital_pct"], "MAX_LOSS": sizing["max_loss"],
            "MAX_LOSS_PCT": sizing["max_loss_pct"], "RR_RATIO": rr,
            "SIGNAL_STRENGTH": row.get("SIGNAL_STRENGTH","CONFIRMED"),
            "source_type": source_type, "conviction": conviction,
            "CONVICTION_MULT": c_mult, "DD_MULT": dd_mult,
            "ORDER_TYPE": "LIMIT", "PRODUCT_TYPE": row.get("PRODUCT_TYPE","MIS"),
            "EXCHANGE": "NSE", "TRANSACTION": "BUY",
            "STATUS": "PAPER" if PAPER_TRADE else "PENDING",
        })
        print(f"  [OK] {symbol:<20} Limit=INR{limit_price:<8} Qty={sizing['quantity']:<5} Conv={conviction}({source_type})")
    print("-"*65)
    print(f"  ORDERS: {len(orders)} | Capital: INR{total_capital_used:,.0f} | Risk: INR{total_risk_used:,.0f}")
    return pd.DataFrame(orders)


def main():
    print("\n[LAYER 6] Risk Management & Position Sizing")
    print("="*65)
    state   = load_state()
    dd_mult = get_drawdown_risk_multiplier(state)

    regime_info = detect_regime()
    regime = regime_info['regime']
    regime_mult = regime_multiplier(regime)
    state['regime'] = regime

    combined_mult = dd_mult * regime_mult
    print(f"  Capital  : INR {state['equity']['current_capital']:,.0f}")
    print(f"  Drawdown : {state['equity']['current_dd_pct']:.2f}%")
    print(f"  DD Mult  : {dd_mult:.2f}x")
    print(f"  Regime   : {regime} | VIX={regime_info.get('india_vix')} | NIFTY={regime_info.get('nifty_trend')} | BANKNIFTY={regime_info.get('banknifty_trend')}")
    print(f"  Reg Mult : {regime_mult:.2f}x")
    print(f"  COMBINED : {combined_mult:.2f}x")
    if dd_mult == 0.0:
        print("  STOP: Trading halted — drawdown > 15%")
        pd.DataFrame().to_csv(ORDERS_FILE, index=False)
        return
    if not os.path.exists(SIGNALS_FILE):
        print(f"  [ERROR] {SIGNALS_FILE} not found — run layer_5 first")
        return
    signals = pd.read_csv(SIGNALS_FILE)
    if signals.empty:
        print("  [INFO] No signals — no orders today")
        pd.DataFrame().to_csv(ORDERS_FILE, index=False)
        return
    try:
        universe_df = read_master_universe()
    except Exception as e:
        print(f"  [WARN] Watchlist error: {e} — defaulting conviction to MANUAL/2")
        universe_df = pd.DataFrame(columns=["symbol","source_type","conviction","conviction_mult"])
    filtered = apply_risk_filters(signals, combined_mult)
    if filtered.empty:
        pd.DataFrame().to_csv(ORDERS_FILE, index=False)
        return
    orders_df = build_orders(filtered, universe_df, combined_mult)
    if orders_df.empty:
        pd.DataFrame().to_csv(ORDERS_FILE, index=False)
        return
    orders_df.to_csv(ORDERS_FILE, index=False)
    print(f"  [SAVED] {ORDERS_FILE}")
    write_layer_output("Layer6_Orders", orders_df)
    log_equity_snapshot(state)
    print("\n[LAYER 6] Complete")


if __name__ == "__main__":
    main()
