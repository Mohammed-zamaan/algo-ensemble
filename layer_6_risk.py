# layer_6_risk.py
# LAYER 6 — Risk Management & Position Sizing

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")

# ─────────────────────────────────────────────
# PORTFOLIO CONFIG
# ─────────────────────────────────────────────
TOTAL_CAPITAL         = float(os.getenv("TOTAL_CAPITAL", 100000))
RISK_PER_TRADE_PCT    = 0.02
MAX_CAPITAL_PER_TRADE = 0.20
MAX_POSITIONS         = 5
MAX_DAILY_RISK_PCT    = 0.06
MIN_RR_RATIO          = 1.5
ONLY_CONFIRMED        = True

SIGNALS_FILE = "trade_signals.csv"
ORDERS_FILE  = "trade_orders.csv"
# ─────────────────────────────────────────────


def calculate_position_size(entry, stop_loss, capital):
    risk_amount    = capital * RISK_PER_TRADE_PCT
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return {"quantity": 0, "reason": "Invalid SL"}
    qty_by_risk    = int(risk_amount / risk_per_share)
    qty_by_cap     = int((capital * MAX_CAPITAL_PER_TRADE) / entry)
    quantity       = min(qty_by_risk, qty_by_cap)
    if quantity <= 0:
        return {"quantity": 0, "reason": "Position size rounds to 0"}
    capital_used = round(quantity * entry, 2)
    max_loss     = round(quantity * risk_per_share, 2)
    return {
        "quantity":     quantity,
        "capital_used": capital_used,
        "capital_pct":  round(capital_used / capital * 100, 2),
        "max_loss":     max_loss,
        "max_loss_pct": round(max_loss / capital * 100, 2),
        "reason":       "OK",
    }


def apply_risk_filters(df):
    print(f"  Input: {len(df)} signal(s)")
    if ONLY_CONFIRMED:
        before = len(df)
        df = df[df["BREAKOUT_15M"] == True].copy()
        print(f"    - {before - len(df)} skipped (no 15m confirmation)")
    before = len(df)
    df = df[df["RR_RATIO"] >= MIN_RR_RATIO].copy()
    print(f"    - {before - len(df)} skipped (RR < {MIN_RR_RATIO})")
    if len(df) > MAX_POSITIONS:
        df = df.head(MAX_POSITIONS)
        print(f"    - Capped at {MAX_POSITIONS} max positions")
    print(f"  After filters: {len(df)} signal(s) remain")
    return df


def build_orders(df):
    orders = []
    total_capital_used = 0
    total_risk_used    = 0

    print("\n" + "-"*65)
    print(f"  TOTAL CAPITAL : INR {TOTAL_CAPITAL:,.0f}")
    print(f"  RISK PER TRADE: {RISK_PER_TRADE_PCT*100:.1f}%")
    print(f"  MAX POSITIONS : {MAX_POSITIONS}")
    print("-"*65)

    for _, row in df.iterrows():
        symbol = row["symbol"]
        entry  = float(row["ENTRY_PRICE"])
        target = float(row["TARGET_PRICE"])
        sl     = float(row["STOP_LOSS"])
        rr     = float(row["RR_RATIO"])

        sizing = calculate_position_size(entry, sl, TOTAL_CAPITAL)
        if sizing["quantity"] == 0:
            print(f"  [SKIP] {symbol:<20} {sizing['reason']}")
            continue

        if total_risk_used + sizing["max_loss"] > TOTAL_CAPITAL * MAX_DAILY_RISK_PCT:
            print(f"  [SKIP] {symbol:<20} Daily risk cap reached")
            continue

        total_capital_used += sizing["capital_used"]
        total_risk_used    += sizing["max_loss"]

        order = {
            "symbol":           symbol,
            "COMPOSITE_SCORE":  float(row["COMPOSITE_SCORE"]),
            "ENTRY_PRICE":      entry,
            "TARGET_PRICE":     target,
            "STOP_LOSS":        sl,
            "QUANTITY":         sizing["quantity"],
            "CAPITAL_USED":     sizing["capital_used"],
            "CAPITAL_PCT":      sizing["capital_pct"],
            "MAX_LOSS":         sizing["max_loss"],
            "MAX_LOSS_PCT":     sizing["max_loss_pct"],
            "RR_RATIO":         rr,
            "SIGNAL_STRENGTH":  row["SIGNAL_STRENGTH"],
            "ORDER_TYPE":       "LIMIT" if row["SIGNAL_STRENGTH"] == "STRONG" else "MARKET",
            "PRODUCT_TYPE":     "INTRADAY",
            "EXCHANGE":         "NSE",
            "TRANSACTION":      "BUY",
            "STATUS":           "PENDING",
        }

        print(f"  [ORDER] {symbol:<20} Qty={sizing['quantity']}  "
              f"Entry={entry:.2f}  Target={target:.2f}  SL={sl:.2f}  "
              f"Capital=INR {sizing['capital_used']:,.0f} ({sizing['capital_pct']}%)  "
              f"MaxLoss=INR {sizing['max_loss']:,.0f}  RR={rr}")
        orders.append(order)

    print("-"*65)
    if orders:
        print(f"  TOTAL CAPITAL DEPLOYED : INR {total_capital_used:,.0f} "
              f"({total_capital_used/TOTAL_CAPITAL*100:.1f}%)")
        print(f"  TOTAL MAX RISK TODAY   : INR {total_risk_used:,.0f} "
              f"({total_risk_used/TOTAL_CAPITAL*100:.1f}%)")
        print(f"  ORDERS READY           : {len(orders)}")
    else:
        print("  No orders generated.")

    return pd.DataFrame(orders)


def run_risk_management(signals_file=SIGNALS_FILE):
    if not os.path.exists(signals_file):
        print(f"[LAYER 6] ERROR: {signals_file} not found.")
        print("[LAYER 6] Run layer_5_signals.py first.")
        return pd.DataFrame()

    signals = pd.read_csv(signals_file)
    if signals.empty:
        print("[LAYER 6] No signals from Layer 5.")
        return pd.DataFrame()

    print("\n" + "="*65)
    print("  LAYER 6 — RISK MANAGEMENT & POSITION SIZING")
    print("="*65)

    filtered = apply_risk_filters(signals)
    if filtered.empty:
        print("  No signals passed risk filters.")
        return pd.DataFrame()

    return build_orders(filtered)


def save_orders(df, path=ORDERS_FILE):
    if df.empty:
        print("\n  [INFO] No orders to save.")
        return
    df.to_csv(path, index=False)
    print(f"\n  [SAVED] {len(df)} order(s) -> {path}")
    print("  Ready for Layer 7 — SmartAPI Order Execution")


if __name__ == "__main__":
    print("[LAYER 6] Starting risk management...")
    orders = run_risk_management(SIGNALS_FILE)
    save_orders(orders, ORDERS_FILE)
