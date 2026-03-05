import os, json
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

STATE_FILE     = "state/positions_state.json"
EQUITY_FILE    = "state/equity_curve.csv"
TRADE_LOG_FILE = "state/trade_log.csv"

DEFAULT_STATE = {
    "last_updated": None,
    "trading_date": None,
    "open_positions": {},
    "closed_today": [],
    "equity": {
        "initial_capital": float(os.getenv("TOTAL_CAPITAL", 500000)),
        "current_capital": float(os.getenv("TOTAL_CAPITAL", 500000)),
        "peak_capital":    float(os.getenv("TOTAL_CAPITAL", 500000)),
        "current_dd_pct":  0.0,
        "total_pnl":       0.0,
        "total_trades":    0,
        "winning_trades":  0,
    },
    "regime": "NORMAL",
    "paper_trade": os.getenv("PAPER_TRADE", "true").lower() == "true",
    "session_status": "IDLE",
}

def load_state() -> dict:
    os.makedirs("state", exist_ok=True)
    if not os.path.exists(STATE_FILE):
        save_state(DEFAULT_STATE.copy())
        return DEFAULT_STATE.copy()
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        if "equity" not in state:
            raise KeyError("equity missing")
        return state
    except Exception:
        save_state(DEFAULT_STATE.copy())
        return DEFAULT_STATE.copy()

def save_state(state: dict):
    os.makedirs("state", exist_ok=True)
    state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["trading_date"] = str(date.today())
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def update_equity(state: dict, realised_pnl: float = 0.0) -> dict:
    eq = state["equity"]
    eq["current_capital"] += realised_pnl
    eq["total_pnl"]       += realised_pnl
    eq["total_trades"]    += 1
    if realised_pnl > 0:
        eq["winning_trades"] += 1
    if eq["current_capital"] > eq["peak_capital"]:
        eq["peak_capital"] = eq["current_capital"]
    dd = (eq["peak_capital"] - eq["current_capital"]) / eq["peak_capital"] * 100
    eq["current_dd_pct"] = round(dd, 4)
    state["equity"] = eq
    return state

def get_drawdown_risk_multiplier(state: dict) -> float:
    dd = float(state["equity"].get("current_dd_pct", 0.0))
    if dd < 5.0:
        return 1.0
    if dd < 10.0:
        return 0.5
    if dd < 15.0:
        return 0.25
    return 0.0

def log_trade(trade: dict):
    os.makedirs("state", exist_ok=True)
    df_new = pd.DataFrame([trade])
    if os.path.exists(TRADE_LOG_FILE) and os.path.getsize(TRADE_LOG_FILE) > 0:
        df = pd.concat([pd.read_csv(TRADE_LOG_FILE), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(TRADE_LOG_FILE, index=False)

def add_open_position(state: dict, trade: dict) -> dict:
    symbol = trade["symbol"]
    state["open_positions"][symbol] = {
        "symbol":      symbol,
        "entry_price": float(trade["ENTRY_PRICE"]),
        "stop_loss":   float(trade["STOP_LOSS"]),
        "target":      float(trade["TARGET_PRICE"]),
        "quantity":    int(trade["QUANTITY"]),
        "entry_time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode":        trade.get("MODE", "INTRADAY"),
        "product":     trade.get("PRODUCT_TYPE", "MIS"),
        "order_id":    trade.get("order_id", "PAPER"),
    }
    save_state(state)
    return state

def close_position(state: dict, symbol: str, exit_price: float, reason: str) -> dict:
    if symbol not in state["open_positions"]:
        return state
    pos = state["open_positions"].pop(symbol)
    pnl = (float(exit_price) - float(pos["entry_price"])) * int(pos["quantity"])
    closed = {
        **pos,
        "exit_price": float(exit_price),
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exit_reason": reason,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / (pos["entry_price"] * pos["quantity"]) * 100, 4),
    }
    state["closed_today"].append(closed)
    state = update_equity(state, pnl)
    log_trade(closed)
    save_state(state)
    return state

def log_equity_snapshot(state: dict):
    os.makedirs("state", exist_ok=True)
    snap = {
        "date": str(date.today()),
        "current_capital": state["equity"]["current_capital"],
        "peak_capital": state["equity"]["peak_capital"],
        "current_dd_pct": state["equity"]["current_dd_pct"],
        "total_pnl": state["equity"]["total_pnl"],
        "total_trades": state["equity"]["total_trades"],
        "winning_trades": state["equity"]["winning_trades"],
        "regime": state.get("regime", "NORMAL"),
    }
    df_new = pd.DataFrame([snap])
    if os.path.exists(EQUITY_FILE) and os.path.getsize(EQUITY_FILE) > 0:
        df = pd.concat([pd.read_csv(EQUITY_FILE), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(EQUITY_FILE, index=False)

def print_summary(state: dict):
    eq = state["equity"]
    print("\n" + "="*55)
    print("  STATE SUMMARY")
    print("="*55)
    print(f"  Date          : {state.get('trading_date')}")
    print(f"  Mode          : {'PAPER' if state.get('paper_trade') else 'LIVE'}")
    print(f"  Regime        : {state.get('regime','NORMAL')}")
    print(f"  Capital       : ₹{eq['current_capital']:,.0f}")
    print(f"  Peak Capital  : ₹{eq['peak_capital']:,.0f}")
    print(f"  Total PnL     : ₹{eq['total_pnl']:+,.0f}")
    print(f"  Drawdown      : {eq['current_dd_pct']:.2f}%")
    print(f"  Total Trades  : {eq['total_trades']}")
    win_rate = eq["winning_trades"]/eq["total_trades"]*100 if eq["total_trades"] else 0
    print(f"  Win Rate      : {win_rate:.1f}%")
    print(f"  Open Positions: {len(state.get('open_positions',{}))}")
    print("="*55)
