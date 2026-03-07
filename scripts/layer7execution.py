# layer_7_execution.py
# HASH-008: Order Execution
import os, time, pyotp, pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from state_manager import load_state, save_state, add_open_position, close_position, log_equity_snapshot, print_summary
load_dotenv()

PAPER_TRADE     = os.getenv("PAPER_TRADE","true").lower() == "true"
ANGEL_API_KEY   = os.getenv("ANGEL_API_KEY","")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_CODE","")
ANGEL_PASSWORD  = os.getenv("ANGEL_PIN","")
ANGEL_TOTP_KEY  = os.getenv("ANGEL_TOTP_SECRET","")
ORDERS_FILE     = "trade_orders.csv"
PAPER_TRADES    = "state/paper_trades.csv"
EXECUTION_LOG   = "state/execution_log.csv"
MAX_RETRIES     = 3
_token_cache    = {}

def get_angel_session():
    from SmartApi import SmartConnect

    totp = pyotp.TOTP(ANGEL_TOTP_KEY).now()

    obj = SmartConnect(api_key=ANGEL_API_KEY)

    data = obj.generateSession(
        ANGEL_CLIENT_ID,
        ANGEL_PASSWORD,
        totp
    )

    if not data["status"]:
        raise ConnectionError(
            f"Angel login failed: {data.get('message')}"
        )

    print(f"[AUTH] Angel session established")

    return obj

def get_symbol_token(obj, symbol):
    if symbol in _token_cache:
        return _token_cache[symbol]
    try:
        nse = symbol.replace("-EQ","")
        data = obj.searchScrip("NSE", nse)
        if data["status"] and data["data"]:
            token = data["data"][0]["symboltoken"]
            _token_cache[symbol] = token
            return token
    except Exception as e:
        print(f"  [TOKEN] Failed for {symbol}: {e}")
    return None

def place_bracket_order(obj, order):
    symbol = order["symbol"]
    token  = get_symbol_token(obj, symbol)
    if not token:
        return {"status":"FAILED","reason":f"No token for {symbol}"}
    if order.get("PRODUCT_TYPE","MIS") == "CNC":
        return place_regular_limit_order(obj, order, token)
    try:
        params = {
            "variety":"BO","tradingsymbol":symbol.replace("-EQ","") + "-EQ",
            "symboltoken":token,"transactiontype":"BUY","exchange":"NSE",
            "ordertype":"LIMIT","producttype":"MIS","duration":"DAY",
            "price":str(order["LIMIT_PRICE"]),
            "squareoff":str(round(float(order["TARGET_PRICE"])-float(order["LIMIT_PRICE"]),2)),
            "stoploss":str(round(float(order["LIMIT_PRICE"])-float(order["STOP_LOSS"]),2)),
            "quantity":str(int(order["QUANTITY"])),
        }
        r = obj.placeOrder(params)
        if r["status"]:
            oid = r["data"]["orderid"]
            print(f"  [ORDER] OK {symbol} | ID:{oid} | Entry:INR{order['LIMIT_PRICE']} SL:INR{order['STOP_LOSS']} TGT:INR{order['TARGET_PRICE']} Qty:{order['QUANTITY']}")
            return {"status":"PLACED","order_id":oid,"reason":"OK"}
        return {"status":"FAILED","reason":r.get("message","Unknown")}
    except Exception as e:
        return {"status":"ERROR","reason":str(e)}

def place_regular_limit_order(obj, order, token):
    try:
        params = {
            "variety":"NORMAL","tradingsymbol":order["symbol"].replace("-EQ","") + "-EQ",
            "symboltoken":token,"transactiontype":"BUY","exchange":"NSE",
            "ordertype":"LIMIT","producttype":"CNC","duration":"DAY",
            "price":str(order["LIMIT_PRICE"]),"quantity":str(int(order["QUANTITY"])),
        }
        r = obj.placeOrder(params)
        if r["status"]:
            oid = r["data"]["orderid"]
            print(f"  [ORDER] CNC OK {order['symbol']} | ID:{oid}")
            place_gtt(obj, order, token)
            return {"status":"PLACED","order_id":oid,"reason":"OK"}
        return {"status":"FAILED","reason":r.get("message","Unknown")}
    except Exception as e:
        return {"status":"ERROR","reason":str(e)}

def place_gtt(obj, order, token):
    try:
        for price, label in [(order["STOP_LOSS"],"SL"),(order["TARGET_PRICE"],"TGT")]:
            p = str(price)
            obj.gttCreateRule({"type":"SINGLE",
                "rules":[{"tradingsymbol":order["symbol"].replace("-EQ","") + "-EQ",
                    "symboltoken":token,"exchange":"NSE","producttype":"CNC",
                    "transactiontype":"SELL","price":p,"qty":str(int(order["QUANTITY"])),"triggerPrice":p,"disclosedQty":"0"}],
                "condition":{"exchange":"NSE","tradingsymbol":order["symbol"].replace("-EQ",""),
                    "symboltoken":token,"price":p,"trigger_price":p}})
            print(f"  [GTT]  {label} set at INR{price}")
    except Exception as e:
        print(f"  [GTT]  WARNING: {e} — set manually: SL=INR{order['STOP_LOSS']} TGT=INR{order['TARGET_PRICE']}")

def place_with_retry(obj, order):
    for attempt in range(1, MAX_RETRIES+1):
        result = place_bracket_order(obj, order)
        if result["status"] in ("PLACED","FAILED"):
            return result
        print(f"  [RETRY] {attempt}/{MAX_RETRIES}: {result['reason']}")
        time.sleep(2*attempt)
    return {"status":"FAILED","reason":f"Failed after {MAX_RETRIES} retries"}

def log_execution(order, result):
    os.makedirs("state", exist_ok=True)
    rec = {"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "symbol":order["symbol"],"status":result["status"],
           "order_id":result.get("order_id",""),"reason":result.get("reason",""),
           "entry":order.get("LIMIT_PRICE",""),"sl":order.get("STOP_LOSS",""),
           "target":order.get("TARGET_PRICE",""),"qty":order.get("QUANTITY",""),
           "conviction":order.get("conviction",""),"source":order.get("source_type",""),
           "mode":order.get("MODE",""),"paper":PAPER_TRADE}
    df = pd.DataFrame([rec])
    if os.path.exists(EXECUTION_LOG):
        df = pd.concat([pd.read_csv(EXECUTION_LOG), df], ignore_index=True)
    df.to_csv(EXECUTION_LOG, index=False)

def paper_execute(orders_df, state):
    print(f"\n  [PAPER] Simulating {len(orders_df)} order(s) — NOT sent to broker")
    print("-"*65)
    os.makedirs("state", exist_ok=True)
    records, now = [], datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _, order in orders_df.iterrows():
        symbol = order["symbol"]
        fill   = float(order["LIMIT_PRICE"])
        rec    = {**order.to_dict(),"STATUS":"PAPER_FILLED","FILL_PRICE":fill,
                  "FILL_TIME":now,"order_id":f"PAPER-{date.today()}-{symbol}"}
        records.append(rec)
        state = add_open_position(state, {"symbol":symbol,"ENTRY_PRICE":fill,
            "STOP_LOSS":float(order["STOP_LOSS"]),"TARGET_PRICE":float(order["TARGET_PRICE"]),
            "QUANTITY":int(order["QUANTITY"]),"MODE":str(order.get("MODE","INTRADAY")),
            "PRODUCT_TYPE":str(order.get("PRODUCT_TYPE","MIS")),
            "order_id":f"PAPER-{date.today()}-{symbol}"})
        print(f"  [PAPER] {symbol:<20} Entry=INR{fill:<8} SL=INR{order['STOP_LOSS']:<8} TGT=INR{order['TARGET_PRICE']:<8} Qty={int(order['QUANTITY']):<5} Conv={order.get('conviction',2)}")
    df = pd.DataFrame(records)
    if os.path.exists(PAPER_TRADES):
        df = pd.concat([pd.read_csv(PAPER_TRADES), df], ignore_index=True)
    df.to_csv(PAPER_TRADES, index=False)
    print(f"\n  [PAPER] Saved to {PAPER_TRADES}")
    return state

def check_paper_exits(state):
    if not state["open_positions"]: return state
    print("\n  [PAPER] Checking exits for open positions...")
    try:
        import yfinance as yf
        for symbol, pos in list(state["open_positions"].items()):
            hist = yf.Ticker(symbol.replace("-EQ",".NS")).history(period="1d",interval="15m")
            if hist.empty:
                print(f"  [PAPER] No data for {symbol} — carry forward")
                continue
            high, low = hist["High"].max(), hist["Low"].min()
            if low <= pos["stop_loss"]:
                state = close_position(state, symbol, pos["stop_loss"], "PAPER_SL_HIT")
            elif high >= pos["target"]:
                state = close_position(state, symbol, pos["target"], "PAPER_TARGET_HIT")
            elif pos.get("product","MIS") == "MIS":
                state = close_position(state, symbol, float(hist["Close"].iloc[-1]), "PAPER_EOD_SQUAREOFF")
    except Exception as e:
        print(f"  [PAPER] Exit check error: {e}")
    return state

def main():
    print(f"\n[LAYER 7] Order Execution")
    print("="*65)
    print(f"  Mode : {'PAPER' if PAPER_TRADE else 'LIVE'}")
    print(f"  Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)
    if not os.path.exists(ORDERS_FILE):
        print(f"  [ERROR] {ORDERS_FILE} not found — run layer_6 first")
        return
    orders_df = pd.read_csv(ORDERS_FILE)
    if orders_df.empty:
        print("  [INFO] No orders — nothing to execute")
        return
    print(f"  Orders: {len(orders_df)}")
    state = load_state()
    if PAPER_TRADE and state["open_positions"]:
        state = check_paper_exits(state)
    if PAPER_TRADE:
        state = paper_execute(orders_df, state)
    else:
        if not all([ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_KEY]):
            print("  [ERROR] Missing Angel One credentials")
            return
        obj = get_angel_session()
        placed = failed = 0
        for _, order in orders_df.iterrows():
            result = place_with_retry(obj, order.to_dict())
            log_execution(order.to_dict(), result)
            if result["status"] == "PLACED":
                state = add_open_position(state, {**order.to_dict(),"order_id":result["order_id"]})
                placed += 1
            else:
                print(f"  [FAIL] {order['symbol']}: {result['reason']}")
                failed += 1
            time.sleep(0.5)
        print(f"\n  Placed: {placed} | Failed: {failed}")
    save_state(state)
    log_equity_snapshot(state)
    print_summary(state)
    print("\n[LAYER 7] Complete")

if __name__ == "__main__":
    main()
