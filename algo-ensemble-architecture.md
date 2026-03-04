# algo-ensemble - System Architecture & Trading Algorithm
Version: 1.0 | Date: March 2, 2026 | Author: Mohammed Zamaan

1. SYSTEM OVERVIEW
algo-ensemble is a fully automated NSE equity trading system combining
Comet ML screening, Donchian breakout signals, ATR risk management,
and Angel One SmartAPI live order execution.

Mode       | Timeframe   | Hold        | Product
Intraday   | 15-min bars | Same day    | MIS
Swing      | Daily bars  | 3-10 days   | CNC
Positional | Daily bars  | Weeks-months| CNC

2. DAILY SCHEDULE
08:45 AM  python pre_market_fetch.py     -> pre_market_cache.json
09:15 AM  python layer_4_elimination.py  -> trade_candidates.csv
09:20 AM  python layer_5_signals.py      -> signals.csv
09:25 AM  python layer_6_risk.py         -> approved_trades.csv
09:30 AM  python layer_7_execution.py    -> order_log.json
15:15 PM  SmartAPI auto square-off MIS

3. TRADING ALGORITHM

3.1 Stock Universe
Source    : zamaan/stock-screener - 86 NSE EQ experiments
Scores    : COMPOSITE_SCORE, SENTIMENT_SCORE, VOLATILITY_SCORE
Threshold : >= 50 intraday/swing | >= 65 positional

3.2 Breakout Detection
Intraday  : 20-bar Donchian 15-min | Volume > 2.0x average
Swing     : 20-day Donchian daily  | Volume > 1.5x average
Positional: 55-day Donchian daily  | Volume > 1.3x | Near 52w high

3.3 Adaptive Donchian Engine
High Volatility ATR%>=2.2% : Entry 55-bar | Exit 25-bar | Trail 4xATR
Normal         1.2%-2.2%   : Entry 20-bar | Exit 10-bar | Trail 3xATR
Low Volatility ATR%<=1.2%  : Entry 20-bar | Exit 10-bar | Trail 3xATR
All computed vectorised with np.where - no Python loops.

3.4 Entry Filters (all must pass)
Session  : 09:15-15:30 NSE hours (intraday only)
SMA Trend: Close > SMA-200 (positional) / SMA-50 (swing)
ADX      : ADX >= 20-25 AND ADX rising

3.5 Signal Generation
Entry  = Close at breakout bar
SL     = Entry - (SL_Mult x ATR)
Target = Entry + (Target_Mult x ATR)
Mode       | ATR SL | ATR Target | Min R:R
Intraday   | 1.5x   | 2.5x       | 1.5
Swing      | 2.0x   | 4.0x       | 2.0
Positional | 3.0x   | 8.0x       | 2.5

3.6 Position Sizing
Risk Amount    = Capital x Risk%
Risk per Share = Entry - Stop Loss
Quantity       = floor(Risk Amount / Risk per Share)
Parameter      | Intraday | Swing   | Positional
Capital        | 500000   | 500000  | 500000
Risk per Trade | 2%       | 2%      | 3%
Max Positions  | 5        | 6       | 5
Max per Stock  | 20%      | 20%     | 20%

3.7 Trailing Stop
Activation : (hi_since - entry) >= trail_start_atr x ATR AND bars >= min_hold
Trail Stop : hi_since - (trail_mult_eff x ATR)
Final Stop : max(initial_stop, trail_stop)

3.8 Exit Conditions
Donchian Exit : Close < Donchian Lower Band
Stop Loss     : Low <= Stop Price
Target Hit    : High >= Target Price
Session End   : Auto square-off 15:15 MIS

3.9 Order Execution
1. login_from_env()                   Single TOTP session
2. load_scrip_master()                Load token map once
3. resolve_symbol_to_token_offline()  Local lookup no API call
4. placeOrderFullResponse(LIMIT MIS/CNC with squareoff + stoploss)
5. Sleep 0.5s per order
6. terminateSession()

4. BACKTEST RESULTS (Intraday 6 Months)
Total Trades  : 116
Win Rate      : 50.9%
Total Return  : +19.17%
Total P&L     : Rs 95,850
Profit Factor : 1.47
Max Drawdown  : -4.89%

5. DESIGN PRINCIPLES
1. Separation of Concerns - Data, strategy, risk, execution are independent
2. Single Source of Truth - get_top_stocks() only in comet_screener.py
3. Pre-market Caching    - Sentiment cached at 08:45, never blocks orders
4. Live vs Historical    - fetch_candles_live() vs fetch_candles_chunked()
5. Vectorised Engine     - np.where throughout, no Python loops
6. No Hardcoded Secrets  - All credentials via environment variables
