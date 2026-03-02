# swing_layer_9_backtest.py
# Backtests the swing system over the past 6 months on DAILY bars

import re
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.trading_ensemble.data.comet_screener import get_top_stocks
from src.trading_ensemble.data.dual_source import to_yf_ticker


# Strategy params (must match swing_layer_5 & swing_layer_6)
DONCHIAN_PERIOD       = 20
VOLUME_MULTIPLIER     = 1.5
ATR_PERIOD            = 14
ATR_SL_MULT           = 2.0
ATR_TARGET_MULT       = 4.0
MIN_RR                = 2.0
RISK_PER_TRADE        = 0.02
CAPITAL               = 500000
MAX_POSITIONS         = 6
MIN_SCORE             = 60
LOOKBACK_MONTHS       = 6

