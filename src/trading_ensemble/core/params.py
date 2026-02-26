"""Strategy parameters from StageB notebook [file:35]."""
from dataclasses import dataclass

@dataclass
class StrategyParams:
    # Capital & costs
    initial_capital: float = 20000.0
    commission_pct: float = 0.0  # per side, percent

    # Long-only cash equity style
    long_only: bool = True

    # Session (NSE)
    use_session: bool = True
    session_start: str = "0915"
    session_end: str = "1530"

    # Trend filter
    use_sma: bool = True
    sma_len: int = 200

    # Donchian adaptive windows
    don_entry_base: int = 20
    don_exit_base: int = 10
    don_entry_lo: int = 20
    don_entry_hi: int = 55
    don_exit_lo: int = 10
    don_exit_hi: int = 25

    # ATR stops/trailing
    atr_len: int = 14
    stop_mult: float = 2.2
    trail_mult_lo: float = 3.0
    trail_mult_hi: float = 4.0
    min_hold_bars: int = 3
    trail_start_atr: float = 1.2

    # ADX
    use_adx: bool = True
    di_len: int = 14
    adx_smooth: int = 14
    adx_min: float = 22.0
    adx_rising: bool = True

    # ATR regime thresholds
    atrp_len: int = 50
    atrp_lo: float = 1.2
    atrp_hi: float = 2.2

    # Position sizing (no leverage)
    risk_pct: float = 1.0
    min_qty: int = 1
