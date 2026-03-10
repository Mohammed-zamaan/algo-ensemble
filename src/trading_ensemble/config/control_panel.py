from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ControlPanel:
    system_trading_enabled: bool = True
    paper_trade: bool = True
    enabled_modes: str = "INTRADAY,SWING,POSITIONAL"
    default_mode: str = "INTRADAY"
    max_new_orders_per_run: int = 4
    max_intraday_orders: int = 2
    max_swing_orders: int = 2
    max_positional_orders: int = 1
    risk_multiplier: float = 1.0
    cash_buffer_pct: float = 0.10
    max_gross_exposure_pct: float = 0.90
    max_single_position_pct: float = 0.12
    max_sector_exposure_pct: float = 0.25
    force_regime: str = ""
    pause_new_entries: bool = False
    force_exit_all: bool = False
    write_outputs_to_sheets: bool = True
    near_trigger_alerts_enabled: bool = True
    near_trigger_min_score: float = 95.0
    near_trigger_max_breakout_gap_pct: float = 1.0
    max_near_trigger_alerts: int = 10
    trend_risk_multiplier: float = 1.15
    chop_risk_multiplier: float = 0.60
    high_vol_risk_multiplier: float = 0.50
    trend_max_new_orders: int = 4
    chop_max_new_orders: int = 2
    high_vol_max_new_orders: int = 1

    # Phase 20A — trigger monitor controls
    trigger_monitor_enabled: bool = True
    trigger_poll_seconds: int = 60
    market_open_time: str = "09:15"
    new_entry_cutoff_time: str = "15:00"
    max_promotions_per_cycle: int = 1
    dedupe_open_positions: bool = True
    dedupe_pending_orders: bool = True


def _to_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _to_int(value, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def _to_float(value, default: float) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _to_str(value, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _records_to_dict(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {}

    cols = [str(c).strip().lower() for c in df.columns]
    df = df.copy()
    df.columns = cols

    required = {"key", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ControlPanel tab missing required columns: {sorted(missing)}")

    result: dict[str, object] = {}
    for _, row in df.iterrows():
        key = str(row["key"]).strip()
        if not key:
            continue
        result[key.lower()] = row["value"]
    return result


def build_control_panel(raw: dict[str, object]) -> ControlPanel:
    return ControlPanel(
        system_trading_enabled=_to_bool(raw.get("system_trading_enabled"), True),
        paper_trade=_to_bool(raw.get("paper_trade"), True),
        enabled_modes=_to_str(raw.get("enabled_modes"), "INTRADAY,SWING,POSITIONAL").upper(),
        default_mode=_to_str(raw.get("default_mode"), "INTRADAY").upper(),
        max_new_orders_per_run=_to_int(raw.get("max_new_orders_per_run"), 4),
        max_intraday_orders=_to_int(raw.get("max_intraday_orders"), 2),
        max_swing_orders=_to_int(raw.get("max_swing_orders"), 2),
        max_positional_orders=_to_int(raw.get("max_positional_orders"), 1),
        risk_multiplier=_to_float(raw.get("risk_multiplier"), 1.0),
        cash_buffer_pct=_to_float(raw.get("cash_buffer_pct"), 0.10),
        max_gross_exposure_pct=_to_float(raw.get("max_gross_exposure_pct"), 0.90),
        max_single_position_pct=_to_float(raw.get("max_single_position_pct"), 0.12),
        max_sector_exposure_pct=_to_float(raw.get("max_sector_exposure_pct"), 0.25),
        force_regime=_to_str(raw.get("force_regime"), "").upper(),
        pause_new_entries=_to_bool(raw.get("pause_new_entries"), False),
        force_exit_all=_to_bool(raw.get("force_exit_all"), False),
        write_outputs_to_sheets=_to_bool(raw.get("write_outputs_to_sheets"), True),
        near_trigger_alerts_enabled=_to_bool(raw.get("near_trigger_alerts_enabled"), True),
        near_trigger_min_score=_to_float(raw.get("near_trigger_min_score"), 95.0),
        near_trigger_max_breakout_gap_pct=_to_float(raw.get("near_trigger_max_breakout_gap_pct"), 1.0),
        max_near_trigger_alerts=_to_int(raw.get("max_near_trigger_alerts"), 10),
        trend_risk_multiplier=_to_float(raw.get("trend_risk_multiplier"), 1.15),
        chop_risk_multiplier=_to_float(raw.get("chop_risk_multiplier"), 0.60),
        high_vol_risk_multiplier=_to_float(raw.get("high_vol_risk_multiplier"), 0.50),
        trend_max_new_orders=_to_int(raw.get("trend_max_new_orders"), 4),
        chop_max_new_orders=_to_int(raw.get("chop_max_new_orders"), 2),
        high_vol_max_new_orders=_to_int(raw.get("high_vol_max_new_orders"), 1),
    

        # Phase 20A — trigger monitor controls
        trigger_monitor_enabled=_to_bool(raw.get("trigger_monitor_enabled"), True),
        trigger_poll_seconds=_to_int(raw.get("trigger_poll_seconds"), 60),
        market_open_time=_to_str(raw.get("market_open_time"), "09:15"),
        new_entry_cutoff_time=_to_str(raw.get("new_entry_cutoff_time"), "15:00"),
        max_promotions_per_cycle=_to_int(raw.get("max_promotions_per_cycle"), 1),
        dedupe_open_positions=_to_bool(raw.get("dedupe_open_positions"), True),
        dedupe_pending_orders=_to_bool(raw.get("dedupe_pending_orders"), True),
    )


def load_control_panel_csv(path: str) -> ControlPanel:
    df = pd.read_csv(path)
    raw = _records_to_dict(df)
    return build_control_panel(raw)


def load_control_panel_google_service_account(
    spreadsheet_id: str,
    worksheet_name: str,
    service_account_file: str,
) -> ControlPanel:
    import gspread

    gc = gspread.service_account(filename=service_account_file)
    sheet = gc.open_by_key(spreadsheet_id)
    worksheet = sheet.worksheet(worksheet_name)

    records = worksheet.get_all_records()
    df = pd.DataFrame(records)
    raw = _records_to_dict(df)
    return build_control_panel(raw)


def load_control_panel(settings) -> ControlPanel:
    source = settings.watchlist_source.strip().lower()

    if source == "google_sheets":
        return load_control_panel_google_service_account(
            spreadsheet_id=settings.watchlist_gsheet_id,
            worksheet_name="ControlPanel",
            service_account_file=settings.google_service_account_json,
        )

    return ControlPanel()