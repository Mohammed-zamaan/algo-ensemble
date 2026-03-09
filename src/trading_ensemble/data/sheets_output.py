from __future__ import annotations

from datetime import datetime

import pandas as pd
import gspread


WORKSHEET_COLUMN_ORDERS = {
    "SelectedCandidates": [
        "symbol",
        "COMPOSITE_SCORE",
        "VOLATILITY_SCORE",
        "LTP",
        "DONCHIAN_UPPER",
        "VOLUME_RATIO",
        "BREAKOUT",
        "BREAKOUT_READY",
        "VOLUME_READY",
        "SENTIMENT_SCORE",
        "RETURN_20C_PCT",
        "HIGH_52W",
        "experiment_name",
    ],
    "ConfirmedSignals": [
        "symbol",
        "MODE",
        "SIGNAL_STATUS",
        "SETUP_REASON",
        "SIGNAL_STRENGTH",
        "COMPOSITE_SCORE",
        "WATCHLIST_CONVICTION",
        "SECTOR",
        "PRIORITY",
        "ENTRY_PRICE",
        "STOP_LOSS",
        "TARGET_PRICE",
        "RR_RATIO",
        "ATR",
        "ATR_RATIO_PCT",
        "ADX",
        "VOLUME_RATIO",
        "DONCHIAN_UPPER",
        "BREAKOUT_READY",
        "VOLUME_READY",
        "ATR_READY",
        "PRODUCT_TYPE",
        "SIGNAL_TIME",
    ],
    "ApprovedOrders": [
        "symbol",
        "MODE",
        "SIGNAL_STATUS",
        "SIGNAL_STRENGTH",
        "ENTRY_PRICE",
        "LIMIT_PRICE",
        "STOP_LOSS",
        "TARGET_PRICE",
        "RR_RATIO",
        "QUANTITY",
        "CAPITAL_USED",
        "PRODUCT_TYPE",
        "ORDER_TYPE",
        "TRANSACTION",
        "EXCHANGE",
        "STATUS",
        "conviction",
        "sector",
    ],
    "ExecutionLog": [
        "symbol",
        "MODE",
        "STATUS",
        "order_id",
        "FILL_PRICE",
        "FILL_TIME",
        "ENTRY_PRICE",
        "LIMIT_PRICE",
        "STOP_LOSS",
        "TARGET_PRICE",
        "QUANTITY",
        "PRODUCT_TYPE",
        "TRANSACTION",
    ],
    "TriggerEvaluations": [
        "symbol",
        "MODE",
        "promotion_candidate",
        "promotion_reason",
        "readiness_score",
        "latest_close",
        "live_donchian_upper",
        "breakout_gap_pct",
        "latest_volume",
        "live_volume_ratio",
        "volume_gap_pct",
        "snapshot_time",
        "ENTRY_PRICE",
        "STOP_LOSS",
        "TARGET_PRICE",
        "RR_RATIO",
        "PRODUCT_TYPE",
    ],
    "PromotedSignals": [
        "symbol",
        "MODE",
        "promotion_candidate",
        "promotion_reason",
        "latest_close",
        "live_donchian_upper",
        "live_volume_ratio",
        "snapshot_time",
        "ENTRY_PRICE",
        "STOP_LOSS",
        "TARGET_PRICE",
        "RR_RATIO",
        "PRODUCT_TYPE",
    ],
    "TriggerMonitorStatus": [
        "metric",
        "value",
    ],
}


def _get_client(service_account_file: str):
    return gspread.service_account(filename=service_account_file)


def _get_or_create_worksheet(spreadsheet, title: str, rows: int = 500, cols: int = 40):
    try:
        return spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)


def _prepare_dataframe(worksheet_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    prepared = df.copy()

    preferred = WORKSHEET_COLUMN_ORDERS.get(worksheet_name, [])
    existing_preferred = [col for col in preferred if col in prepared.columns]
    remaining = [col for col in prepared.columns if col not in existing_preferred]

    prepared = prepared[existing_preferred + remaining]
    return prepared


def write_dataframe_to_sheet(
    spreadsheet_id: str,
    worksheet_name: str,
    service_account_file: str,
    df: pd.DataFrame,
) -> None:
    gc = _get_client(service_account_file)
    sh = gc.open_by_key(spreadsheet_id)
    ws = _get_or_create_worksheet(sh, worksheet_name)

    ws.clear()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prepared = _prepare_dataframe(worksheet_name, df)

    if prepared.empty:
        ws.update(
            "A1",
            [
                ["generated_at", timestamp],
                ["worksheet", worksheet_name],
                ["row_count", 0],
                [],
                ["status"],
                ["EMPTY"],
            ],
        )
        return

    values = prepared.fillna("").astype(str).values.tolist()
    header = list(prepared.columns)

    output = [
        ["generated_at", timestamp],
        ["worksheet", worksheet_name],
        ["row_count", len(prepared)],
        [],
        header,
        *values,
    ]

    ws.update("A1", output)


def maybe_write_output(settings, control_panel, worksheet_name: str, df: pd.DataFrame) -> None:
    if control_panel is None:
        return
    if not getattr(control_panel, "write_outputs_to_sheets", True):
        return
    if settings.watchlist_source.lower() != "google_sheets":
        return

    write_dataframe_to_sheet(
        spreadsheet_id=settings.watchlist_gsheet_id,
        worksheet_name=worksheet_name,
        service_account_file=settings.google_service_account_json,
        df=df,
    )