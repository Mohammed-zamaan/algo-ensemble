from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


def now_ist() -> datetime:
    return datetime.now(IST)


def now_utc() -> datetime:
    return datetime.now(UTC)


def fmt_ist(dt: datetime | None = None) -> str:
    value = dt or now_ist()
    return value.strftime("%Y-%m-%d %H:%M:%S")


def fmt_utc(dt: datetime | None = None) -> str:
    value = dt or now_utc()
    return value.strftime("%Y-%m-%d %H:%M:%S")