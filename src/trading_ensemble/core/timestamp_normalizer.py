from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .timeutils import IST, now_ist

logger = logging.getLogger(__name__)

KNOWN_STRING_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


@dataclass
class NormalizedTimestamp:
    value_ist: datetime
    parsed_timezone: str
    validation_outcome: str
    freshness_outcome: str
    age_minutes: float | None


def _to_aware_ist(dt: datetime, assume_exchange_local_ist: bool) -> tuple[datetime, str]:
    if dt.tzinfo is None:
        if assume_exchange_local_ist:
            aware = dt.replace(tzinfo=IST)
            return aware, "ASSUMED_IST"
        aware = dt.replace(tzinfo=timezone.utc).astimezone(IST)
        return aware, "ASSUMED_UTC"
    return dt.astimezone(IST), str(dt.tzinfo)


def _parse_timestamp_value(raw_value: Any) -> datetime | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, datetime):
        return raw_value

    if isinstance(raw_value, (int, float)):
        epoch = float(raw_value)
        if epoch > 1_000_000_000_000:
            epoch = epoch / 1000.0
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    raw_str = str(raw_value).strip()
    if not raw_str:
        return None

    if raw_str.isdigit():
        epoch = float(raw_str)
        if epoch > 1_000_000_000_000:
            epoch = epoch / 1000.0
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    iso_candidate = raw_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate)
    except ValueError:
        pass

    for fmt in KNOWN_STRING_FORMATS:
        try:
            return datetime.strptime(raw_str, fmt)
        except ValueError:
            continue

    return None


def normalize_external_timestamp(
    *,
    source: str,
    raw_value: Any,
    assume_exchange_local_ist: bool = True,
    max_future_seconds: int = 120,
    max_age_minutes: float | None = None,
) -> NormalizedTimestamp | None:
    parsed = _parse_timestamp_value(raw_value)
    if parsed is None:
        logger.warning(json.dumps({
            "event": "timestamp_normalization",
            "source": source,
            "raw_timestamp": str(raw_value),
            "parsed_timezone": "UNPARSEABLE",
            "normalized_ist_timestamp": None,
            "validation_outcome": "MALFORMED",
            "freshness_outcome": "UNKNOWN",
        }))
        return None

    normalized, parsed_tz = _to_aware_ist(parsed, assume_exchange_local_ist=assume_exchange_local_ist)
    current = now_ist()
    future_seconds = (normalized - current).total_seconds()

    if future_seconds > max_future_seconds:
        logger.warning(json.dumps({
            "event": "timestamp_normalization",
            "source": source,
            "raw_timestamp": str(raw_value),
            "parsed_timezone": parsed_tz,
            "normalized_ist_timestamp": normalized.isoformat(timespec="seconds"),
            "validation_outcome": "FUTURE_REJECTED",
            "freshness_outcome": "UNKNOWN",
        }))
        return None

    age_minutes = max(0.0, (current - normalized).total_seconds() / 60.0)
    freshness_outcome = "NOT_CHECKED"
    if max_age_minutes is not None:
        freshness_outcome = "STALE" if age_minutes > max_age_minutes else "FRESH"

    normalized_obj = NormalizedTimestamp(
        value_ist=normalized,
        parsed_timezone=parsed_tz,
        validation_outcome="OK",
        freshness_outcome=freshness_outcome,
        age_minutes=age_minutes,
    )

    logger.info(json.dumps({
        "event": "timestamp_normalization",
        "source": source,
        "raw_timestamp": str(raw_value),
        "parsed_timezone": parsed_tz,
        "normalized_ist_timestamp": normalized.isoformat(timespec="seconds"),
        "validation_outcome": "OK",
        "freshness_outcome": freshness_outcome,
        "age_minutes": round(age_minutes, 3),
    }))
    return normalized_obj
