from __future__ import annotations


VALID_MODES = {"INTRADAY", "SWING", "POSITIONAL"}


def parse_modes(value: str) -> list[str]:
    if not value or not str(value).strip():
        return []

    modes = [m.strip().upper() for m in str(value).split(",")]
    return [m for m in modes if m in VALID_MODES]


def resolve_symbol_modes(symbol_obj, settings) -> list[str]:
    force_override = str(getattr(symbol_obj, "force_override", "") or "").strip().upper()
    allowed_modes = str(getattr(symbol_obj, "allowed_modes", "") or "").strip().upper()
    enabled_modes = parse_modes(settings.enabled_modes)

    if force_override:
        return [force_override] if force_override in VALID_MODES else enabled_modes

    parsed_allowed = parse_modes(allowed_modes)
    if parsed_allowed:
        constrained = [m for m in parsed_allowed if m in enabled_modes]
        return constrained or enabled_modes

    return enabled_modes