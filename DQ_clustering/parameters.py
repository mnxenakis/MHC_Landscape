"""Central registry for dq_clustering default parameters."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

PACKAGE_ROOT = Path(__file__).resolve().parent
PARAMETER_LOG = PACKAGE_ROOT / "parameters.log"


def _parse_value(raw: str) -> Any:
    text = raw.split("#", 1)[0].strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered == "nan":
        return math.nan
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text, 10)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


@lru_cache(maxsize=1)
def get_parameter_defaults(log_path: Path | None = None) -> dict[str, Any]:
    """
    Load parameter defaults exclusively from parameters.log.

    This function intentionally does not ship any in-code fallback; if the log
    is missing or incomplete the caller must supply values manually.
    """

    target = (log_path or PARAMETER_LOG).resolve()
    if not target.is_file():
        raise FileNotFoundError(
            f"Parameter log not found at {target}. Please create it before running the CLI."
        )

    defaults: Dict[str, Any] = {}
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            defaults[key.strip()] = _parse_value(value)

    if not defaults:
        raise ValueError(
            f"Parameter log {target} did not define any key=value pairs."
        )
    return defaults


__all__ = ["get_parameter_defaults", "PARAMETER_LOG"]
