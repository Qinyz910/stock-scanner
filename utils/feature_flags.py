import os
from typing import Dict

# Feature flags default to False for safe rollout
_DEFAULT_FLAGS = {
    "ENABLE_FACTORS": False,
    "ENABLE_BACKTEST": False,
    "ENABLE_PORTFOLIO": False,
    "ENABLE_RISK": False,
    "ENABLE_ML": False,
    "ENABLE_RECO": False,
}


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def load_flags() -> Dict[str, bool]:
    flags = {}
    for k, default in _DEFAULT_FLAGS.items():
        env_v = os.getenv(k)
        flags[k] = _parse_bool(env_v) if env_v is not None else default
    return flags


FLAGS = load_flags()


def is_enabled(name: str) -> bool:
    return FLAGS.get(name, False)
