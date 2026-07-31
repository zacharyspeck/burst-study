"""burst -- configuration scaffold for the burst injection study.

Intentionally almost empty. This package currently contains the config system
and nothing else; training, model, data and analysis code arrive later and are
expected to read every setting from `burst.config`.
"""

from burst.config import ARMS, Config, ConfigError, load_config, run_name_for

__all__ = ["ARMS", "Config", "ConfigError", "load_config", "run_name_for"]
__version__ = "0.1.0"
