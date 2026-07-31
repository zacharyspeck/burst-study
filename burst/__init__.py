"""burst -- configuration scaffold for the burst injection study.

Intentionally almost empty. This package currently contains the config system
and nothing else; training, model, data and analysis code arrive later and are
expected to read every setting from `burst.config`.

Import from the module directly:

    from burst.config import load_config, Config, ConfigError

Deliberately NOT re-exported here. Importing `burst.config` from this file
would make `python -m burst.config` load the module twice -- once via this
package's __init__, once as __main__ -- which runpy warns about and which is
exactly the kind of subtle double-import that is not worth the convenience of a
shorter import path.
"""

__version__ = "0.1.0"
