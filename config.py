"""Keep legacy ``import config`` callers on the shared weather configuration."""

import sys

from ai_weather import config as _config

# Share one module instance so runtime updates are visible through either import.
sys.modules[__name__] = _config
