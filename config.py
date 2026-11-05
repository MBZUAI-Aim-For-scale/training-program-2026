"""Keep legacy configuration imports on the shared Demo 4 module."""

from pathlib import Path
import sys

_source_root = Path(__file__).resolve().parent / "demos/demo4"
if (_source_root / "ai_weather/config.py").is_file():
    sys.path.insert(0, str(_source_root))

from ai_weather import config as _config

sys.modules[__name__] = _config
