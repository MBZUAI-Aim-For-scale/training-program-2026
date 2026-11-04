"""Country boundary lookups must close readers on every exit path."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class ShapefileResourceTests(unittest.TestCase):
    def test_reader_closes_on_match_missing_country_and_error(self):
        path = Path("apps/benchmarking/ROMP/momp/utils/land_mask.py")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_shp")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
        for outcome in ("match", "missing", "error"):
            with self.subTest(outcome=outcome):
                reader = MagicMock()
                geometry = object()
                reader.records.return_value = [SimpleNamespace(attributes={"NAME": "Ethiopia"}, geometry=geometry)]
                if outcome == "error":
                    reader.records.side_effect = RuntimeError("read failed")
                shapereader = MagicMock()
                shapereader.Reader.return_value = reader
                cartopy = MagicMock()
                cartopy.io.shapereader = shapereader
                with patch.dict("sys.modules", {"cartopy": cartopy, "cartopy.io": cartopy.io,
                                                "cartopy.io.shapereader": shapereader}):
                    if outcome == "error":
                        with self.assertRaises(RuntimeError):
                            namespace["get_shp"]()
                    else:
                        result = namespace["get_shp"](region="Ethiopia" if outcome == "match" else "Missing")
                        self.assertIs(result, geometry if outcome == "match" else None)
                reader.close.assert_called_once()
