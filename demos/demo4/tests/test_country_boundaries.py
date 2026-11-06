"""Boundary downloads are atomic, reusable, and independent across levels/countries."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_weather import country_boundaries as boundaries


def boundary_json(name="Lima", west=-78):
    return json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"shapeName": name},
        "geometry": {"type": "Polygon", "coordinates": [[
            [west, -13], [west + 2, -13], [west + 2, -11], [west, -11], [west, -13]
        ]]},
    }]}).encode()


class CountryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def read(self, url):
        if "/api/" in url:
            return json.dumps({"gjDownloadURL": "https://example.org/geometry.geojson",
                               "boundaryLicense": "CC BY 4.0"}).encode()
        return boundary_json()

    def test_downloads_normalize_names_preserve_metadata_and_reuse_cache(self):
        from ai_weather.spatial import load_boundary_frame

        with patch.object(boundaries, "_read_url", side_effect=self.read) as network:
            result = boundaries.download_country_boundaries("Peru", self.root)
            self.assertFalse(result["errors"])
            self.assertEqual(set(result["paths"]), {"ADM0", "ADM1", "ADM2"})
            frame = load_boundary_frame(result["paths"]["ADM1"])
            self.assertEqual(frame["Admin1"].tolist(), ["Lima"])
            self.assertEqual(frame.crs.to_epsg(), 4326)
            metadata = self.root / "PER/ADM1.metadata.json"
            self.assertEqual(json.loads(metadata.read_text())["boundaryLicense"], "CC BY 4.0")
            network.reset_mock()
            self.assertEqual(boundaries.download_country_boundaries("Peru", self.root), result)
            network.assert_not_called()
        self.assertIsNone(boundaries.cached_boundary_path("Kenya", "ADM1", self.root))
        self.assertIsNone(boundaries.cached_boundary_path("../Peru", "ADM1", self.root))
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_failed_level_does_not_discard_successful_downloads(self):
        def read(url):
            if "/ADM2/" in url:
                raise TimeoutError("source timed out")
            return self.read(url)

        with patch.object(boundaries, "_read_url", side_effect=read):
            result = boundaries.download_country_boundaries("Peru", self.root)
        self.assertEqual(set(result["paths"]), {"ADM0", "ADM1"})
        self.assertIn("timed out", result["errors"]["ADM2"])
        self.assertIsNone(boundaries.cached_boundary_path("Peru", "ADM2", self.root))

    def test_bad_geometry_never_becomes_a_cached_boundary(self):
        def read(url):
            if "/api/" in url:
                return self.read(url)
            return b'{"type":"FeatureCollection","features":[]}'

        with patch.object(boundaries, "_read_url", side_effect=read):
            result = boundaries.download_country_boundaries("Peru", self.root)
        self.assertEqual(result["paths"], {})
        self.assertEqual(len(result["errors"]), 3)
        self.assertFalse(list(self.root.rglob("*.geojson")))

    def test_missing_administrative_names_fail_only_subregions(self):
        def read(url):
            if "/api/" in url:
                return self.read(url)
            data = json.loads(boundary_json())
            data["features"][0]["properties"] = {}
            return json.dumps(data).encode()

        with patch.object(boundaries, "_read_url", side_effect=read):
            result = boundaries.download_country_boundaries("Peru", self.root)
        self.assertEqual(set(result["paths"]), {"ADM0"})
        self.assertEqual(set(result["errors"]), {"ADM1", "ADM2"})
