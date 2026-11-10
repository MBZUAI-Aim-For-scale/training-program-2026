"""Boundary downloads are atomic, reusable, and independent across levels/countries."""

import json
from pathlib import Path
import ssl
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

    def test_url_download_supplements_default_trust_with_certifi(self):
        with patch.object(boundaries.ssl, "create_default_context") as create_context, \
                patch.object(boundaries, "urlopen") as network:
            network.return_value.__enter__.return_value.read.return_value = b"boundary"
            self.assertEqual(boundaries._read_url("https://example.org/boundary"), b"boundary")

        create_context.assert_called_once_with()
        context = create_context.return_value
        context.load_verify_locations.assert_called_once_with(cafile=boundaries.certifi.where())
        request = network.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.org/boundary")
        self.assertEqual(request.get_header("User-agent"), "AI-Weather-Benchmark/1.0")
        self.assertEqual(network.call_args.kwargs, {"timeout": 30, "context": context})

    def test_url_download_keeps_certificate_verification_enabled(self):
        with patch.object(boundaries, "urlopen") as network:
            network.side_effect = ssl.SSLCertVerificationError("untrusted certificate")
            with self.assertRaises(ssl.SSLCertVerificationError):
                boundaries._read_url("https://example.org/boundary")

        context = network.call_args.kwargs["context"]
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertGreater(context.cert_store_stats()["x509_ca"], 0)

    def test_provider_geometry_uses_direct_large_file_endpoint(self):
        path = "90a1d52/releaseData/gbOpen/PER/ADM0/boundary.geojson"
        with patch.object(boundaries, "urlopen") as network:
            boundaries._read_url("https://github.com/wmgeolab/geoBoundaries/raw/" + path)
        self.assertEqual(
            network.call_args.args[0].full_url,
            "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/" + path,
        )

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

    def test_every_listed_country_downloads_to_its_own_iso_cache(self):
        for country, iso in boundaries.COUNTRY_ISO3.items():
            with self.subTest(country=country), \
                 patch.object(boundaries, "_read_url", side_effect=self.read) as network:
                result = boundaries.download_country_boundaries(country, self.root)
                self.assertEqual(result["errors"], {})
                self.assertEqual(set(result["paths"]), set(boundaries.BOUNDARY_LEVELS))
                self.assertEqual([call.args[0] for call in network.call_args_list if "/api/" in call.args[0]],
                                 [f"https://www.geoboundaries.org/api/current/gbOpen/{iso}/{level}/"
                                  for level in boundaries.BOUNDARY_LEVELS])
                for level, path in result["paths"].items():
                    self.assertEqual(Path(path), self.root / iso / f"{level}.geojson")
                    self.assertEqual(boundaries.cached_boundary_path(country, level, self.root), Path(path))
                network.reset_mock()
                self.assertEqual(boundaries.download_country_boundaries(country, self.root), result)
                network.assert_not_called()
        self.assertEqual(len(list(self.root.glob("*/ADM0.geojson"))), len(boundaries.COUNTRY_ISO3))

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
