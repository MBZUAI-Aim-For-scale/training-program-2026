"""Regression checks for onset blending's operational data boundaries."""

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from apps.onset_blending.predict.export_blend_output import export_blend_predictions
from apps.onset_blending.python.pipelines._shared.read_spec import load_spec


class BlendingHelperTests(unittest.TestCase):
    def test_specs_resolve_from_package(self):
        spec = load_spec("cv_models_clim_mok_date_2026", "2025_blend")
        self.assertIn("models", spec)

    def test_export_only_selected_issue_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(
                {
                    "id": ["A", "A"],
                    "time": pd.to_datetime(["2026-06-15", "2026-06-16"]),
                    "year": [2026, 2026],
                }
            )
            for name in ("week1", "week2", "week3", "week4", "later"):
                frame[f"cv_{name}"] = 0.2
            frame.to_pickle(root / "preds.pkl")
            export_blend_predictions("2026-06-15", {}, root / "preds.pkl", root)
            result = pd.read_csv(root / "blend_output_summary_20260615.csv")
            self.assertEqual(len(result), 1)
            self.assertTrue(result.time.str.startswith("2026-06-15").all())

    def test_export_rejects_missing_issue_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({"id": ["A"], "time": ["2026-06-16"], "year": [2026]})
            for name in ("week1", "week2", "week3", "week4", "later"):
                frame[f"cv_{name}"] = 0.2
            frame.to_pickle(root / "preds.pkl")
            with self.assertRaisesRegex(ValueError, "issue date"):
                export_blend_predictions("2026-06-15", {}, root / "preds.pkl", root)

    def test_relative_dissemination_path_uses_repository(self):
        module = importlib.import_module("apps.onset_blending.python.prepare_data.nc_utils")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": ["A"]}).to_csv(root / "cells.csv", index=False)
            with patch.object(module, "PROJECT_ROOT", root):
                result = module.filter_by_dissemination_cells(
                    pd.DataFrame({"id": ["A", "B"]}),
                    {"filter": {"dissemination_cells_file": "cells.csv"}},
                )
            self.assertEqual(result.id.tolist(), ["A"])

    def test_export_rejects_invalid_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({"id": ["A"], "time": ["2026-06-15"], "year": [2026]})
            for name in ("week1", "week2", "week3", "week4", "later"):
                frame[f"cv_{name}"] = float("nan")
            frame.to_pickle(root / "preds.pkl")
            with self.assertRaisesRegex(ValueError, "invalid probabilities"):
                export_blend_predictions("2026-06-15", {}, root / "preds.pkl", root)

    def test_export_rejects_missing_probability_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": ["A"], "time": ["2026-06-15"], "year": [2026]}).to_pickle(
                root / "preds.pkl"
            )
            with self.assertRaisesRegex(ValueError, "Missing prediction columns"):
                export_blend_predictions("2026-06-15", {}, root / "preds.pkl", root)

    def test_map_cli_filters_exact_issue_date_without_static_onset_overlay(self):
        from apps.onset_blending.predict import run_maps

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "blend_output_summary_20260615.csv"
            pd.DataFrame(
                {"time": ["2026-06-15", "2026-06-16"], "id": ["A", "A"], "year": [2026, 2026]}
            ).to_csv(source, index=False)
            with (
                patch(
                    "sys.argv",
                    ["run_maps", "--input_file", str(source), "--output_path", str(root)],
                ),
                patch.object(run_maps, "make_maps") as draw,
            ):
                run_maps.main()
            self.assertEqual(len(draw.call_args.args[0]), 1)
            self.assertIsNone(draw.call_args.kwargs["onset_file"])

    def test_map_cli_rejects_unavailable_issue_date(self):
        from apps.onset_blending.predict import run_maps

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "blend_output_summary_20260615.csv"
            pd.DataFrame({"time": ["2026-06-16"], "year": [2026]}).to_csv(source, index=False)
            with (
                patch(
                    "sys.argv",
                    ["run_maps", "--input_file", str(source), "--output_path", str(root)],
                ),
                self.assertRaisesRegex(ValueError, "issue date"),
            ):
                run_maps.main()

    def test_maps_reject_unmatched_districts(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from apps.onset_blending.predict import run_maps

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = root / "district.geojson"
            shape.touch()
            districts = gpd.GeoDataFrame(
                {"adm3_name": ["B"]}, geometry=[Point(39, 8)], crs="EPSG:4326"
            )
            frame = pd.DataFrame({"id": ["A"], "time": ["2026-06-15"]})
            for week in range(1, 5):
                frame[f"week{week}"] = 0.2
            with (
                patch.object(run_maps, "ETH_GEOJSON_PATH", root / "missing"),
                patch.object(run_maps, "ZONE_WOREDA_SHP", shape),
                patch.object(run_maps.gpd, "read_file", return_value=districts),
                self.assertRaisesRegex(ValueError, "district geometry"),
            ):
                run_maps.make_maps(frame, root, date_str="20260615")
