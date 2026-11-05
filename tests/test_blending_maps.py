"""Real map rendering accepts the string paths supplied by shared config."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from PIL import Image
from shapely.geometry import box

from apps.onset_blending.predict import run_maps


class BlendingMapTests(unittest.TestCase):
    def test_configured_string_paths_produce_both_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            country = root / "country.geojson"
            districts = root / "districts.geojson"
            gpd.GeoDataFrame(geometry=[box(38, 8, 40, 10)], crs="EPSG:4326").to_file(country)
            gpd.GeoDataFrame(
                {"adm3_name": ["A", "B"]},
                geometry=[box(38, 8, 39, 10), box(39, 8, 40, 10)],
                crs="EPSG:4326",
            ).to_file(districts)
            frame = pd.DataFrame(
                {
                    "id": ["A", "B"],
                    "time": ["2024-06-13"] * 2,
                    "week1": [0.5, 0.1],
                    "week2": [0.2, 0.1],
                    "week3": [0.1, 0.1],
                    "week4": [0.1, 0.1],
                    "later": [0.1, 0.6],
                }
            )
            for path_type in (str, Path):
                with (
                    self.subTest(path_type=path_type.__name__),
                    patch.object(run_maps, "ETH_GEOJSON_PATH", path_type(country)),
                    patch.object(run_maps, "ZONE_WOREDA_SHP", path_type(districts)),
                ):
                    output = root / path_type.__name__
                    run_maps.make_maps(frame, output, date_str="20240613")
                    for filename in ("prob_weeks1-4_20240613.png", "map_max_period_20240613.png"):
                        path = output / "maps" / filename
                        self.assertTrue(path.is_file())
                        with Image.open(path) as image:
                            image.verify()


if __name__ == "__main__":
    unittest.main()
