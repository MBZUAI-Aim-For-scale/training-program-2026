#!/usr/bin/env python
"""
Download pre-computed hindcasts from various sources.

Sources:
- FuXi-S2S: HuggingFace (FudanFuXi/FuXi-S2S)
- WeatherBench2: Google Cloud Storage (graphcast, pangu_weather, ifs)

Usage:
    # List available models
    python scripts/download_hindcasts.py --list

    # Download FuXi-S2S hindcasts for 2019-2021
    python scripts/download_hindcasts.py --model fuxi_s2s --years 2019 2020 2021

    # Download GraphCast hindcasts from WeatherBench2
    python scripts/download_hindcasts.py --model graphcast --years 2020

    # Download for monsoon season only (Apr-Jul)
    python scripts/download_hindcasts.py --model fuxi_s2s --years 2020 --months 4 5 6 7
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    "fuxi_s2s": {
        "source": "HuggingFace (gated)",
        "repo": "FudanFuXi/FuXi-S2S",
        "years": "2002-2021",
        "lead_time": "42 days",
        "type": "deterministic",
        "resolution": "0.25°",
        "note": "Requires HuggingFace access approval",
    },
    # WeatherBench2 models (public, no approval needed)
    "fuxi": {
        "source": "WeatherBench2 (GCS)",
        "bucket": "gs://weatherbench2/datasets",
        "years": "2020",
        "lead_time": "15 days",
        "type": "deterministic",
        "resolution": "0.25°",
    },
    "graphcast": {
        "source": "WeatherBench2 (GCS)",
        "bucket": "gs://weatherbench2/datasets",
        "years": "2020",
        "lead_time": "10 days",
        "type": "deterministic",
        "resolution": "0.25°",
    },
    "pangu": {
        "source": "WeatherBench2 (GCS)",
        "bucket": "gs://weatherbench2/datasets",
        "years": "2018-2022",
        "lead_time": "7 days",
        "type": "deterministic",
        "resolution": "0.25°",
    },
    "gencast": {
        "source": "WeatherBench2 (GCS)",
        "bucket": "gs://weatherbench2/datasets",
        "years": "2020",
        "lead_time": "15 days",
        "type": "probabilistic (50 members)",
        "resolution": "0.25°",
    },
    "neuralgcm_deterministic": {
        "source": "WeatherBench2 (GCS)",
        "bucket": "gs://weatherbench2/datasets",
        "years": "2020",
        "lead_time": "10 days",
        "type": "deterministic",
        "resolution": "0.25°",
    },
    "neuralgcm_ens": {
        "source": "WeatherBench2 (GCS)",
        "bucket": "gs://weatherbench2/datasets",
        "years": "2020",
        "lead_time": "10 days",
        "type": "probabilistic (50 members)",
        "resolution": "0.25°",
    },
    "hres": {
        "source": "WeatherBench2 (GCS)",
        "bucket": "gs://weatherbench2/datasets",
        "years": "2020",
        "lead_time": "10 days",
        "type": "deterministic (ECMWF IFS)",
        "resolution": "0.25°",
    },
    "ifs_ens": {
        "source": "WeatherBench2 (GCS)",
        "bucket": "gs://weatherbench2/datasets",
        "years": "2020",
        "lead_time": "15 days",
        "type": "probabilistic (51 members, ECMWF)",
        "resolution": "~2.8° (regridded)",
    },
}


def list_models():
    """Print available models and their details."""
    print("\nAvailable hindcast models:")
    print("=" * 70)

    for name, info in AVAILABLE_MODELS.items():
        print(f"\n{name}")
        print("-" * len(name))
        for key, value in info.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("\nDependencies:")
    print("  - FuXi-S2S: pip install huggingface-hub py7zr")
    print("    (or: brew install p7zip)")
    print("  - WeatherBench2: pip install gcsfs")


def download_fuxi_s2s(
    years: list,
    months: list,
    data_dir: Path,
):
    """Download FuXi-S2S hindcasts from HuggingFace."""
    try:
        from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        return

    repo_id = "FudanFuXi/FuXi-S2S"
    data_dir = data_dir / "fuxi_s2s"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading FuXi-S2S hindcasts to {data_dir}")
    logger.info(f"Years: {years}, Months: {months}")

    # List available files
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        logger.info(f"Found {len(files)} files in FuXi-S2S repository")

        # Show first few files for reference
        logger.info("Sample files:")
        for f in files[:5]:
            logger.info(f"  {f}")

    except Exception as e:
        logger.error(f"Could not list files in repository: {e}")
        logger.info("Attempting direct download with expected file patterns...")

    # Download files for specified years
    for year in years:
        if year < 2002 or year > 2021:
            logger.warning(f"Year {year} not available (FuXi-S2S covers 2002-2021)")
            continue

        logger.info(f"\nDownloading hindcasts for {year}...")

        # Generate twice-weekly dates for monsoon season
        for month in months:
            start_day = 1
            end_day = 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30
            if month == 2:
                end_day = 29 if year % 4 == 0 else 28

            for day in range(start_day, end_day + 1):
                try:
                    date = datetime(year, month, day)
                    # Check if Monday or Thursday (twice-weekly)
                    if date.weekday() not in [0, 3]:
                        continue

                    date_str = date.strftime("%Y%m%d")

                    # Try different file patterns
                    file_patterns = [
                        f"hindcasts/{year}/{date_str}.nc",
                        f"hindcasts/{date_str}.nc",
                        f"{year}/{date_str}.nc",
                        f"tp/{date_str}.nc",
                    ]

                    for pattern in file_patterns:
                        try:
                            local_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=pattern,
                                repo_type="dataset",
                                local_dir=data_dir,
                            )
                            logger.info(f"Downloaded: {pattern}")
                            break
                        except Exception:
                            continue

                except Exception as e:
                    logger.debug(f"No hindcast for {date}: {e}")


def download_weatherbench2(
    model_name: str,
    years: list,
    months: list,
    data_dir: Path,
    region: dict = None,
):
    """Download WeatherBench2 hindcasts from Google Cloud Storage."""
    try:
        import gcsfs
    except ImportError:
        logger.error("gcsfs not installed. Run: pip install gcsfs zarr")
        return

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from monsoon_benchmark.models.weatherbench2 import WeatherBench2Loader

    if region is None:
        region = {"lat": (5, 40), "lon": (60, 100)}  # India region

    loader = WeatherBench2Loader(
        model_name=model_name,
        data_dir=data_dir / "weatherbench2",
        use_gcs_direct=True,
    )
    loader.load_weights()

    logger.info(f"Downloading {model_name} hindcasts from WeatherBench2")
    logger.info(f"Years: {years}, Months: {months}")
    logger.info(f"Region: {region}")

    for year in years:
        # List available dates
        available_dates = loader.list_available_dates(year=year, months=months)

        if not available_dates:
            logger.warning(f"No hindcasts found for {year}")
            continue

        logger.info(f"Found {len(available_dates)} hindcasts for {year}")

        # Download subset of dates for monsoon (twice weekly)
        # Filter for Mondays and Thursdays
        monsoon_dates = [d for d in available_dates
                        if d.weekday() in [0, 3]  # Monday, Thursday
                        and d.month in months]

        logger.info(f"Downloading {len(monsoon_dates)} twice-weekly hindcasts")

        for init_date in monsoon_dates:
            try:
                out_path = loader.download_hindcast(
                    init_time=init_date,
                    lead_days=15,
                    variables=["total_precipitation_6hr", "total_precipitation_24hr_from_6hr"],
                    region=region,
                )
                logger.info(f"Downloaded: {out_path.name}")
            except Exception as e:
                logger.warning(f"Failed to download {init_date}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-computed hindcasts for monsoon benchmarking"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    parser.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to download hindcasts for",
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020],
        help="Years to download (default: 2020)",
    )

    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7],
        help="Months to download (default: 4 5 6 7 for monsoon)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/hindcasts"),
        help="Directory to save hindcasts (default: data/hindcasts)",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.model:
        parser.print_help()
        print("\nError: --model is required unless using --list")
        return

    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "fuxi_s2s":
        download_fuxi_s2s(args.years, args.months, args.data_dir)
    elif args.model in ["fuxi", "graphcast", "pangu", "gencast",
                        "neuralgcm_deterministic", "neuralgcm_ens", "hres", "ifs_ens"]:
        # WeatherBench2 models
        download_weatherbench2(args.model, args.years, args.months, args.data_dir)
    else:
        print(f"Unknown model: {args.model}")
        print("Run with --list to see available models")


if __name__ == "__main__":
    main()
