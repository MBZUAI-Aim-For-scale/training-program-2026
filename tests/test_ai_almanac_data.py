"""demos/demo1/ai_almanac_data.py: country data into AI Almanac datasets (links, marker, names, what it skips)."""
import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos" / "demo1"))


@pytest.fixture
def aad(tmp_path, monkeypatch):
    monkeypatch.setenv("ALMANAC_DATASETS_DIR", str(tmp_path / "ai-almanac-data"))
    import ai_almanac_data
    m = importlib.reload(ai_almanac_data)
    monkeypatch.setattr(m.country_data, "data_dir", lambda c: tmp_path / f"{c.lower()}_data")
    monkeypatch.setattr(m, "ALMANAC", tmp_path / "ai-almanac")
    calls = []
    monkeypatch.setattr(m, "almanac", lambda args, stdin, dry: calls.append((args, stdin is not None)) or 0)
    m.calls = calls
    (tmp_path / "ai-almanac").mkdir()
    (tmp_path / "ai-almanac" / "compose.spark.yaml").write_text("")
    return m


def country_files(root: Path, country: str, years=(2000, 2001)):
    d = root / f"{country.lower()}_data"
    for y in years:
        for rel in (f"CHIRPSv3_Country_0p25deg/{country}/CHIRPSv3_{country}_{y}_box025.nc",
                    f"CHIRTS_Country_0p25deg/Tmax/{country}/CHIRTS-ERA5_Tmax_{country}_{y}_box025.nc",
                    *[f"Forecasts_Country_0p25deg/{m}/{v}/{country}/{m}_{v}_{country}_{y}_box025.nc"
                      for m in ("AIFS_v2", "AIFS_ENS_v2", "graphcast") for v in ("tp", "t2m_max")]):
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(rel.encode())
    return d


def test_slug_matches_ai_almanacs_region_ids(aad):
    assert (aad.slug("Sierra_Leone"), aad.display("Sierra_Leone"), aad.slug("Nigeria")) == (
        "sierra-leone", "Sierra Leone", "nigeria")


def test_links_rainfall_only_marks_and_registers(aad, tmp_path):
    data = country_files(tmp_path, "Sierra_Leone")
    assert aad.run("Sierra_Leone") == 0
    root = tmp_path / "ai-almanac-data" / "sierra-leone"
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*.nc")) == sorted(
        [f"obs/CHIRPSv3_Sierra_Leone_{y}_box025.nc" for y in (2000, 2001)]
        + [f"models/{m}/{m}_tp_Sierra_Leone_{y}_box025.nc" for m in ("AIFS_v2", "AIFS_ENS_v2", "graphcast")
           for y in (2000, 2001)])                                       # (tp only; no CHIRTS, no t2m)
    src = data / "CHIRPSv3_Country_0p25deg/Sierra_Leone/CHIRPSv3_Sierra_Leone_2000_box025.nc"
    assert os.path.samefile(src, root / "obs" / src.name)               # a hard link: no extra space
    assert (root / ".fleet-demos-keep").exists()                        # (the fleet's sync leaves it alone)
    names = (root / "ai-almanac-names.yaml").read_text()
    assert "  sierra-leone/obs:\n    name: 'CHIRPS v3'\n    obs_date_marks: start\n" in names
    assert "  sierra-leone/models/AIFS_v2:\n    name: AIFSv2\n    forecast_model_id: aifs2\n" in names
    assert aad.calls == [
        (["python", "-", "Sierra Leone", "/datasets/sierra-leone/obs"], True),
        (["ai-almanac", "sources", "discover", "/datasets", "--register", "--names",
          "/datasets/sierra-leone/ai-almanac-names.yaml"], False)]
    assert aad.link_files(src.parent, root / "obs", dry=False) == (0, 2)  # (again: nothing to do)


def test_bucket_countries_hand_made_copies_and_other_folders(aad, tmp_path):
    country_files(tmp_path, "Ethiopia")
    assert aad.run("Ethiopia") == 0 and aad.calls == []                 # (its datasets are the bucket's)
    data = country_files(tmp_path, "Nigeria")
    copy = tmp_path / "ai-almanac-data" / "nigeria" / "obs"            # (made by hand, like Nigeria's)
    copy.mkdir(parents=True)
    f = data / "CHIRPSv3_Country_0p25deg/Nigeria/CHIRPSv3_Nigeria_2000_box025.nc"
    (copy / f.name).write_bytes(f.read_bytes())
    assert aad.run("Nigeria") == 0 and len(aad.calls) == 2              # (adopted: kept, marked, registered)
    assert (copy / f.name).exists() and (copy.parent / ".fleet-demos-keep").exists()
    other = tmp_path / "ai-almanac-data" / "kenya" / "obs"             # (not made from the country data)
    other.mkdir(parents=True)
    (other / "IMD_2000.nc").write_bytes(b"x")
    country_files(tmp_path, "Kenya")
    aad.calls.clear()
    assert aad.run("Kenya") == 0 and aad.calls == [] and not (other.parent / ".fleet-demos-keep").exists()


def test_dry_run_changes_nothing_and_no_data(aad, tmp_path):
    country_files(tmp_path, "Rwanda")
    assert aad.run("Rwanda", dry=True) == 0
    assert not (tmp_path / "ai-almanac-data").exists()
    assert aad.run("Peru") == 1                                        # (no country data yet)
