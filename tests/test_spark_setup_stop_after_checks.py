"""A full `spark-setup` run stops the demos it started; Launch (--only) and --build-only don't."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos" / "demo1"))

import spark_setup  # noqa: E402

CFG = {"demo": [
    {"id": "cdt", "title": "Demo 2", "image": "cdt", "container": "cdt", "port": 8504},
    {"id": "demo3", "title": "Demo 3", "image": "demo3", "container": "demo3", "port": 8502},
    {"id": "almanac", "title": "AI Almanac", "image": "ai-almanac:spark", "container": "ai-almanac-spark-almanac-1",
     "port": 8765, "checkout": "~/ai-almanac"},
]}


def ids(steps):
    return [s["id"] for s in steps]


def test_only_a_full_run_plans_the_stop():
    assert ids(spark_setup.plan(CFG, None, False, False, True)[1])[-1] == "demos.stop"
    for only, host_only, build_only, stop_after in ((["demo3"], False, False, False),   # Launch
                                                    (None, False, True, False),         # --build-only
                                                    (None, True, False, False),         # --host-only
                                                    (None, False, False, False)):       # --keep-running
        assert "demos.stop" not in ids(spark_setup.plan(CFG, only, host_only, build_only, stop_after)[1])


def test_stops_what_it_started_and_leaves_what_was_running(monkeypatch, tmp_path):
    run = SimpleNamespace(data={"steps": [
        {"id": "cdt.start", "detail": "container started"},
        {"id": "demo3.start", "detail": "already running (since 2026-11-12 08:00)"},  # (someone's)
        {"id": "almanac.start", "detail": "container started"},
    ]})
    calls = []
    monkeypatch.setattr(spark_setup, "sh", lambda cmd, log, timeout=0: calls.append(cmd) or (0, ""))
    monkeypatch.setattr(spark_setup, "inspect", lambda kind, name: {"State": {"Running": name == "demo3"}})
    with open(tmp_path / "log", "w") as log:
        detail = spark_setup.step_stop_started(run, "demos.stop", log, CFG["demo"])
    assert calls == [["docker", "stop", "-t", "30", "cdt"], ["docker", "stop", "-t", "30", "ai-almanac-spark-almanac-1"]]
    assert detail == "stopped cdt, ai-almanac-spark-almanac-1; left running (it was before): demo3"
