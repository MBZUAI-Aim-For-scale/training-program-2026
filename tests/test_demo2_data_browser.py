"""Demo 2's Data folder (demos/demo1/data_browser.py): it never leaves the folder, and keeps files as they are."""
import importlib
import io
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos" / "demo1"))


@pytest.fixture
def fb(tmp_path, monkeypatch):
    monkeypatch.setenv("CDT_DATA", str(tmp_path / "data"))
    import data_browser
    return importlib.reload(data_browser)


def test_names_and_paths_stay_inside(fb, tmp_path):
    for bad in ("", ".", "..", "a/b", "..\\x", "x\0"):
        with pytest.raises(fb.BadName):
            fb.check_name(bad)
    with pytest.raises(fb.BadName):
        fb.resolve("../outside")
    (tmp_path / "secret").write_text("no")
    os.symlink(tmp_path / "secret", fb.root() / "link")       # a link pointing out
    assert [e["name"] for e in fb.listing()] == []
    with pytest.raises(fb.BadName):
        fb.resolve("link")


def test_upload_keeps_or_replaces(fb):
    assert fb.save_upload("", "a.csv", b"1", replace=False) == "added"
    assert fb.save_upload("", "a.csv", b"2", replace=False) == "kept"
    assert (fb.root() / "a.csv").read_bytes() == b"1"
    assert fb.save_upload("", "a.csv", b"2", replace=True) == "replaced"
    assert (fb.root() / "a.csv").read_bytes() == b"2"
    assert not [p for p in fb.root().iterdir() if p.name.startswith(".")]   # (no temporary file left)


def test_unzip_refuses_paths_out_of_the_folder(fb):
    good, bad = io.BytesIO(), io.BytesIO()
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("stations/rr.csv", "x")
        z.writestr("__MACOSX/._rr.csv", "junk")
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../../evil.txt", "x")
    assert fb.unzip("", good.getvalue(), replace=False) == (1, 0)
    assert (fb.root() / "stations" / "rr.csv").read_text() == "x"
    assert not (fb.root() / "__MACOSX").exists()
    with pytest.raises(fb.BadName):
        fb.unzip("", bad.getvalue(), replace=False)
    assert not (fb.root().parent / "evil.txt").exists()


def test_text_keeps_its_encoding_and_line_ends(fb):
    (fb.root() / "w.csv").write_bytes("Station;Débit\r\nA;1\r\n".encode("latin-1"))
    text, enc, crlf = fb.read_text("w.csv")
    assert (text, enc, crlf) == ("Station;Débit\nA;1\n", "latin-1", True)
    fb.write_text("w.csv", text + "B;2\n", enc, crlf)
    assert (fb.root() / "w.csv").read_bytes() == "Station;Débit\r\nA;1\r\nB;2\r\n".encode("latin-1")


def test_folders_move_duplicate_delete_zip(fb):
    fb.make_folder("", "in")
    fb.new_file("", "n.txt")
    assert fb.folders() == ["", "in"]
    with pytest.raises(fb.BadName):
        fb.move(["in"], "in")                                     # (not into itself)
    assert fb.duplicate("n.txt") == "n copy.txt" and fb.duplicate("n.txt") == "n copy 2.txt"
    assert fb.move(["n.txt", "n copy.txt"], "in") == 2
    assert sorted(e["name"] for e in fb.listing("in")) == ["n copy.txt", "n.txt"]
    names = zipfile.ZipFile(io.BytesIO(fb.zip_bytes(["in"]))).namelist()
    assert sorted(names) == ["in/n copy.txt", "in/n.txt"]
    with pytest.raises(fb.BadName):
        fb.delete("")
    fb.delete("in")
    assert [e["name"] for e in fb.listing()] == ["n copy 2.txt"]
    assert fb.rename("n copy 2.txt", "notes.txt") == "notes.txt"
    assert fb.editable(fb.listing()[0]) is None


def test_table_edits_touch_only_the_edited_rows(fb):
    text = 'Stations,"A",B\nLON,1.5\nLAT,2,3\n'          # (quotes it doesn't need; a short row)
    lines = text[:-1].split("\n")
    rows = [["Stations", "A", "B"], ["LON", "1.5", ""], ["LAT", "2", "3"]]
    lens = [3, 2, 3]
    same = fb.table_text(rows, ["1", "2", "3"], lines, lens, True, ",", {}, set(), [])
    assert same == text                                     # nothing edited: byte for byte the same
    out = fb.table_text(rows, ["1", "2", "3"], lines, lens, True, ",", {1: {"2": "9.5"}}, {2}, [{"1": "ELV", "2": "7"}])
    assert out == 'Stations,"A",B\nLON,9.5\nELV,7\n'
    semi = fb.table_text([["a;b", "c"]], ["1", "2"], None, [2], False, ";", {0: {"2": "d"}}, set(), [])
    assert semi == '"a;b";d'
