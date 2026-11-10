"""Files in Demo 2's data folder, for the dashboard's Data folder section (dashboard.py).

The folder is demos/demo2/data (CDT_DATA overrides it, as in demos/demo2/docker/run.sh); the CDT
container mounts it at /data, so what's added, edited or removed here is what CDT sees, at once.
The dashboard runs as the account that owns it, so files keep that owner.

Everything stays inside the folder: a name is one path component (no "/", no ".."), and a path
that resolves outside the folder (a link pointing elsewhere, say) is refused.
Standard library only, like spark_setup.py.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CDT_DATA") or HERE.parent / "demo2" / "data")
EDIT_MAX = 2 * 1024 * 1024          # bytes: bigger files can be downloaded, not edited here
TEXT_SUFFIXES = {".csv", ".txt", ".tsv", ".dat", ".json", ".xml", ".r", ".R", ".md", ".log", ".cfg", ".ini", ""}


class BadName(ValueError):
    pass


def root() -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT.resolve()


def check_name(name: str) -> str:
    """A file or folder name typed or uploaded: one plain path component."""
    name = (name or "").strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\0" in name:
        raise BadName("Use a plain name, without / or \\")
    if len(name.encode()) > 255:
        raise BadName("That name is too long")
    return name


def resolve(rel: str = "") -> Path:
    """The folder's path for a path relative to it ("" is the folder itself); never outside it."""
    base = root()
    p = (base / rel).resolve()
    if p != base and base not in p.parents:
        raise BadName("That's outside the data folder")
    return p


def relative(p: Path) -> str:
    return "" if p == root() else str(p.relative_to(root()))


def listing(rel: str = "") -> list[dict]:
    """The folder's entries: folders first, then files, each sorted by name (case-insensitive)."""
    d = resolve(rel)
    out = []
    for e in d.iterdir():
        try:
            target = e.resolve()
            if target != root() and root() not in target.parents:
                continue  # (a link that points outside: not shown)
            st = e.stat()
        except OSError:
            continue
        out.append({"name": e.name, "rel": str(Path(rel) / e.name),
                    "dir": e.is_dir(), "size": None if e.is_dir() else st.st_size, "mtime": st.st_mtime})
    return sorted(out, key=lambda x: (not x["dir"], x["name"].lower()))


def size_text(n: int | None) -> str:
    if n is None:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def items_text(rel: str) -> str:
    try:
        n = sum(1 for _ in resolve(rel).iterdir())
    except OSError:
        return ""
    return "empty" if n == 0 else f"{n} item{'s' if n != 1 else ''}"


def when_text(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def save_upload(rel_dir: str, name: str, data: bytes, replace: bool) -> str:
    """Writes an uploaded file into the folder (atomically). Returns "added", "replaced" or "kept"."""
    d = resolve(rel_dir)
    dest = d / check_name(name)
    if dest.exists() and not replace:
        return "kept"
    if dest.is_dir():
        raise BadName(f"{name} is a folder")
    existed = dest.exists()
    tmp = d / f".{dest.name}.upload"
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return "replaced" if existed else "added"


def make_folder(rel_dir: str, name: str) -> str:
    p = resolve(rel_dir) / check_name(name)
    if p.exists():
        raise BadName(f"{name} is already there")
    p.mkdir()
    return relative(p)


def rename(rel: str, new_name: str) -> str:
    p = resolve(rel)
    if p == root():
        raise BadName("The data folder itself can't be renamed")
    dest = p.parent / check_name(new_name)
    if dest.exists():
        raise BadName(f"{new_name} is already there")
    p.rename(dest)
    return relative(dest)


def delete(rel: str) -> None:
    p = resolve(rel)
    if p == root():
        raise BadName("The data folder itself can't be deleted")
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink()


def editable(entry: dict) -> str | None:
    """Why a file can't be edited here (None if it can): only small text files."""
    if entry["dir"]:
        return "It's a folder"
    if entry["size"] > EDIT_MAX:
        return f"It's over {size_text(EDIT_MAX)}: download it to edit it"
    if Path(entry["name"]).suffix.lower() not in {s.lower() for s in TEXT_SUFFIXES}:
        return "Only text files (CSV, TXT, ...) can be edited here"
    return None


def read_text(rel: str) -> tuple[str, str, bool]:
    """The file's text with "\n" line ends, its encoding (UTF-8, else Latin-1), and whether its
    lines ended in "\r\n": write_text puts both back, so an edit changes only what was edited."""
    raw = resolve(rel).read_bytes()
    try:
        text, enc = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, enc = raw.decode("latin-1"), "latin-1"
    return text.replace("\r\n", "\n"), enc, b"\r\n" in raw


def write_text(rel: str, text: str, encoding: str = "utf-8", crlf: bool = False) -> None:
    """Saves edited text (atomically), in the file's own encoding and line ends."""
    p = resolve(rel)
    text = text.replace("\r\n", "\n")
    if crlf:
        text = text.replace("\n", "\r\n")
    tmp = p.parent / f".{p.name}.edit"
    tmp.write_bytes(text.encode(encoding, errors="replace"))
    os.replace(tmp, p)


# --- More than one entry at a time, zips, moving and copying ----------------------------------

ZIP_MAX = 4 * 1024 ** 3            # bytes: a bigger selection isn't zipped here (copy it over SSH instead)
UNZIP_MAX = 8 * 1024 ** 3          # bytes: what an uploaded zip may unpack to
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
TABLE_SUFFIXES = {".csv", ".tsv"}


def walk_files(rel: str):
    """Every file under rel (or rel itself, if it's a file), as paths; links pointing outside skipped."""
    p = resolve(rel)
    if p.is_file():
        yield p
        return
    for dirpath, dirnames, filenames in os.walk(p):
        for n in filenames:
            f = Path(dirpath) / n
            try:
                if root() in f.resolve().parents:
                    yield f
            except OSError:
                pass


def total_size(rels: list[str]) -> int:
    return sum(f.stat().st_size for r in rels for f in walk_files(r))


def zip_bytes(rels: list[str]) -> bytes:
    """A zip of the entries (folders with what's in them), paths relative to their folder."""
    import io
    import zipfile
    if total_size(rels) > ZIP_MAX:
        raise BadName(f"That's over {size_text(ZIP_MAX)}: too big to zip here")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for r in rels:
            base = resolve(r).parent
            for f in walk_files(r):
                z.write(f, str(f.relative_to(base)))
    return buf.getvalue()


def unzip(rel_dir: str, data: bytes, replace: bool) -> tuple[int, int]:
    """Unpacks an uploaded zip into the folder. Every member must land inside it; (written, kept)."""
    import io
    import zipfile
    d = resolve(rel_dir)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members = [m for m in z.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX/")]
        if sum(m.file_size for m in members) > UNZIP_MAX:
            raise BadName(f"It unpacks to over {size_text(UNZIP_MAX)}")
        written = kept = 0
        for m in members:
            parts = [check_name(x) for x in Path(m.filename.replace("\\", "/")).parts]
            dest = d.joinpath(*parts).resolve()
            if d not in dest.parents:
                raise BadName(f"{m.filename} would land outside the folder")
            if dest.exists() and not replace:
                kept += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            written += 1
    return written, kept


def folders() -> list[str]:
    """Every folder in the data folder, "" first, for moving things into."""
    out = [""]
    for dirpath, dirnames, _ in os.walk(root()):
        dirnames[:] = sorted(n for n in dirnames if not n.startswith("."))
        out += [relative(Path(dirpath) / n) for n in dirnames]
    return out


def move(rels: list[str], to_dir: str, replace: bool = False) -> int:
    d = resolve(to_dir)
    moved = 0
    for r in rels:
        p = resolve(r)
        if p == d or p in d.parents:
            raise BadName(f"{p.name} can't go into itself")
        dest = d / p.name
        if dest == p:
            continue
        if dest.exists():
            if not replace:
                raise BadName(f"{p.name} is already in that folder")
            delete(relative(dest))
        shutil.move(str(p), str(dest))
        moved += 1
    return moved


def duplicate(rel: str) -> str:
    """A copy next to it: "name copy.ext", "name copy 2.ext", ..."""
    p = resolve(rel)
    stem, suffix = (p.name, "") if p.is_dir() else (p.stem, p.suffix)
    for i in range(1, 1000):
        dest = p.parent / f"{stem} copy{'' if i == 1 else f' {i}'}{suffix}"
        if not dest.exists():
            break
    (shutil.copytree if p.is_dir() else shutil.copy2)(p, dest)
    return relative(dest)


def new_file(rel_dir: str, name: str) -> str:
    p = resolve(rel_dir) / check_name(name)
    if p.exists():
        raise BadName(f"{name} is already there")
    p.write_bytes(b"")
    return relative(p)


def kind(entry: dict) -> str:
    if entry["dir"]:
        return "Folder"
    s = Path(entry["name"]).suffix.lower()
    return {".csv": "CSV table", ".tsv": "TSV table", ".nc": "NetCDF", ".xlsx": "Excel", ".xls": "Excel",
            ".zip": "Zip", ".txt": "Text", ".json": "JSON", ".rds": "R data", ".shp": "Shapefile"}.get(
        s, "Image" if s in IMAGE_SUFFIXES else (s[1:].upper() + " file" if s else "File"))


def table_text(rows: list[list[str]], cols: list[str], lines: list[str] | None, lens: list[int], ends_nl: bool,
               sep: str, edited: dict[int, dict], deleted: set[int], added: list[dict]) -> str:
    """A CSV's text after the dashboard's table editor: `rows` as shown (padded to the widest), the
    original `lines` (None if a cell spans lines), each row's own length, and the editor's edits
    (cells by column name), deleted row positions and added rows. Rows nobody touched keep their
    exact line; edited and added ones are written with `sep`, without the padding."""
    import csv
    import io

    def line(cells: list[str], keep: int) -> str:
        while len(cells) > keep and cells[-1] == "":
            cells = cells[:-1]
        out = io.StringIO()
        csv.writer(out, delimiter=sep, lineterminator="").writerow(cells)
        return out.getvalue()

    out = []
    for i, cells in enumerate(rows):
        if i in deleted:
            continue
        if i not in edited and lines is not None:
            out.append(lines[i])
            continue
        cells = list(cells)
        for col, val in edited.get(i, {}).items():
            cells[cols.index(str(col))] = "" if val is None else str(val)
        out.append(line(cells, lens[i]))
    out += [line(["" if r.get(c) is None else str(r.get(c)) for c in cols], 0) for r in added]
    return "\n".join(out) + ("\n" if ends_nl or added else "")
