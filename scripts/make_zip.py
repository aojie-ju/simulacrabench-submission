"""Build the upload archive from submission/ with main.py at the archive root.

Entries are sorted and carry a fixed timestamp, so the same code always yields
the same SHA-256.

Usage: python scripts/make_zip.py [submission_dir] [output_zip]
"""
import hashlib
import sys
import zipfile
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".zip"}
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def build(src: Path, dest: Path) -> str:
    if not (src / "main.py").is_file():
        sys.exit(f"ERROR: {src / 'main.py'} not found")
    files = sorted(
        path for path in src.rglob("*")
        if path.is_file()
        and not SKIP_DIRS.intersection(path.relative_to(src).parts)
        and path.suffix not in SKIP_SUFFIXES
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(src).as_posix(), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return hashlib.sha256(dest.read_bytes()).hexdigest()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "submission"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "dist" / "submission.zip"
    digest = build(src.resolve(), dest.resolve())
    print(f"{dest}  sha256:{digest[:16]}")
