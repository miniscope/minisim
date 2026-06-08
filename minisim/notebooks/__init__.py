"""Bundled teaching notebooks, plus a CLI to copy them somewhere writable.

``pip install minisim`` ships the notebooks read-only inside the installed
package; running them in place is awkward (the install tree is often not
writable, and a notebook writes outputs next to itself). ``minisim-notebooks``
copies the bundles out to a directory you own::

    minisim-notebooks                 # -> ./minisim-notebooks/
    minisim-notebooks ~/work/nb       # -> a directory you choose
    minisim-notebooks --force ./nb    # overwrite an existing copy

No data download is needed: minisim *generates* its recordings from code, so the
notebooks have no external dataset to fetch (and the movies they write are
outputs, never packaged inputs).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# The packaged bundles live under this subpackage, in `training/`.
_BUNDLES = Path(__file__).resolve().parent / "training"

# Generated artifacts a notebook may write next to itself. They are excluded
# from the wheel (see [tool.pdm.build] in pyproject), but a developer's working
# tree can still hold them, so the copier skips them too - a fresh bundle never
# carries someone else's rendered movies or checkpoints.
_IGNORE = shutil.ignore_patterns(
    "*.avi", "*.mp4", "*.mkv", "*.mov", "__pycache__", ".ipynb_checkpoints"
)


def bundles_dir() -> Path:
    """Filesystem path to the packaged notebook bundles (the ``training/`` tree)."""
    return _BUNDLES


def copy_notebooks(dest: str | Path = "minisim-notebooks", *, force: bool = False) -> Path:
    """Copy the bundled notebooks to ``dest`` and return the destination path.

    Generated movies and checkpoints are skipped. Raises ``FileExistsError`` if
    ``dest`` already exists and ``force`` is not set.
    """
    target = Path(dest).expanduser()
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists; pass --force to overwrite, or choose another path."
        )
    shutil.copytree(_BUNDLES, target, ignore=_IGNORE, dirs_exist_ok=force)
    return target


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``minisim-notebooks``."""
    parser = argparse.ArgumentParser(
        prog="minisim-notebooks",
        description="Copy minisim's bundled teaching notebooks to a writable directory.",
    )
    parser.add_argument(
        "dest",
        nargs="?",
        default="minisim-notebooks",
        help="destination directory (default: ./minisim-notebooks).",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite the destination if it already exists.",
    )
    args = parser.parse_args(argv)

    try:
        out = copy_notebooks(args.dest, force=args.force)
    except FileExistsError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    n_notebooks = sum(1 for _ in out.rglob("*.ipynb"))
    print(f"Copied {n_notebooks} notebook(s) to {out}")
    print(f"Next:  cd {out} && pip install 'minisim[notebook]' && jupyter lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
