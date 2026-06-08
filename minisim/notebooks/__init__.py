"""Bundled teaching notebooks, plus a CLI to copy them somewhere writable.

``pip install minisim`` ships the notebooks read-only inside the installed
package; running them in place is awkward (the install tree is often not
writable, and a notebook writes outputs next to itself). The ``minisim-notebooks``
command lists the bundles and copies the ones you want out to a directory you
own::

    minisim-notebooks list                  # show available notebooks
    minisim-notebooks copy 01_anatomy        # -> ./minisim-notebooks/01_anatomy
    minisim-notebooks copy --all -o ~/work   # copy every bundle under ~/work
    minisim-notebooks copy 01_anatomy -f     # overwrite an existing copy

The ``list``/``copy`` verbs mirror minian's ``minian notebooks`` CLI, so the two
sister tools feel the same.

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

# Default parent directory `copy` writes bundles into when no -o is given.
DEFAULT_DEST = "minisim-notebooks"


def bundles_dir() -> Path:
    """Filesystem path to the packaged notebook bundles (the ``training/`` tree)."""
    return _BUNDLES


def _description(bundle: Path) -> str:
    """One-line description of a bundle, taken from its README's title line."""
    readme = bundle / "README.md"
    if readme.is_file():
        for line in readme.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                return stripped.lstrip("#").strip()
    return "(no description)"


def notebooks() -> dict[str, str]:
    """Map each bundled notebook's name to its one-line description.

    A bundle is any subdirectory of ``training/`` that contains a ``.ipynb``;
    its name is the directory name (e.g. ``01_anatomy``).
    """
    return {
        child.name: _description(child)
        for child in sorted(_BUNDLES.iterdir())
        if child.is_dir() and any(child.glob("*.ipynb"))
    }


def copy(name: str, dest: str | Path = DEFAULT_DEST, *, force: bool = False) -> Path:
    """Copy one bundled notebook into ``dest/<name>`` and return that path.

    Generated movies and checkpoints are skipped. Raises ``KeyError`` if ``name``
    is not a bundled notebook, or ``FileExistsError`` if the target already exists
    and ``force`` is not set.
    """
    src = _BUNDLES / name
    if not src.is_dir() or not any(src.glob("*.ipynb")):
        raise KeyError(name)
    target = Path(dest).expanduser() / name
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists; pass --force to overwrite, or choose another path."
        )
    shutil.copytree(src, target, ignore=_IGNORE, dirs_exist_ok=force)
    return target


def _print_table(rows: list[tuple[str, str]]) -> None:
    """Print ``(name, description)`` rows as two left-aligned columns."""
    if not rows:
        return
    width = max(len(name) for name, _ in rows)
    for name, desc in rows:
        print(f"{name:<{width}}  {desc}")


def _cmd_list(args: argparse.Namespace) -> int:
    available = notebooks()
    if not available:
        print("No notebooks are bundled with this install.", file=sys.stderr)
        return 1
    _print_table(list(available.items()))
    return 0


def _cmd_copy(args: argparse.Namespace) -> int:
    names = list(notebooks()) if args.all else ([args.name] if args.name else [])
    if not names:
        print(
            "Give a notebook name or --all (see `minisim-notebooks list`).",
            file=sys.stderr,
        )
        return 1

    dest = args.output or DEFAULT_DEST
    for name in names:
        try:
            out = copy(name, dest, force=args.force)
        except KeyError:
            print(
                f"error: unknown notebook {name!r} (see `minisim-notebooks list`).",
                file=sys.stderr,
            )
            return 1
        except FileExistsError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        print(f"copied {name} -> {out}")

    print(f"Next:  cd {dest} && pip install 'minisim[notebook]' && jupyter lab")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``minisim-notebooks`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="minisim-notebooks",
        description="List and copy minisim's bundled teaching notebooks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available notebooks").set_defaults(func=_cmd_list)

    cp = sub.add_parser("copy", help="copy a notebook into a directory")
    cp.add_argument("name", nargs="?", help="notebook name (see `list`)")
    cp.add_argument("--all", action="store_true", help="copy every notebook")
    cp.add_argument(
        "-o", "--output", help=f"destination directory (default: ./{DEFAULT_DEST})"
    )
    cp.add_argument(
        "-f", "--force", action="store_true", help="overwrite existing copies"
    )
    cp.set_defaults(func=_cmd_copy)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``minisim-notebooks``."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
