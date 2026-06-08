"""The bundled-notebooks packaging and the ``minisim-notebooks`` copier."""

from __future__ import annotations

import pytest

from minisim.notebooks import bundles_dir, copy_notebooks, main


def test_bundles_dir_ships_the_anatomy_notebook():
    """The package carries the training bundle (notebook + README)."""
    root = bundles_dir()
    assert root.is_dir()
    notebooks = list(root.rglob("*.ipynb"))
    assert notebooks, "no .ipynb shipped under notebooks/training"
    assert (root / "01_anatomy" / "README.md").is_file()


def test_copy_notebooks_copies_notebook_and_skips_videos(tmp_path):
    """A fresh copy carries the notebook but never generated movies."""
    dest = tmp_path / "out"
    result = copy_notebooks(dest)

    assert result == dest
    assert list(dest.rglob("*.ipynb")), "notebook not copied"
    assert not list(dest.rglob("*.avi")), "generated video leaked into the copy"


def test_copy_notebooks_refuses_existing_dir_without_force(tmp_path):
    dest = tmp_path / "out"
    copy_notebooks(dest)
    with pytest.raises(FileExistsError):
        copy_notebooks(dest)
    # force overwrites in place rather than raising.
    assert copy_notebooks(dest, force=True) == dest


def test_main_returns_zero_and_reports(tmp_path, capsys):
    rc = main([str(tmp_path / "out")])
    assert rc == 0
    assert "Copied" in capsys.readouterr().out


def test_main_returns_one_on_existing_dir(tmp_path):
    dest = tmp_path / "out"
    assert main([str(dest)]) == 0
    assert main([str(dest)]) == 1  # second run without --force fails cleanly
