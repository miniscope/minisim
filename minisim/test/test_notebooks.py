"""The bundled-notebooks packaging and the ``minisim-notebooks`` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

import minisim.notebooks as nb
from minisim.notebooks import (
    _description,
    _print_table,
    bundles_dir,
    copy,
    main,
    notebooks,
)


def _make_bundle(category: Path, name: str, *, title: str) -> Path:
    """Create a fake bundle (dir + one .ipynb + a titled README) under ``category``."""
    bundle = category / name
    bundle.mkdir(parents=True)
    (bundle / f"{name}.ipynb").write_text("{}", encoding="utf-8")
    (bundle / "README.md").write_text(f"# {title}\n", encoding="utf-8")
    return bundle


@pytest.fixture
def two_category_roots(tmp_path, monkeypatch):
    """Point bundle discovery at a temp training/ + studio/ pair with one bundle each."""
    training, studio = tmp_path / "training", tmp_path / "studio"
    _make_bundle(training, "01_anatomy", title="Anatomy lesson")
    _make_bundle(studio, "build_recording", title="Build a recording")
    monkeypatch.setattr(nb, "_CATEGORY_ROOTS", (training, studio))
    return training, studio


def test_bundles_dir_ships_the_anatomy_notebook():
    """The package carries the training bundle (notebook + README)."""
    root = bundles_dir()
    assert root.is_dir()
    assert list(root.rglob("*.ipynb")), "no .ipynb shipped under notebooks/training"
    assert (root / "01_anatomy" / "README.md").is_file()


def test_notebooks_lists_bundles_with_descriptions():
    """`notebooks()` discovers each bundle and reads its README title."""
    available = notebooks()
    assert "01_anatomy" in available
    # The description comes from the README title line, not the placeholder.
    assert available["01_anatomy"] != "(no description)"
    assert available["01_anatomy"]


def test_copy_copies_notebook_into_named_subdir_and_skips_videos(tmp_path):
    """A fresh copy lands under dest/<name> and never carries generated movies."""
    result = copy("01_anatomy", tmp_path)

    assert result == tmp_path / "01_anatomy"
    assert list(result.rglob("*.ipynb")), "notebook not copied"
    assert not list(result.rglob("*.avi")), "generated video leaked into the copy"


def test_copy_unknown_notebook_raises_keyerror(tmp_path):
    with pytest.raises(KeyError):
        copy("does_not_exist", tmp_path)


def test_description_falls_back_when_bundle_has_no_readme(tmp_path):
    """A bundle with no README still gets a (placeholder) description."""
    assert _description(tmp_path) == "(no description)"


def test_description_reads_the_readme_title(tmp_path):
    (tmp_path / "README.md").write_text("# My Title\n\nbody\n", encoding="utf-8")
    assert _description(tmp_path) == "My Title"


def test_print_table_handles_no_rows(capsys):
    _print_table([])
    assert capsys.readouterr().out == ""


def test_copy_refuses_existing_dir_without_force(tmp_path):
    copy("01_anatomy", tmp_path)
    with pytest.raises(FileExistsError):
        copy("01_anatomy", tmp_path)
    # force overwrites in place rather than raising.
    assert copy("01_anatomy", tmp_path, force=True) == tmp_path / "01_anatomy"


def test_cli_list_reports_a_notebook(capsys):
    assert main(["list"]) == 0
    assert "01_anatomy" in capsys.readouterr().out


def test_cli_list_on_empty_install_fails_cleanly(capsys, monkeypatch):
    """With no bundles discoverable, `list` reports and exits non-zero."""
    monkeypatch.setattr("minisim.notebooks.notebooks", dict)
    assert main(["list"]) == 1
    assert "No notebooks" in capsys.readouterr().err


def test_cli_copy_by_name_returns_zero_and_reports(tmp_path, capsys):
    assert main(["copy", "01_anatomy", "-o", str(tmp_path)]) == 0
    assert "copied 01_anatomy" in capsys.readouterr().out
    assert (tmp_path / "01_anatomy" / "01_anatomy.ipynb").is_file()


def test_cli_copy_all_copies_every_bundle(tmp_path):
    assert main(["copy", "--all", "-o", str(tmp_path)]) == 0
    for name in notebooks():
        assert (tmp_path / name).is_dir()


def test_cli_copy_without_name_fails_cleanly(tmp_path, capsys):
    assert main(["copy", "-o", str(tmp_path)]) == 1
    assert "name or --all" in capsys.readouterr().err


def test_cli_copy_unknown_notebook_fails_cleanly(tmp_path, capsys):
    assert main(["copy", "does_not_exist", "-o", str(tmp_path)]) == 1
    assert "unknown notebook" in capsys.readouterr().err


def test_cli_copy_existing_dir_fails_without_force(tmp_path):
    assert main(["copy", "01_anatomy", "-o", str(tmp_path)]) == 0
    assert main(["copy", "01_anatomy", "-o", str(tmp_path)]) == 1  # no --force


def test_cli_copy_force_overwrites_existing(tmp_path):
    assert main(["copy", "01_anatomy", "-o", str(tmp_path)]) == 0
    assert main(["copy", "01_anatomy", "-o", str(tmp_path), "--force"]) == 0


def test_cli_requires_a_subcommand():
    """Bare `minisim-notebooks` exits non-zero rather than doing nothing."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


# -- multi-category discovery (training/ + studio/) -------------------------


def test_discovery_spans_both_categories_training_first(two_category_roots):
    """`notebooks()` finds bundles in every category root, teaching ladder first."""
    names = list(notebooks())
    assert names == ["01_anatomy", "build_recording"]  # training listed before studio


def test_copy_finds_a_studio_bundle_by_name(two_category_roots, tmp_path):
    """A studio-category bundle is addressable by name alone, like a training one."""
    out = copy("build_recording", tmp_path / "dest")
    assert out == tmp_path / "dest" / "build_recording"
    assert (out / "build_recording.ipynb").is_file()


def test_missing_category_root_is_skipped_not_fatal(tmp_path, monkeypatch):
    """A category root that does not exist (partial install) is silently skipped."""
    training = tmp_path / "training"
    _make_bundle(training, "01_anatomy", title="Anatomy lesson")
    monkeypatch.setattr(nb, "_CATEGORY_ROOTS", (training, tmp_path / "studio"))
    assert list(notebooks()) == ["01_anatomy"]


def test_real_install_ships_the_studio_category():
    """The packaged tree carries the studio/ category beside training/."""
    studio = bundles_dir().parent / "studio"
    assert studio.is_dir(), "studio/ category root not shipped"
    assert (studio / "build_recording" / "README.md").is_file()
