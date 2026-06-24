"""Sphinx configuration for the Minisim documentation site.

Docs are authored in Markdown (MyST). The API reference renders from the live
package: prose docstrings via autodoc + napoleon, and the typed Spec/steps via
autodoc-pydantic, so the reference cannot drift from the code.
"""

import logging
from importlib.metadata import version as _dist_version

# -- Project information -----------------------------------------------------

project = "Minisim"
author = "Daniel Aharoni"
copyright = "2026, Daniel Aharoni"  # noqa: A001 (Sphinx requires this name)

# Full version from the installed distribution (pdm-backend derives it from the
# `v*` git tags); `release` is the full string, `version` the short X.Y.
release = _dist_version("minisim")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",  # Markdown parser + notebook rendering (supersedes myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_copybutton",
    "sphinx_design",
]

# Treat warnings (broken refs, autodoc import failures) as errors on RTD so the
# build cannot silently ship a half-rendered reference.
nitpicky = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST / notebooks --------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # ::: fenced directives, nicer in Markdown
    "deflist",
    "dollarmath",  # $...$ math for the optics/physics pages
]
myst_heading_anchors = 3

# Do NOT execute notebooks at build time. The anatomy notebook runs a full
# forward simulation and uses ipywidgets (which do not render statically); it is
# rendered from its committed outputs. Flip to "auto" once it ships outputs.
nb_execution_mode = "off"

# -- autodoc / autodoc-pydantic ----------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Render the pydantic models as configuration tables: fields with types,
# defaults, and constraints, but keep validator/JSON noise out of the page.
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = True
autodoc_pydantic_model_member_order = "bysource"

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Minisim"
html_static_path = ["_static"]
html_favicon = "_static/logo/minisim_icon.png"

html_theme_options = {
    "logo": {
        # the 'M' icon (dark tile reads on both light and dark navbars)
        "image_light": "_static/logo/minisim_icon.png",
        "image_dark": "_static/logo/minisim_icon.png",
        "alt_text": "Minisim",
    },
    "github_url": "https://github.com/miniscope/minisim",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/minisim/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    "navbar_align": "left",
}

html_context = {
    "github_user": "miniscope",
    "github_repo": "minisim",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Resilient intersphinx ---------------------------------------------------

# Inventory hosts (notably docs.scipy.org) intermittently time out from the build
# network. Under `-W` that turns a transient ConnectTimeout into a hard build
# failure even though nothing in the docs is wrong. Drop only the intersphinx
# "failed to reach any of the inventories" warning so flaky hosts cannot break the
# build; cross-refs into a missing inventory simply render as plain text (nitpicky
# is off), and every real warning (broken refs, autodoc import failures) still fails.


class _IgnoreUnreachableInventory(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "failed to reach any of the inventories" not in record.getMessage()


def setup(app):
    # Filter the intersphinx logger directly, so the record is dropped in
    # Logger.handle (before any handler, incl. Sphinx's warning-is-error one, runs).
    from sphinx.util import logging as sphinx_logging

    sphinx_logging.getLogger("sphinx.ext.intersphinx").logger.addFilter(
        _IgnoreUnreachableInventory()
    )
