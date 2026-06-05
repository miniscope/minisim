# Releasing minisim to PyPI

minisim is published to PyPI by the `Publish to PyPI` GitHub Actions workflow
(`.github/workflows/publish.yml`) using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC),
so no API tokens are stored in this repo.

The package version is derived from the **git tag** by `pdm-backend`'s SCM
source (see `[tool.pdm.version]` in `pyproject.toml`). There is no
`version = "..."` field to keep in sync: the version that ends up on PyPI is
whatever you tag. Builds between tags get a PEP 440 dev version like
`0.1.0.dev41+g<sha>`.

## One-time setup

On [pypi.org](https://pypi.org/manage/account/publishing/), add a trusted
publisher for the `minisim` project:
- Repository: `miniscope/minisim`
- Workflow: `publish.yml`
- Environment: `pypi`

Then create a GitHub environment named `pypi` in the repo settings (Settings ->
Environments). The `publish` job references it.

## Cutting a release

From a clean `main`:

```bash
# Auto-bump from conventional commits, write the changelog, and create the tag.
# `version_provider = "scm"` means cz reads the current version from git and
# does NOT touch source files.
pdm run cz bump

# Push the commit and the tag; the v* tag push triggers the publish workflow.
git push origin main --follow-tags
```

To set the version manually instead, tag directly:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

On a `v*` tag push the workflow builds an sdist + pure-Python wheel (version
derived from the tag), runs `twine check`, and uploads to PyPI.

## Dry run

Trigger the workflow via `workflow_dispatch` (Actions tab -> Publish to PyPI ->
Run workflow) to build the artifacts and upload them to the run summary
**without publishing**. Useful for validating metadata before tagging.
