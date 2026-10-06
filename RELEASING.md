# Releasing alphafade

Releases are fully automated by `.github/workflows/release.yml` and use PyPI **Trusted
Publishing**: GitHub proves its identity to PyPI directly, so no API token or password is ever
stored in the repo or in GitHub secrets.

## One-time setup (done)

On <https://pypi.org> → *Your account* → *Publishing*, a pending publisher for `alphafade`
(owner `JSand15`, repository `alphafade`, workflow `release.yml`) was registered with the
environment name `testpypi`. The workflow tries the GitHub environment `pypi` first and falls
back to `testpypi`; both upload to pypi.org, so either registration works. The `pypi` GitHub
environment has a required reviewer (you approve the upload in the Actions tab).

TestPyPI is not used: it has no publisher for alphafade.

## Each release

```zsh
# 1. bump __version__ in src/alphafade/__init__.py and move "Unreleased" notes in
#    CHANGELOG.md under the new version, then commit
git tag v0.1.0
git push origin main v0.1.0
```

The workflow then:
1. checks the tag matches `__version__` and runs the tests,
2. builds the sdist and wheel once,
3. publishes them to **PyPI** (after your approval on the `pypi` environment, or via the
   `testpypi`-named fallback environment if that is the registered one),
4. installs that exact version back from PyPI and imports it.

If step 1 or 2 fails, nothing is uploaded. A version number can only be uploaded
once to each index, so fix the problem and tag a new patch version (e.g. `v0.1.1`).
