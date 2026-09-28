# Releasing alphafade

Releases are fully automated by `.github/workflows/release.yml` and use PyPI **Trusted
Publishing**: GitHub proves its identity to PyPI directly, so no API token or password is ever
stored in the repo or in GitHub secrets.

## One-time setup (Jeevun does this once, in a browser)

1. **TestPyPI:** log in at <https://test.pypi.org> → *Your account* → *Publishing* →
   *Add a new pending publisher*:
   - PyPI project name: `alphafade`
   - Owner: `JSand15`
   - Repository name: `alphafade`
   - Workflow name: `release.yml`
   - Environment name: `testpypi`
2. **PyPI:** the same at <https://pypi.org> → *Your account* → *Publishing*, but with
   environment name `pypi`.
3. **GitHub environments:** repo → *Settings* → *Environments* → create `testpypi` and
   `pypi`. Optional but recommended: on `pypi`, add yourself as a *required reviewer*, so the
   real PyPI upload waits for your click after the TestPyPI check passes.

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
3. publishes them to **TestPyPI**,
4. installs that exact version back from TestPyPI and imports it,
5. publishes the same files to **PyPI** (after your approval, if you set a reviewer).

If step 3 or 4 fails, nothing reaches the real PyPI. A version number can only be uploaded
once to each index, so fix the problem and tag a new patch version (e.g. `v0.1.1`).
