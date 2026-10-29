#!/usr/bin/env bash
# Run the checks locally, then create and push the release tag.
#
#   scripts/release.sh 0.1.0rc1     pre-release, published to TestPyPI
#   scripts/release.sh 0.1.0        release, published to PyPI after approval
#
# This script never publishes anything itself: pushing the tag starts the
# release workflow in GitHub Actions. Run it from the project's environment
# (nox and git must be on PATH).
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 VERSION   (for example 0.1.0 or 0.1.0rc1)" >&2
    exit 2
fi
version="$1"
tag="v${version}"
cd "$(dirname "$0")/.."

fail() { echo "error: $*" >&2; exit 1; }

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]] \
    || fail "VERSION must look like 0.1.0 or 0.1.0rc1, got '$version'"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || fail "run this on the main branch"
[[ -z "$(git status --porcelain)" ]] || fail "the working tree is not clean"
git fetch --quiet origin main --tags
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] \
    || fail "local main differs from origin/main (pull or push first)"
! git rev-parse -q --verify "refs/tags/${tag}" >/dev/null || fail "tag ${tag} already exists"
python scripts/release_tools.py notes "$tag" >/dev/null \
    || fail "fix CHANGELOG.md for ${tag} first"

nox -s lint typecheck tests build

echo
echo "All checks passed. About to tag and push:"
echo "  tag:     ${tag} ($(git rev-parse --short HEAD))"
echo "  target:  $([[ "$version" =~ (a|b|rc) ]] && echo TestPyPI || echo PyPI)"
read -r -p "Create and push ${tag}? [y/N] " answer
[[ "$answer" == "y" || "$answer" == "Y" ]] || fail "aborted, nothing was pushed"

git tag -a "$tag" -m "intelloptim ${version}"
git push origin "$tag"
echo "Pushed ${tag}. Follow the release workflow in the Actions tab."
