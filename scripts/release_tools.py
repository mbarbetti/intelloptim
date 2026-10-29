"""Helpers shared by ``scripts/release.sh``, ``noxfile.py`` and the release workflow.

Commands (run ``python scripts/release_tools.py --help``):

* ``notes TAG``            print the changelog text of the release
* ``prerelease TAG``       print ``true`` or ``false``
* ``check-dist DIR TAG``   check that the built files carry the version of TAG
* ``check-contents DIR``   check what the wheel and the sdist contain
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import tarfile
import zipfile
from typing import List, Optional

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

_CHANGELOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "CHANGELOG.md"
)


def parse_tag(tag: str) -> Version:
    """Return the version of a release tag such as ``v0.1.0`` or ``v0.1.0rc1``.

    Args:
        tag (str): The tag name, with the leading ``v``.

    Returns:
        Version: The parsed version.

    Raises:
        ValueError: If the tag is not ``v`` followed by a valid version.
    """
    if not tag.startswith("v"):
        raise ValueError(f"release tags start with 'v': {tag!r}")
    try:
        version = Version(tag[1:])
    except InvalidVersion as exc:
        raise ValueError(f"not a valid version: {tag!r}") from exc
    if version.local is not None or version.epoch or version.is_devrelease:
        raise ValueError(f"unsupported version in tag: {tag!r}")
    return version


def is_prerelease(tag: str) -> bool:
    """Whether *tag* names a pre-release (``a``, ``b`` or ``rc``)."""
    return parse_tag(tag).is_prerelease


def changelog_section(text: str, tag: str) -> str:
    """Return the changelog entry that describes the release *tag*.

    A final release uses its own ``## [X.Y.Z]`` section. A pre-release has no
    section of its own and uses ``## [Unreleased]``.

    Args:
        text (str): Content of ``CHANGELOG.md``.
        tag (str): The release tag.

    Returns:
        str: The body of the section, stripped.

    Raises:
        ValueError: If the section is missing or empty.
    """
    version = parse_tag(tag)
    name = "Unreleased" if version.is_prerelease else str(version)
    heading = re.compile(rf"^## \[{re.escape(name)}\](?: .*)?$", re.MULTILINE)
    match = heading.search(text)
    if match is None:
        raise ValueError(f"CHANGELOG.md has no '## [{name}]' section")
    rest = text[match.end() :]
    following = re.search(r"^## \[|^\[[^\]]+\]: ", rest, re.MULTILINE)
    body = (rest[: following.start()] if following else rest).strip()
    if not body:
        raise ValueError(f"the '## [{name}]' section of CHANGELOG.md is empty")
    return body


def check_dist(directory: str, tag: str) -> None:
    """Check that every built file in *directory* has the version of *tag*.

    Raises:
        ValueError: If a file has another version, or no distribution is found.
    """
    expected = parse_tag(tag)
    files = sorted(glob.glob(os.path.join(directory, "*.whl"))) + sorted(
        glob.glob(os.path.join(directory, "*.tar.gz"))
    )
    if not files:
        raise ValueError(f"no distribution found in {directory}")
    for path in files:
        name = os.path.basename(path)
        if name.endswith(".whl"):
            found = parse_wheel_filename(name)[1]
        else:
            found = parse_sdist_filename(name)[1]
        if found != expected:
            raise ValueError(f"{name} has version {found}, the tag says {expected}")


def check_contents(directory: str) -> None:
    """Check what the wheel and the sdist of ``intelloptim`` contain.

    The wheel must carry the package and ``py.typed`` and nothing from the test
    suite; the sdist must carry the licence files and the changelog.

    Raises:
        ValueError: If something is missing or does not belong there.
    """
    wheels = glob.glob(os.path.join(directory, "*.whl"))
    sdists = glob.glob(os.path.join(directory, "*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"expected one wheel and one sdist in {directory}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
    if "intelloptim/py.typed" not in names:
        raise ValueError("the wheel lacks intelloptim/py.typed")
    stray = [n for n in names if not n.startswith(("intelloptim/", "intelloptim-"))]
    if stray:
        raise ValueError(f"the wheel contains files outside the package: {stray[:5]}")
    with tarfile.open(sdists[0]) as sdist:
        members = [m.name.split("/", 1)[-1] for m in sdist.getmembers()]
    for required in (
        "LICENSE",
        "NOTICE",
        "README.md",
        "CHANGELOG.md",
        "pyproject.toml",
    ):
        if required not in members:
            raise ValueError(f"the sdist lacks {required}")


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point; returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("notes").add_argument("tag")
    sub.add_parser("prerelease").add_argument("tag")
    check = sub.add_parser("check-dist")
    check.add_argument("directory")
    check.add_argument("tag")
    sub.add_parser("check-contents").add_argument("directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "notes":
            with open(_CHANGELOG, encoding="utf-8") as handle:
                print(changelog_section(handle.read(), args.tag))
        elif args.command == "prerelease":
            print("true" if is_prerelease(args.tag) else "false")
        elif args.command == "check-dist":
            check_dist(args.directory, args.tag)
        else:
            check_contents(args.directory)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
