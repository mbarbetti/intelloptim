"""``scripts/release_tools.py`` — tag parsing, changelog notes, dist checks."""

from __future__ import annotations

import importlib.util
import io
import os
import tarfile
import zipfile
from typing import Any, Iterable

import pytest

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "release_tools.py"
)
_SPEC = importlib.util.spec_from_file_location("release_tools", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
rt: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rt)

_CHANGELOG = """# Changelog

## [Unreleased]

### Added

- Something in progress.

## [0.2.0] - 2026-11-01

### Fixed

- A bug.

## [0.1.0] - 2026-10-30

First release.

[Unreleased]: https://example.org/compare/v0.2.0...HEAD
[0.2.0]: https://example.org/compare/v0.1.0...v0.2.0
"""


class TestParseTag:
    @pytest.mark.parametrize("tag", ["v0.1.0", "v1.2.3", "v0.1.0rc1", "v2.0.0b3"])
    def test_accepts_release_and_prerelease_tags(self, tag: str) -> None:
        assert str(rt.parse_tag(tag)) == tag[1:]

    @pytest.mark.parametrize(
        "tag", ["0.1.0", "v", "vfoo", "v1.0.0+local", "v1.0.0.dev1", "v1!2.0.0"]
    )
    def test_rejects_other_tags(self, tag: str) -> None:
        with pytest.raises(ValueError):
            rt.parse_tag(tag)

    def test_prerelease_flag(self) -> None:
        assert rt.is_prerelease("v0.1.0rc1")
        assert rt.is_prerelease("v0.1.0a1")
        assert not rt.is_prerelease("v0.1.0")


class TestChangelogSection:
    def test_final_release_uses_its_own_section(self) -> None:
        body = rt.changelog_section(_CHANGELOG, "v0.2.0")
        assert body == "### Fixed\n\n- A bug."

    def test_oldest_section_stops_at_the_link_references(self) -> None:
        assert rt.changelog_section(_CHANGELOG, "v0.1.0") == "First release."

    def test_prerelease_uses_the_unreleased_section(self) -> None:
        assert "Something in progress" in rt.changelog_section(_CHANGELOG, "v0.3.0rc1")

    def test_missing_section(self) -> None:
        with pytest.raises(ValueError, match=r"\[0.9.0\]"):
            rt.changelog_section(_CHANGELOG, "v0.9.0")

    def test_empty_section(self) -> None:
        text = "## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n\nText.\n"
        with pytest.raises(ValueError, match="empty"):
            rt.changelog_section(text, "v0.1.0rc1")

    def test_version_is_matched_exactly(self) -> None:
        text = "## [0.1.0] - x\n\nOnly this.\n"
        with pytest.raises(ValueError):
            rt.changelog_section(text, "v0.1")  # normalises to 0.1, not 0.1.0

    def test_the_repository_changelog_describes_the_first_release(self) -> None:
        path = os.path.join(os.path.dirname(_PATH), "..", "CHANGELOG.md")
        with open(path, encoding="utf-8") as handle:
            assert rt.changelog_section(handle.read(), "v0.1.0")


def _touch(directory: Any, *names: str) -> None:
    for name in names:
        (directory / name).write_bytes(b"")


class TestCheckDist:
    def test_matching_version(self, tmp_path: Any) -> None:
        _touch(
            tmp_path,
            "intelloptim-0.1.0rc1-py3-none-any.whl",
            "intelloptim-0.1.0rc1.tar.gz",
        )
        rt.check_dist(str(tmp_path), "v0.1.0rc1")

    def test_mismatching_version(self, tmp_path: Any) -> None:
        _touch(tmp_path, "intelloptim-0.1.1-py3-none-any.whl")
        with pytest.raises(ValueError, match="0.1.1"):
            rt.check_dist(str(tmp_path), "v0.1.0")

    def test_dev_version_from_an_untagged_commit_is_refused(
        self, tmp_path: Any
    ) -> None:
        _touch(tmp_path, "intelloptim-0.1.1.dev3+gabc123-py3-none-any.whl")
        with pytest.raises(ValueError):
            rt.check_dist(str(tmp_path), "v0.1.0")

    def test_empty_directory(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="no distribution"):
            rt.check_dist(str(tmp_path), "v0.1.0")


def _wheel(path: Any, names: Iterable[str]) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        for name in names:
            wheel.writestr(name, "")


def _sdist(path: Any, names: Iterable[str]) -> None:
    with tarfile.open(path, "w:gz") as sdist:
        for name in names:
            info = tarfile.TarInfo(f"intelloptim-0.1.0/{name}")
            sdist.addfile(info, io.BytesIO(b""))


_GOOD_SDIST = ["LICENSE", "NOTICE", "README.md", "CHANGELOG.md", "pyproject.toml"]
_GOOD_WHEEL = [
    "intelloptim/__init__.py",
    "intelloptim/py.typed",
    "intelloptim-0.1.0.dist-info/METADATA",
]


class TestCheckContents:
    def _build(self, tmp_path: Any, wheel: Iterable[str], sdist: Iterable[str]) -> str:
        _wheel(tmp_path / "intelloptim-0.1.0-py3-none-any.whl", wheel)
        _sdist(tmp_path / "intelloptim-0.1.0.tar.gz", sdist)
        return str(tmp_path)

    def test_good_distributions(self, tmp_path: Any) -> None:
        rt.check_contents(self._build(tmp_path, _GOOD_WHEEL, _GOOD_SDIST))

    def test_wheel_without_py_typed(self, tmp_path: Any) -> None:
        wheel = [n for n in _GOOD_WHEEL if not n.endswith("py.typed")]
        with pytest.raises(ValueError, match="py.typed"):
            rt.check_contents(self._build(tmp_path, wheel, _GOOD_SDIST))

    def test_wheel_with_tests(self, tmp_path: Any) -> None:
        wheel = [*_GOOD_WHEEL, "tests/unit/test_x.py"]
        with pytest.raises(ValueError, match="outside the package"):
            rt.check_contents(self._build(tmp_path, wheel, _GOOD_SDIST))

    @pytest.mark.parametrize("missing", ["LICENSE", "NOTICE", "CHANGELOG.md"])
    def test_sdist_without_a_required_file(self, tmp_path: Any, missing: str) -> None:
        sdist = [n for n in _GOOD_SDIST if n != missing]
        with pytest.raises(ValueError, match=missing):
            rt.check_contents(self._build(tmp_path, _GOOD_WHEEL, sdist))

    def test_needs_exactly_one_of_each(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="one wheel and one sdist"):
            rt.check_contents(str(tmp_path))


class TestCommandLine:
    def test_prerelease(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert rt.main(["prerelease", "v0.1.0rc2"]) == 0
        assert capsys.readouterr().out.strip() == "true"
        assert rt.main(["prerelease", "v0.1.0"]) == 0
        assert capsys.readouterr().out.strip() == "false"

    def test_errors_return_a_failure_status(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert rt.main(["prerelease", "nonsense"]) == 1
        assert "error:" in capsys.readouterr().err

    def test_notes_for_the_first_release(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert rt.main(["notes", "v0.1.0"]) == 0
        assert capsys.readouterr().out.strip()
