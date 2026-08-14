"""Phase 0 smoke test: proves the package imports and CI is wired up end to end."""

from app import __version__


def test_package_imports_and_reports_a_version() -> None:
    assert __version__
