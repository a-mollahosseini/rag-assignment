"""Helpers for optional runtime dependencies."""

from __future__ import annotations


def missing_dependency_error(package: str, extra: str | None = None) -> RuntimeError:
    message = (
        f"Missing dependency: {package}. Install project requirements first:\n"
        "  pip install -r requirements.txt\n"
        "  pip install -e ."
    )
    if extra:
        message += f"\n{extra}"
    return RuntimeError(message)
