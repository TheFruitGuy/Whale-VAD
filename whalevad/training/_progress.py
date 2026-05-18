"""Optional tqdm progress bar with graceful fallback.

If ``tqdm`` is installed the helpers return a real progress bar; otherwise
they're transparent no-ops, so the training pipeline runs unchanged.
"""

from __future__ import annotations

from typing import Iterable, Optional, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore

    HAS_TQDM = True
except ImportError:  # pragma: no cover - tqdm is optional
    HAS_TQDM = False
    _tqdm = None  # type: ignore[assignment]


def progress(
    iterable: Iterable[T],
    *,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    leave: bool = True,
    disable: bool = False,
    unit: str = "it",
):
    """Wrap ``iterable`` with a tqdm bar when available."""
    if not HAS_TQDM or disable:
        return iterable
    return _tqdm(iterable, desc=desc, total=total, leave=leave, unit=unit)
