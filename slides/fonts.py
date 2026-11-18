"""Locate the DejaVu Sans faces used for text measurement and preview rendering.

DejaVu ships inside matplotlib, so an environment that can build the figures can
also measure text for the slides; the usual system font directories are the
fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

NAMES = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")


def _candidates() -> list[Path]:
    dirs: list[Path] = []
    try:                                    # matplotlib bundles the TTFs
        import matplotlib
        dirs.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except ImportError:
        pass
    try:                                    # whatever fontconfig resolved
        from matplotlib import font_manager as fm
        dirs.append(Path(fm.findfont("DejaVu Sans")).parent)
    except Exception:
        pass
    dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
             Path("/Library/Fonts")]
    return dirs


def dejavu_paths() -> tuple[Path, Path]:
    """Return (regular, bold) paths for DejaVu Sans, or exit with a hint."""
    for d in _candidates():
        hits = list(d.rglob(NAMES[0])) + list(d.rglob(NAMES[1]))
        if len(hits) >= 2:
            reg = next(p for p in hits if p.name == NAMES[0])
            bold = next(p for p in hits if p.name == NAMES[1])
            return reg, bold
    sys.exit("DejaVu Sans not found. Install matplotlib (it bundles the font) or a "
             "fonts-dejavu package, then re-run.")
