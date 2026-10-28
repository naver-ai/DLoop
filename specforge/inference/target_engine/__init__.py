# coding=utf-8
# Derived from SpecForge (https://github.com/sgl-project/SpecForge)
# MIT License
# Modified for DLoop

"""Target engines: the backend-agnostic capture surface (TargetEngine)."""

from .base import KNOWN_BACKENDS, TargetEngine

__all__ = [
    "TargetEngine",
    "KNOWN_BACKENDS",
]

# NOTE: the DFlash engines (dflash_target_model) are intentionally NOT eagerly
# imported here — that module imports sglang internals unconditionally, and this
# package must stay importable without the pinned sglang. Import them from the
# submodule.
