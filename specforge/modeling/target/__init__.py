# Derived from SpecForge (https://github.com/sgl-project/SpecForge)
# MIT License
# Modified for DLoop

from specforge.inference.target_engine.base import KNOWN_BACKENDS, TargetEngine
from specforge.inference.target_engine.eagle3_target_model import (
    Eagle3TargetEngine,
    Eagle3TargetModel,
    HFEagle3TargetEngine,
    HFEagle3TargetModel,
    SGLangEagle3TargetEngine,
    SGLangEagle3TargetModel,
    SGLangServerEagle3TargetEngine,
    get_eagle3_target_model,
)

from .target_head import TargetHead

__all__ = [
    # Generic (Phase B) surface
    "TargetEngine",
    "KNOWN_BACKENDS",
    # EAGLE3 engines
    "Eagle3TargetEngine",
    "SGLangEagle3TargetEngine",
    "HFEagle3TargetEngine",
    "SGLangServerEagle3TargetEngine",
    "get_eagle3_target_model",
    # Back-compat aliases (pre-Phase-B names)
    "Eagle3TargetModel",
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "TargetHead",
]

# NOTE: the DFlash engines are intentionally NOT eagerly imported here — that
# module imports sglang internals unconditionally, and this package must stay
# importable without the pinned sglang. Import them from
# specforge.inference.target_engine.dflash_target_model.
