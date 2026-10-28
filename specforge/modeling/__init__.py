# Derived from SpecForge (https://github.com/sgl-project/SpecForge)
# MIT License
# Modified for DLoop

from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .target import (
    Eagle3TargetEngine,
    HFEagle3TargetEngine,
    HFEagle3TargetModel,
    SGLangEagle3TargetEngine,
    SGLangEagle3TargetModel,
    TargetEngine,
    get_eagle3_target_model,
)

__all__ = [
    "LlamaForCausalLMEagle3",
    # Generic (Phase B) surface
    "TargetEngine",
    "Eagle3TargetEngine",
    "SGLangEagle3TargetEngine",
    "HFEagle3TargetEngine",
    "get_eagle3_target_model",
    # Back-compat aliases (pre-Phase-B names)
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
]
