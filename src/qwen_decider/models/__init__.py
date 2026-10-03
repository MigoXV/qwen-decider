"""Process-local registration; installed packages are never patched."""

from importlib.metadata import version

_registered = False


def register():
    global _registered
    if _registered:
        return
    expected = {"vllm": "0.14.1", "torch": "2.9.1", "transformers": "4.57.6"}
    for name, wanted in expected.items():
        if version(name).split("+")[0] != wanted:
            raise RuntimeError(f"{name} must remain {wanted}, got {version(name)}")
    # Worker lifetime must not exceed its API parent (Linux deployment).
    import ctypes
    import multiprocessing
    import os
    import signal
    import sys

    parent = multiprocessing.parent_process()
    if parent is not None and sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot install parent death signal")
        if os.getppid() != parent.pid:
            os.kill(os.getpid(), signal.SIGTERM)
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .config import Qwen3_5TextConfig

    AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
    _CONFIG_REGISTRY["qwen3_5_text"] = Qwen3_5TextConfig
    ModelRegistry.register_model(
        "Qwen3_5ForCausalLM", "qwen_decider.models.model:QwenDeciderForCausalLM"
    )

    _registered = True
