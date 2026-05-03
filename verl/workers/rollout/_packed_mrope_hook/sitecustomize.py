"""sitecustomize.py — registers a Packed-MROPE Qwen3-VL subclass with vLLM.

Auto-imported by site.py during every Python interpreter startup, including
the EngineCore subprocess that vLLM v1 spawns
(VLLM_WORKER_MULTIPROC_METHOD=spawn). PYTHONPATH must include this file's
directory.

Per-request packed positions are read from ``${PACKED_MROPE_SHM_DIR}``
(default ``/dev/shm/packed_mrope``), keyed by the SHA-1 of the prefill
token tuple. Wire format:

    int32 delta (LE) | int32 L (LE) | int64[3 * L] positions (row-major)

On miss, falls through to the default mrope position computation, so
non-multiturn paths and runs with ``past_k_steps=None`` are untouched.

See ``docs/agent/implementation/past_k_packed_multiturn.md`` for the full
design.
"""

import os
import sys


def _install_packed_qwen3_vl():
    # Gate: only run when explicitly enabled. Otherwise every Python interpreter
    # startup (including Ray's 128 prestart workers) would import vllm and pin
    # the CPUs. Set PACKED_MROPE_ENABLED=1 in the actor/rollout launch script.
    if os.environ.get("PACKED_MROPE_ENABLED") != "1":
        return

    # Skip Ray prestart workers — they never load vllm but pay the import cost.
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0.endswith("default_worker.py") or argv0.endswith("setup_worker.py"):
        return

    try:
        from vllm import ModelRegistry
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )
    except Exception:
        # vllm not installed in this Python — non-vLLM processes simply skip.
        return

    import hashlib

    shm_dir = os.environ.get("PACKED_MROPE_SHM_DIR", "/dev/shm/packed_mrope")
    marker_path = os.environ.get("PACKED_MROPE_MARKER")  # optional debug log

    def _key(input_tokens):
        return hashlib.sha1(
            b",".join(str(int(t)).encode() for t in input_tokens)
        ).hexdigest()[:16]

    class PackedQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
        def get_mrope_input_positions(self, input_tokens, mm_features):
            import struct

            import numpy as np
            import torch

            key = _key(input_tokens)
            override_path = os.path.join(shm_dir, f"{key}.bin")
            hit = os.path.exists(override_path)

            if marker_path:
                try:
                    with open(marker_path, "a") as f:
                        f.write(
                            f"{'HIT' if hit else 'MISS'} pid={os.getpid()} "
                            f"tokens_len={len(input_tokens)} key={key}\n"
                        )
                except Exception:
                    pass

            if hit:
                with open(override_path, "rb") as f:
                    delta, L = struct.unpack("<ii", f.read(8))
                    arr = np.frombuffer(f.read(3 * L * 8), dtype=np.int64)
                positions = torch.from_numpy(arr.reshape(3, L).copy()).contiguous()
                return positions, int(delta)

            return super().get_mrope_input_positions(input_tokens, mm_features)

    ModelRegistry.register_model(
        "Qwen3VLForConditionalGeneration",
        PackedQwen3VLForConditionalGeneration,
    )
    print(
        f"[packed_mrope_hook pid={os.getpid()}] "
        f"Registered PackedQwen3VLForConditionalGeneration (shm_dir={shm_dir})",
        file=sys.stderr,
    )


_install_packed_qwen3_vl()
