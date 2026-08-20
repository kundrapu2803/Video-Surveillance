"""Thread/env setup. MUST be imported (and configure_runtime() called) before
torch is imported anywhere in the process -- see plan section M2.

Defaults are measured on this machine (Intel i5-1235U, 10C/12T heterogeneous
P+E cores): 4 threads gave the best sgemm throughput (374 GFLOP/s vs 361 at 10
threads), and KMP_AFFINITY must NOT be set because pinning would fight Windows
Thread Director's P/E-core scheduling on a heterogeneous part.
"""
from __future__ import annotations

import os
import sys

_configured = False


def configure_runtime(threads: int = 4) -> None:
    global _configured
    if _configured:
        return

    if "torch" in sys.modules:
        raise RuntimeError(
            "torch was imported before cctv.runtime.configure_runtime() -- "
            "thread-count env vars only take effect if set before torch's "
            "native libraries initialize. Fix the import order."
        )

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    # Deliberately NOT setting KMP_AFFINITY -- see module docstring.

    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass

    _configured = True


def assert_precision_supported(half: bool, device: str) -> None:
    if half and device == "cpu":
        raise ValueError(
            "--half on CPU is refused: this machine's CPU (Alder Lake-U) has no "
            "AVX512-FP16/AMX-BF16, so emulated fp16 convolution is SLOWER than "
            "fp32, not faster. Drop --half or pick a GPU device."
        )
