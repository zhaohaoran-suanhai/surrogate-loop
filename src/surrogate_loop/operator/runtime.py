from __future__ import annotations

import platform
import random
import warnings
from importlib.metadata import version
from typing import Any

import numpy as np

_last_device_fallback_reason: str | None = None


def require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "神经算子依赖未安装，请运行 uv sync --extra operator --all-groups"
        ) from error
    return torch


def resolve_device(requested: str) -> Any:
    global _last_device_fallback_reason
    _last_device_fallback_reason = None
    torch = require_torch()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但 torch.cuda.is_available() 为 False")
        _probe_cuda(torch)
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError(f"未知设备策略：{requested}")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _probe_cuda(torch)
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError as error:
        _last_device_fallback_reason = f"CUDA 初始化探测失败：{error}"
        warnings.warn(
            f"{_last_device_fallback_reason}；已回退 CPU",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return torch.device("cuda")


def seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_summary(device: Any) -> dict[str, object]:
    torch = require_torch()
    device_name = (
        torch.cuda.get_device_name(device) if getattr(device, "type", None) == "cuda" else "CPU"
    )
    summary: dict[str, object] = {
        "python": platform.python_version(),
        "torch": version("torch"),
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
    }
    if _last_device_fallback_reason is not None:
        summary["device_fallback_reason"] = _last_device_fallback_reason
    return summary


def _probe_cuda(torch: Any) -> None:
    tensor = torch.empty(1, device="cuda")
    del tensor
