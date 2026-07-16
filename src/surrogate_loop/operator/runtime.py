from __future__ import annotations

import platform
import random
from importlib.metadata import version
from typing import Any

import numpy as np


def require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "神经算子依赖未安装，请运行 uv sync --extra operator --all-groups"
        ) from error
    return torch


def resolve_device(requested: str) -> Any:
    torch = require_torch()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但 torch.cuda.is_available() 为 False")
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError(f"未知设备策略：{requested}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return {
        "python": platform.python_version(),
        "torch": version("torch"),
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
    }
