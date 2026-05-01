import platform
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class MacChipInfo:
    is_macos: bool
    is_apple_silicon: bool
    machine: str
    processor: str


def get_mac_chip_info() -> MacChipInfo:
    is_macos = platform.system().lower() == "darwin"
    machine = platform.machine() or ""
    processor = platform.processor() or ""
    # Apple Silicon Macs report arm64.
    is_apple_silicon = bool(is_macos and machine.lower() == "arm64")
    return MacChipInfo(
        is_macos=is_macos,
        is_apple_silicon=is_apple_silicon,
        machine=machine,
        processor=processor,
    )


def get_best_torch_device(prefer_mps: bool = True) -> torch.device:
    """
    選擇最適合的 PyTorch device：
    - NVIDIA GPU：cuda
    - Apple Silicon（且 PyTorch 支援）：mps
    - 其他：cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_mps:
        mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if mps_built and mps_available:
            return torch.device("mps")

    return torch.device("cpu")


def get_device_display_info(device: torch.device) -> Tuple[str, Optional[str]]:
    """
    回傳 (device_type, device_detail) 用於顯示/紀錄。
    """
    if device.type == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return "cuda", f"{name} ({total_mem:.2f} GB)"

    if device.type == "mps":
        chip = get_mac_chip_info()
        if chip.is_apple_silicon:
            return "mps", f"Apple Silicon ({chip.machine})"
        return "mps", "Apple MPS"

    return "cpu", "CPU"


def get_dataloader_kwargs_for_device(device: torch.device) -> Dict[str, Any]:
    """
    依 device 給 DataLoader 的建議參數（穩定性優先）。
    - pin_memory：對 cuda 有幫助；對 mps/cpu 通常無益
    - num_workers：mps 在部分環境上多 worker 可能不穩，預設改 0
    """
    if device.type == "cuda":
        return {"num_workers": 4, "pin_memory": True}
    if device.type == "mps":
        return {"num_workers": 0, "pin_memory": False}
    return {"num_workers": 4, "pin_memory": False}
