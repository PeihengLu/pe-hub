"""Discover and resolve PyTorch compute devices for training and inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
except Exception:  # pragma: no cover - optional in some environments
    torch = None  # type: ignore[assignment]

AUTO_DEVICE = "auto"

# Higher priority kinds are preferred when choosing the default accelerator.
_KIND_PRIORITY = {
    "cuda": 0,
    "xpu": 1,
    "mps": 2,
    "npu": 3,
    "tpu": 4,
    "cpu": 99,
}


@dataclass(frozen=True)
class DeviceDescriptor:
    """A selectable compute device."""

    device_id: str
    kind: str
    index: Optional[int]
    name: str
    is_accelerator: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _torch_ready() -> bool:
    return torch is not None


def _cuda_devices() -> List[DeviceDescriptor]:
    if not _torch_ready() or not torch.cuda.is_available():
        return []
    count = int(torch.cuda.device_count())
    devices: List[DeviceDescriptor] = []
    for index in range(count):
        try:
            props = torch.cuda.get_device_properties(index)
            name = props.name
        except Exception:  # noqa: BLE001
            name = f"CUDA device {index}"
        devices.append(
            DeviceDescriptor(
                device_id=f"cuda:{index}",
                kind="cuda",
                index=index,
                name=name,
                is_accelerator=True,
            )
        )
    return devices


def _mps_device() -> Optional[DeviceDescriptor]:
    if not _torch_ready():
        return None
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not torch.backends.mps.is_available():
        return None
    return DeviceDescriptor(
        device_id="mps",
        kind="mps",
        index=None,
        name="Apple Metal (MPS)",
        is_accelerator=True,
    )


def _xpu_devices() -> List[DeviceDescriptor]:
    if not _torch_ready():
        return []
    xpu_mod = getattr(torch, "xpu", None)
    if xpu_mod is None or not xpu_mod.is_available():
        return []
    count = int(xpu_mod.device_count())
    devices: List[DeviceDescriptor] = []
    for index in range(count):
        try:
            name = xpu_mod.get_device_name(index)
        except Exception:  # noqa: BLE001
            name = f"Intel XPU {index}"
        devices.append(
            DeviceDescriptor(
                device_id=f"xpu:{index}",
                kind="xpu",
                index=index,
                name=name,
                is_accelerator=True,
            )
        )
    return devices


def _cpu_device() -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id="cpu",
        kind="cpu",
        index=None,
        name="CPU",
        is_accelerator=False,
    )


def list_devices(*, include_cpu: bool = True) -> List[DeviceDescriptor]:
    """Return all devices usable for PE model training/inference."""
    devices: List[DeviceDescriptor] = []
    devices.extend(_cuda_devices())
    devices.extend(_xpu_devices())
    mps = _mps_device()
    if mps is not None:
        devices.append(mps)
    if include_cpu:
        devices.append(_cpu_device())

    devices.sort(
        key=lambda item: (
            _KIND_PRIORITY.get(item.kind, 50),
            item.index if item.index is not None else -1,
            item.device_id,
        )
    )
    return devices


def list_device_ids(*, include_cpu: bool = True) -> List[str]:
    return [device.device_id for device in list_devices(include_cpu=include_cpu)]


def list_accelerator_ids() -> List[str]:
    return [device.device_id for device in list_devices(include_cpu=False)]


def default_device_id(*, include_cpu_fallback: bool = True) -> str:
    """Return the best available device id (first accelerator, else CPU)."""
    accelerators = list_accelerator_ids()
    if accelerators:
        return accelerators[0]
    if include_cpu_fallback:
        return "cpu"
    raise RuntimeError("No compute devices available")


def resolve_device_id(device_id: Optional[str] = None) -> str:
    """Resolve ``auto``/``None`` to the default device or validate an explicit id."""
    if device_id is None or device_id == AUTO_DEVICE:
        return default_device_id()
    known = set(list_device_ids(include_cpu=True))
    if device_id not in known:
        raise ValueError(f"Unknown device '{device_id}'. Available: {sorted(known)}")
    return device_id


def resolve_device(device_id: Optional[str] = None) -> "torch.device":
    """Resolve a device id string to ``torch.device``."""
    if not _torch_ready():
        raise RuntimeError("PyTorch is required to resolve devices")
    resolved = resolve_device_id(device_id)
    return torch.device(resolved)


def device_to_lightning_accelerator(device: "torch.device") -> str:
    """Map ``torch.device`` to a PyTorch Lightning accelerator name."""
    if device.type == "cuda":
        return "gpu"
    if device.type == "mps":
        return "mps"
    if device.type == "xpu":
        return "xpu"
    return "cpu"


def cuda_index_from_device(device: "torch.device") -> int:
    """Return the CUDA device index for vendor code that expects an integer gpu id."""
    if device.type != "cuda":
        return 0
    return int(device.index or 0)


def format_devices_for_cli(devices: Optional[Sequence[DeviceDescriptor]] = None) -> str:
    devices = list(devices or list_devices())
    lines = ["Available devices:"]
    for device in devices:
        tag = "accelerator" if device.is_accelerator else "cpu"
        lines.append(f"  {device.device_id:8}  [{tag:11}]  {device.name}")
    lines.append(f"Default: {default_device_id()}")
    return "\n".join(lines)
