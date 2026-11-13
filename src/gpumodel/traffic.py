"""Logical ATen tensor traffic estimate, NOT a hardware DRAM-transaction measurement."""

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class LogicalTrafficMode(TorchDispatchMode):
    def __init__(self, device_type=None):
        super().__init__()
        self.device_type = device_type
        self.logical_bytes = 0
        self.ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        op_name = str(func).lower()
        # Views, layout metadata and aliasing do not necessarily access device DRAM.
        metadata_only = any(s in op_name for s in (
            "aten.view", "aten.as_strided", "aten.permute", "aten.transpose",
            "aten.expand", "aten.detach", "aten.unsqueeze", "aten.squeeze",
            "aten.alias", "aten.split", "aten.slice", "aten.select",
        ))
        if not metadata_only:
            self.logical_bytes += self._tensor_bytes(args)
            self.logical_bytes += self._tensor_bytes(kwargs)
            self.logical_bytes += self._tensor_bytes(result)
        self.ops += 1
        return result

    def _tensor_bytes(self, value):
        if isinstance(value, torch.Tensor):
            if self.device_type is not None and value.device.type != self.device_type:
                return 0
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(self._tensor_bytes(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(self._tensor_bytes(v) for v in value)
        return 0
