"""Ascend cube core count for persistent ``T.Kernel(core_num, is_npu=True)`` launches."""

from __future__ import annotations

import torch

try:
    CORE_NUM = int(torch.npu.get_device_properties("npu").cube_core_num)
except Exception:
    CORE_NUM = 24
