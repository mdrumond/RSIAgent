# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0; see LICENSE.CANNBot.
# Exact function retained from cf33f091aa645731cf5a47ade64434b3c60b55ae75591c9226e1e769fed5707a.
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)
BAD_ROW_K = 50
BAD_ROW_FLOOR_US = 10.0

def _aggregate_kernel_rows(name: str, durations, active_count: int) -> Tuple[float, int]:
    """做法 B：以中位数为基准剔除坏行，再按迭代段内求和、跨迭代取均值。

    kernel_details.csv 与 msprof 旧格式（op_summary / task_time）共用的聚合逻辑：
    - 识别坏行：阈值 = max(median × BAD_ROW_K, BAD_ROW_FLOOR_US)，超阈值行剔除，
      以中位数为基准天然规避 MAD≈0 的退化边界；
    - 聚合：剩余行按 active_count 迭代切段、段内直接求和（= 该迭代该 kernel 总耗时，
      与 framework 侧"实际总耗时=求和"口径一致），跨迭代取均值。
    返回 (单次调用平均耗时 avg_us, 发射次数 launch_count)。
    """
    if active_count <= 0 or durations is None or len(durations) == 0:
        return 0.0, 0
    launch_count_f = len(durations) / active_count
    launch_count = int(round(launch_count_f))
    if abs(launch_count_f - launch_count) > 0.01:
        logger.warning(
            "kernel %s 的发射次数 %.3f 非整数（CSV 混入了非 active 行或 forward 内条件分支）",
            name, launch_count_f,
        )
    if launch_count <= 0:
        return 0.0, 0

    durations = np.asarray(durations, dtype=float)
    median_m = float(np.median(durations))
    thr = max(median_m * BAD_ROW_K, BAD_ROW_FLOOR_US)
    clean = durations[durations <= thr]
    removed = len(durations) - len(clean)
    if removed > 0:
        logger.warning(
            "kernel %s 剔除 %d 个异常采样（阈值 %.2fus，中位数 %.2fus）",
            name, removed, thr, median_m,
        )
    n = len(clean)
    n -= n % active_count  # 只保留末尾可整除段，避免剔除坏行后 reshape 错位
    if n >= active_count:
        iter_sums = clean[-n:].reshape(active_count, -1).sum(axis=1)
        avg_us = float(iter_sums.mean())
    else:
        avg_us = float(clean.sum()) / active_count
    return avg_us, launch_count
