# CANNBot provenance

`cannbot_aggregate.py` retains the exact `_aggregate_kernel_rows` function from
the orchestration workspace's `demo/benchmarks/cannbot/benchmark.py`, SHA-256
`cf33f091aa645731cf5a47ade64434b3c60b55ae75591c9226e1e769fed5707a`.
The original source header names Huawei Technologies Co., Ltd. (2026) and the
CANN Open Software License Agreement Version 2.0. The accompanying license
text is retained as `LICENSE.CANNBot`.

The executable adapter uses standard-library operations for the same policy;
the differential test executes the retained function with NumPy and checks
multiple launch counts, repeat counts, outliers, and incomplete sample groups.
No frequency-locking or cache-clearing utilities are copied or invoked.
