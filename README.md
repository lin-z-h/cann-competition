# BatchMatmulMaxSum Ascend C solution

The submission file is `kernel.asc`.  It implements FP16/BF16 matrix
multiplication with FP32 accumulation, row-wise MaxSim, and an ordered M-axis
sum.  Work is partitioned by `(batch, M tile)` and no atomics are used.

Local CPU semantic checks:

```bash
python tests/reference_test.py
```

NPU build/run (CANN 9.0.0 environment):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p build && cd build
cmake .. && make -j4
```

Only `kernel.asc` is editable in the online judge template.  Copy that file's
contents into the answer editor for submission.
