#!/bin/bash

# ── 配置 ──
BASEDIR="."
OUTPUT="$BASEDIR/gds_results_sdb1.txt"
TEST_PATH="/docker/gds_tmp"
GDSIO="/usr/local/cuda-13/gds/tools/gdsio"
DURATION=120
COOLDOWN=30

# 创建测试目录（如果不存在）
mkdir -p $TEST_PATH

run_test() {
    local label="$1"
    local args="$2"
    echo "==============================" >> $OUTPUT
    echo "TEST: $label" >> $OUTPUT
    echo "CMD: $GDSIO $args" >> $OUTPUT
    date >> $OUTPUT
    $GDSIO $args >> $OUTPUT 2>&1
    date >> $OUTPUT
    echo "" >> $OUTPUT
    echo "Cooling down ${COOLDOWN}s..." >> $OUTPUT
    sync
    sleep $COOLDOWN
}

# ── 输出头 ──
echo "GDS Benchmark Results on sdb1 (/docker)" > $OUTPUT
echo "Run: $(date)" >> $OUTPUT
echo "" >> $OUTPUT

# ── 循环不同 block size ──
for BS in 256K 512K 1M 2M 4M 8M 16M; do
    # Non-GDS 写入
    run_test "Non-GDS Write | $BS" \
        "-D $TEST_PATH -d 0 -w 8 -s 500M -i $BS -x 1 -I 1 -T $DURATION"
    # Non-GDS 读取
    run_test "Non-GDS Read  | $BS" \
        "-D $TEST_PATH -d 0 -w 8 -s 500M -i $BS -x 1 -I 0 -T $DURATION"
    # GDS 写入
    run_test "GDS Write     | $BS" \
        "-D $TEST_PATH -d 0 -w 8 -s 500M -i $BS -x 0 -I 1 -T $DURATION"
    # GDS 读取
    run_test "GDS Read      | $BS" \
        "-D $TEST_PATH -d 0 -w 8 -s 500M -i $BS -x 0 -I 0 -T $DURATION"
done

echo "==============================" >> $OUTPUT
echo "All done: $(date)" >> $OUTPUT