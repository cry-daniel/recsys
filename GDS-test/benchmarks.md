## The Figure Set (6 figures total)


### IB bandwidth (already done ✓)



### fio baseline — bar chart per storage, read vs write

GPFS — Sequential Write
```bash
fio --name=gpfs_seqwrite --ioengine=libaio --iodepth=32 \
    --rw=write --bs=4m --direct=1 --size=1G \
    --filename=/jobtmp/psaluja/gds_tmp/fio_test \
    --numjobs=32 --runtime=30 --time_based \
    --output=gpfs_write.json --output-format=json
```

GPFS — Sequential Read
```
fio --name=gpfs_seqread --ioengine=libaio --iodepth=32 \
    --rw=read --bs=4m --direct=1 --size=1G \
    --filename=/jobtmp/psaluja/gds_tmp/fio_test \
    --numjobs=32 --runtime=30 --time_based \
    --output=gpfs_read.json --output-format=json
```

VAST — Sequential Write

```
fio --name=vast_seqwrite --ioengine=libaio --iodepth=32 \
    --rw=write --bs=4m --direct=1 --size=1G \
    --filename=/oscar/scratch/psaluja/gds_tmp/fio_test \
    --numjobs=32 --runtime=30 --time_based \
    --output=vast_write.json --output-format=json
```

VAST — Sequential Read
```
fio --name=vast_seqread --ioengine=libaio --iodepth=32 \
    --rw=read --bs=4m --direct=1 --size=1G \
    --filename=/oscar/scratch/psaluja/gds_tmp/fio_test \
    --numjobs=32 --runtime=30 --time_based \
    --output=vast_read.json --output-format=json
```



## GDS vs non-GDS bandwidth — grouped bars by storage system and transfer size

```bash
#!/bin/bash

BASEDIR="/users/psaluja/6_PEARC"
OUTPUT="$BASEDIR/gds_results_v2.txt"
GPFS_PATH="/jobtmp/psaluja/gds_tmp"
VAST_PATH="/oscar/scratch/psaluja/gds_tmp"
GDSIO="/usr/local/cuda-13.1/gds/tools/gdsio"
DURATION=120
COOLDOWN=30

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

echo "GDS Benchmark Results v2" > $OUTPUT
echo "Run: $(date)" >> $OUTPUT
echo "" >> $OUTPUT

for BS in 256K 512K 1M 2M 4M 8M 16M; do

    # ── GPFS ──
    run_test "GPFS | Non-GDS Write | $BS" \
        "-D $GPFS_PATH -d 0 -w 8 -s 500M -i $BS -x 1 -I 1 -T $DURATION"
    run_test "GPFS | Non-GDS Read  | $BS" \
        "-D $GPFS_PATH -d 0 -w 8 -s 500M -i $BS -x 1 -I 0 -T $DURATION"
    run_test "GPFS | GDS Write     | $BS" \
        "-D $GPFS_PATH -d 0 -w 8 -s 500M -i $BS -x 0 -I 1 -T $DURATION"
    run_test "GPFS | GDS Read      | $BS" \
        "-D $GPFS_PATH -d 0 -w 8 -s 500M -i $BS -x 0 -I 0 -T $DURATION"

    # ── VAST ──
    run_test "VAST | Non-GDS Write | $BS" \
        "-D $VAST_PATH -d 0 -w 8 -s 500M -i $BS -x 1 -I 1 -T $DURATION"
    run_test "VAST | Non-GDS Read  | $BS" \
        "-D $VAST_PATH -d 0 -w 8 -s 500M -i $BS -x 1 -I 0 -T $DURATION"
    run_test "VAST | GDS Write     | $BS" \
        "-D $VAST_PATH -d 0 -w 8 -s 500M -i $BS -x 0 -I 1 -T $DURATION"
    run_test "VAST | GDS Read      | $BS" \
        "-D $VAST_PATH -d 0 -w 8 -s 500M -i $BS -x 0 -I 0 -T $DURATION"

done

echo "==============================" >> $OUTPUT
echo "All done: $(date)" >> $OUTPUT
```

The benchmark script executes 16 tests in total, covering sequential read and write operations across two storage backends (GPFS and VAST) at two block sizes (1M and 4M), in both GDS and non-GDS (CPU) modes. Each test runs for 120 seconds using 8 threads with a 500M per-thread working set via \texttt{gdsio}. A 30-second cooldown with \texttt{sync} is inserted between tests to allow the OS page cache and write buffers to settle and to avoid carry-over effects between runs. Results are timestamped and appended to a single output file for reproducibility. The script was run under \texttt{nohup} to ensure completion independent of session state.