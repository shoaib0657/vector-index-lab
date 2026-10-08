# Exact SIFT-128 baseline

[The CSV](exact-sift-128.csv) contains four measured rows from 2026-10-07.
[The metadata](exact-sift-128.json) identifies the inputs, source, executable,
machine, build, commands, and independent answer checks.
The report was prepared on 2026-10-08 from the saved experiment.

## Workload and input identity

The source is the public [ANN Benchmarks SIFT-128 HDF5 file](https://ann-benchmarks.com/sift-128-euclidean.hdf5):
1,000,000 train vectors and 10,000 test vectors, each with 128 float32 coordinates.
The exporter uses `Generator(PCG64(42)).permutation(1000000)[:50000]` for the
50k base. The 10k base is its first 10,000 records, in the same order.
Queries use a separate `Generator(PCG64(43)).permutation(10000)[:100]` from test.
All coordinates retain their values. No normalization is applied.

Base IDs are original train row numbers. Query IDs are original test row
numbers in a separate namespace. Exact top-10 is recomputed within each
sample. Published million-row neighbors are not used as sampled ground truth.

The metadata records the HDF5, selection, manifest, exporter, and input SHA-256
values. Python 3.12.3, NumPy 2.5.3, and h5py 3.16.0 produced the original exports.
Full saved IDs live in the ignored `data/sift-samples-v1/selection.json`.
Recreate the samples with the [README commands](../README.md#prepare-the-sift-samples),
then compare the input-file hashes with the metadata. The manifest also
contains invocation and version details; its hash can differ when those details
change even if the vector-file hashes match.

## Search and timing

The baseline scores all base vectors with double squared L2 and fully sorts
them by distance, then integer source ID. Its cost is `O(Nd + N log N)`
time and `O(N)` temporary space. The baseline retains finite-coordinate
validation inside every search call.

Each process performs 20 warm-up searches, then three timed passes over the
100 queries in saved order. Warm-up uses the first 20 queries. Each row
contains 300 calls. Four processes run sequentially in this order:

1. Run 1, 10k base.
2. Run 1, 50k base.
3. Run 2, 10k base.
4. Run 2, 50k base.

`std::chrono::steady_clock` measures one complete `exact_top_k` call.
Coordinate validation, distance calculations, sorting, and result allocation
are inside the interval. Loading, warm-up, result comparison, CSV writing,
and result destruction after the timed call are outside it.

Summary fields use these definitions:

| Field | Definition |
| --- | --- |
| `median_ms` | Mean of the 150th and 151st sorted call times, in milliseconds. |
| `p95_ms` | Nearest rank: the 285th of 300 sorted call times, in milliseconds. |
| `sum_search_seconds` | Sum of all 300 measured call intervals, in seconds. |
| `qps_from_sum_search_times` | `300 / sum_search_seconds`, a derived serial search rate. |
| `min_ms`, `max_ms` | Smallest and largest recorded call times. |

For example, 10k run 1 has `300 / 1.91856037 = 156.3672453` queries/s.
The rate uses all call times. It does not use the reciprocal of the median.
Table values in the main README are rounded; the CSV retains the saved values.
The runs remain separate so their variation stays visible.

## Correctness and measured source

The measurement's Release build passed all five CTest tests.
Both bases produced identical answer bytes between their two processes.
An independent NumPy check matched all 100 queries and all 1,000 ordered
neighbor ID/distance pairs for each base. For each query, it promoted the
exported float32 values to float64 and calculated:

```python
difference = base_values.astype(np.float64) - query.astype(np.float64)
distances = np.sum(difference * difference, axis=1, dtype=np.float64)
order = np.lexsort((base_ids, distances))[:10]
```

This calculation uses neither the C++ search function nor the source's
published neighbors. The check also confirmed unique IDs, exact query/rank
association, the nested base order and values, and timing pass/query order.
The sampled coordinates were all integer-valued. Conservative squared-distance
bounds were 17,713,152 for 10k and 20,480,000 for 50k, below `2**53`.
This allowed exact distance comparison across the two reduction methods.

Query ID 5061 illustrates why the reference depends on the base:

| Base | Nearest train row ID | Squared L2 |
| --- | ---: | ---: |
| 10k | 987394 | 86614 |
| 50k | 425185 | 76099 |

The measured checkout was dirty `main` at
`9fa80b0ff1309af536d76cdf1cf69e6e8e5f484c`. The new loader and runner were
uncommitted. Use the recorded source-file hashes to identify the implementation;
the parent revision alone is insufficient. The aggregate source identity is
SHA-256 of the complete filename-to-hash map serialized as sorted compact JSON.
It includes the README as it stood during measurement, before this report was
added. Report-only changes do not change the measured C++ implementation.

The saved experiment is local under the ignored
`data/exact-baseline/sift-v1/` directory. It contains raw answer/timing CSVs,
build and test logs, the source snapshot, commands, and the independent checker.
Forty files were checked against its evidence hash record.
Those local raw files are not included in this public report; their output
hashes and statistics are retained here. A fresh checkout can recreate the
inputs and run the baseline with the commands below.

## Repeat the workload

First prepare the samples, then build and run correctness checks as shown in
the main README. Choose a new output directory for every experiment:

```sh
set -eu
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel 2
ctest --test-dir build/release --output-on-failure
experiment=data/exact-baseline/repeat-1
mkdir -p "$experiment"
for run in 1 2; do
    mkdir "$experiment/run-$run"
    for size in 10k 50k; do
        ./build/release/exact_baseline \
            --base "data/sift-samples-v1/base_$size.txt" \
            --queries data/sift-samples-v1/queries_100.txt \
            --k 10 --warmup 20 --passes 3 \
            --results "$experiment/run-$run/top10-$size.csv" \
            --timings "$experiment/run-$run/timings-$size.csv"
    done
done
cmp "$experiment/run-1/top10-10k.csv" "$experiment/run-2/top10-10k.csv"
cmp "$experiment/run-1/top10-50k.csv" "$experiment/run-2/top10-50k.csv"
sha256sum data/sift-samples-v1/*.txt "$experiment"/run-*/top10-*.csv
```

Each answer file should contain 1,000 data rows; each timing file should
contain 300. On the recorded inputs, answer hashes should match the metadata.
New elapsed times will vary. Save the new machine/compiler versions, flags,
source revision and dirty state, input/output hashes, commands, and raw times
with any new measurements.

The observed times describe this full-sort baseline on the recorded machine.
They do not predict another machine's timings. The runner uses one search
thread; the host exposed eight logical CPUs, and CPU affinity was not pinned.
