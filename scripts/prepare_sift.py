"""Export reproducible SIFT samples as source-ID plus float32 text rows.

File format v1: first line is 'row_count dimension'; subsequent lines are
'source_id coordinate_0 ... coordinate_d_minus_1'. Input order is preserved.
No published neighbors are copied: exact top-10 is computed later per sample.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_rows(dataset, ids, chunk_rows=10_000):
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("Source IDs must be a one-dimensional integer list")
    if len(dataset.shape) != 2 or dataset.dtype.kind != "f" or dataset.dtype.itemsize != 4:
        raise ValueError("Source vectors must be a two-dimensional float32 array")
    if chunk_rows <= 0:
        raise ValueError("Chunk size must be positive")
    if np.any(ids < 0) or np.any(ids >= dataset.shape[0]) or len(np.unique(ids)) != len(ids):
        raise ValueError("Source IDs must be distinct and in range")
    ids = ids.astype(np.int64)
    values = np.empty((len(ids), dataset.shape[1]), dtype=np.float32)
    for start in range(0, dataset.shape[0], chunk_rows):
        stop = min(start + chunk_rows, dataset.shape[0])
        positions = np.flatnonzero((ids >= start) & (ids < stop))
        if len(positions):
            block = dataset[start:stop]
            values[positions] = block[ids[positions] - start]
    if not np.isfinite(values).all():
        raise ValueError("Selected coordinates must be finite")
    return values


def write_vectors(path, ids, values):
    path = Path(path)
    ids, values = np.asarray(ids), np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("Export IDs must be nonnegative integers")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Export IDs must be distinct")
    if values.ndim != 2 or values.dtype != np.dtype(np.float32) or len(values) != len(ids):
        raise ValueError("Export requires aligned ID and float32 vector arrays")
    if not np.isfinite(values).all():
        raise ValueError("Export coordinates must be finite")
    rows, dimension = values.shape
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{rows} {dimension}\n")
        for source_id, vector in zip(ids, values):
            coordinates = " ".join(format(float(value), ".9g") for value in vector)
            stream.write(f"{int(source_id)} {coordinates}\n")

    # Check the emitted file, including every ID and every float32 value.
    with temporary.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != f"{rows} {dimension}":
            raise ValueError("Export header mismatch")
        for source_id, vector in zip(ids, values):
            fields = stream.readline().split()
            if len(fields) != dimension + 1 or int(fields[0]) != int(source_id):
                raise ValueError("Export ID/vector alignment mismatch")
            if not np.array_equal(np.asarray(fields[1:], dtype=np.float32), vector):
                raise ValueError("Coordinates did not round-trip to the original float32 values")
        if stream.read().strip():
            raise ValueError("Export contains extra records")
    if path.exists():
        if file_hash(path) != file_hash(temporary):
            raise ValueError(f"Existing export differs; stop and inspect: {path}")
        temporary.unlink()
    else:
        temporary.rename(path)


def write_record(path, record):
    payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing metadata differs; stop and inspect: {path}")
    else:
        path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--query-seed", type=int, default=43)
    args = parser.parse_args()
    import h5py

    source_hash = file_hash(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as source:
        metric = source.attrs.get("distance")
        if isinstance(metric, bytes):
            metric = metric.decode()
        if metric != "euclidean" or source["train"].shape != (1_000_000, 128) or source["test"].shape != (10_000, 128):
            raise ValueError("Expected the SIFT-128 Euclidean base/query schema")
        base_ids = np.random.Generator(np.random.PCG64(args.base_seed)).permutation(len(source["train"]))[:50_000]
        query_ids = np.random.Generator(np.random.PCG64(args.query_seed)).permutation(len(source["test"]))[:100]
        selection = {
            "protocol_version": 1,
            "dataset": "sift-128-euclidean",
            "source_url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
            "source_sha256": source_hash,
            "dimension": 128,
            "coordinate_type": "float32",
            "rng": "PCG64",
            "base_seed": args.base_seed,
            "query_seed": args.query_seed,
            "base_50k_ids": base_ids.tolist(),
            "base_10k_rule": "first 10000 entries of base_50k_ids, in that order",
            "query_ids": query_ids.tolist(),
            "ground_truth": {"k": 10, "metric": "squared_l2", "accumulator": "double",
                             "order": ["distance_ascending", "source_id_ascending"],
                             "scope": "recompute separately within each sampled base"},
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "h5py": h5py.__version__},
        }
        selection_hash = write_record(args.output / "selection.json", selection)
        print("Selection SHA-256:", selection_hash, flush=True)
        base = selected_rows(source["train"], base_ids)
        queries = selected_rows(source["test"], query_ids)
        for name, ids, values in (("train", base_ids, base), ("test", query_ids, queries)):
            for position in (0, len(ids) // 2, len(ids) - 1):
                if not np.array_equal(values[position], source[name][int(ids[position])]):
                    raise ValueError("Direct source-row spot check failed")

    exports = (("base_10k.txt", base_ids[:10_000], base[:10_000]),
               ("base_50k.txt", base_ids, base),
               ("queries_100.txt", query_ids, queries))
    files = {}
    for name, ids, values in exports:
        path = args.output / name
        write_vectors(path, ids, values)
        digest = file_hash(path)
        files[name] = {"rows": len(ids), "dimension": 128, "bytes": path.stat().st_size, "sha256": digest}
        print(f"{name}: rows={len(ids)}, dimension=128, first_id={int(ids[0])}, sha256={digest}", flush=True)

    manifest = {
        "format": "source-id-float32-text-v1",
        "header": "row_count dimension",
        "row": "source_id followed by dimension coordinates",
        "coordinates": "9 significant decimal digits; every value verified after float32 parsing",
        "selection_sha256": selection_hash,
        "source_bytes": args.source.stat().st_size,
        "source_sha256": source_hash,
        "exporter_sha256": file_hash(Path(__file__)),
        "reproduction_command": [".venv/bin/python", "-B", *sys.argv],
        "files": files,
        "versions": selection["versions"],
        "ground_truth_rule": selection["ground_truth"],
    }
    manifest_hash = write_record(args.output / "manifest.json", manifest)
    print("Manifest SHA-256:", manifest_hash)
    print("PASS: finite vectors, source IDs, float32 round-trip, and nested base order")
    print("Matching existing outputs are retained; ground truth is not generated yet.")


if __name__ == "__main__":
    main()
