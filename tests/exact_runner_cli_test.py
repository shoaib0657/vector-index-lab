#!/usr/bin/env python3
"""Check the exact search command with small inputs and known answers."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EXECUTABLE = Path(sys.argv.pop(1)).resolve()


class ExactRunnerTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="exact runner test ")
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.base = self.root / "base.txt"
        self.queries = self.root / "queries.txt"
        self.base.write_text("5 2\n8 1 0\n2 0 1\n5 0 0\n99 3 0\n10 0 0\n")
        self.queries.write_text("2 2\n700 0 0\n701 1 0\n")

    def invoke(self, name="run", changes=None, extra=()):
        results = self.root / (name + "-results.csv")
        timings = self.root / (name + "-timings.csv")
        flags = {
            "--base": self.base, "--queries": self.queries,
            "--k": 3, "--results": results, "--timings": timings,
            "--warmup": 3, "--passes": 2,
        }
        if changes:
            flags.update(changes)
        arguments = [str(EXECUTABLE)]
        for flag, value in flags.items():
            if value is not None:
                arguments.extend([flag, str(value)])
        completed = subprocess.run(arguments + list(extra), capture_output=True, text=True)
        return completed, Path(flags["--results"]), Path(flags["--timings"])

    def require_success(self, completed, results, timings):
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(results.is_file(), "runner did not write exact answers")
        self.assertTrue(timings.is_file(), "runner did not write timing records")

    @staticmethod
    def read_csv(path):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def require_rejection(self, completed, results, timings):
        self.assertNotEqual(completed.returncode, 0, "invalid workload was accepted")
        self.assertTrue(completed.stderr.strip(), "failure needs an explanatory diagnostic")
        self.assertFalse(results.exists(), "invalid workload created a results file")
        self.assertFalse(timings.exists(), "invalid workload created a timings file")

    def test_help_describes_a_runnable_command(self):
        completed = subprocess.run([str(EXECUTABLE), "--help"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--base", completed.stdout)
        self.assertIn("--queries", completed.stdout)
        self.assertIn("--results", completed.stdout)

    def test_known_answers_query_association_and_repeatability(self):
        completed, results, timings = self.invoke("first")
        self.require_success(completed, results, timings)
        header, rows = self.read_csv(results)
        self.assertEqual(header, [
            "base_size", "dimension", "k", "query_id", "rank",
            "neighbor_id", "squared_distance",
        ])
        observed = [
            tuple(int(row[key]) for key in header[:-1]) + (float(row["squared_distance"]),)
            for row in rows
        ]
        # IDs 2 and 8 tie for query 700. ID 2 must win at the cutoff.
        self.assertEqual(observed, [
            (5, 2, 3, 700, 1, 5, 0.0),
            (5, 2, 3, 700, 2, 10, 0.0),
            (5, 2, 3, 700, 3, 2, 1.0),
            (5, 2, 3, 701, 1, 8, 0.0),
            (5, 2, 3, 701, 2, 5, 1.0),
            (5, 2, 3, 701, 3, 10, 1.0),
        ])
        timing_header, measurements = self.read_csv(timings)
        self.assertEqual(timing_header, ["pass", "query_id", "elapsed_us"])
        self.assertEqual(
            [(int(row["pass"]), int(row["query_id"])) for row in measurements],
            [(1, 700), (1, 701), (2, 700), (2, 701)],
        )
        for row in measurements:
            elapsed = float(row["elapsed_us"])
            self.assertTrue(math.isfinite(elapsed) and elapsed >= 0)
        repeated, repeated_results, repeated_timings = self.invoke("second")
        self.require_success(repeated, repeated_results, repeated_timings)
        self.assertEqual(results.read_bytes(), repeated_results.read_bytes())

    def test_uint64_ids_and_double_distance_survive_csv(self):
        self.base.write_text("1 2\n18446744073709551615 0.123456791 1.00000012\n")
        self.queries.write_text("1 2\n9007199254740993 0 0\n")
        completed, results, timings = self.invoke(changes={"--k": 1, "--warmup": 0, "--passes": 1})
        self.require_success(completed, results, timings)
        _, rows = self.read_csv(results)
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["query_id"]), 9007199254740993)
        self.assertEqual(int(rows[0]["neighbor_id"]), 18446744073709551615)
        # These constants give the exact float32 input values in double precision.
        a, b = 0.12345679104328156, 1.0000001192092896
        self.assertEqual(float(rows[0]["squared_distance"]), a * a + b * b)

    def test_default_protocol_returns_top_ten_and_three_passes(self):
        self.base.write_text(
            "10 1\n21 0\n22 1\n23 2\n24 3\n25 4\n26 5\n27 6\n28 7\n29 8\n30 9\n"
        )
        self.queries.write_text("2 1\n700 0\n701 9\n")
        completed, results, timings = self.invoke(
            changes={"--k": None, "--warmup": None, "--passes": None}
        )
        self.require_success(completed, results, timings)
        _, rows = self.read_csv(results)
        self.assertEqual(len(rows), 20)
        self.assertEqual([int(row["neighbor_id"]) for row in rows], [
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
        ])
        _, measurements = self.read_csv(timings)
        self.assertEqual(
            [(int(row["pass"]), int(row["query_id"])) for row in measurements],
            [(1, 700), (1, 701), (2, 700), (2, 701), (3, 700), (3, 701)],
        )
        self.assertIn("warmup=20", completed.stdout)

    def test_invalid_inputs_fail_before_outputs_are_created(self):
        original_base = self.base.read_text()
        original_queries = self.queries.read_text()
        cases = [
            ("dimension mismatch", original_base, "1 1\n700 0\n"),
            ("empty base", "0 2\n", original_queries),
            ("empty queries", original_base, "0 2\n"),
            ("duplicate base IDs", "2 2\n8 1 0\n8 0 1\n", original_queries),
            ("nonfinite query", original_base, "1 2\n700 nan 0\n"),
            ("truncated base", "1 2\n8 1\n", original_queries),
        ]
        for name, base, queries in cases:
            with self.subTest(name=name):
                self.base.write_text(base)
                self.queries.write_text(queries)
                self.require_rejection(*self.invoke())
        self.base.write_text(original_base)
        self.queries.write_text(original_queries)
        self.require_rejection(*self.invoke(changes={"--base": self.root / "absent.txt"}))

    def test_invalid_arguments_fail_before_outputs_are_created(self):
        cases = [
            ("missing required path", {"--queries": None}, ()),
            ("zero k", {"--k": 0}, ()),
            ("k exceeds base", {"--k": 6}, ()),
            ("negative k", {"--k": -1}, ()),
            ("fractional k", {"--k": "1.5"}, ()),
            ("partial numeric token", {"--passes": "2x"}, ()),
            ("numeric overflow", {"--warmup": "18446744073709551616"}, ()),
            ("zero passes", {"--passes": 0}, ()),
            ("unknown flag", {}, ("--mystery", "1")),
            ("duplicate flag", {}, ("--k", "2")),
            ("missing value", {}, ("--passes",)),
        ]
        for name, changes, extra in cases:
            with self.subTest(name=name):
                self.require_rejection(*self.invoke(changes=changes, extra=extra))

    def test_existing_outputs_are_preserved_and_rejected(self):
        for flag in ("--results", "--timings"):
            with self.subTest(flag=flag):
                existing = self.root / ("existing" + flag + ".csv")
                existing.write_bytes(b"previous evidence\n")
                completed, results, timings = self.invoke(changes={flag: existing})
                self.assertNotEqual(completed.returncode, 0, "existing evidence was overwritten")
                self.assertTrue(completed.stderr.strip())
                self.assertEqual(existing.read_bytes(), b"previous evidence\n")
                other = timings if flag == "--results" else results
                self.assertFalse(other.exists())

    def test_aliased_outputs_and_missing_parent_are_rejected(self):
        alias = self.root / "same.csv"
        self.require_rejection(*self.invoke(changes={"--results": alias, "--timings": alias}))
        alias_directory = self.root / "alias-directory"
        alias_directory.mkdir()
        nested_alias = alias_directory / ".." / "same.csv"
        self.require_rejection(*self.invoke(changes={"--results": alias, "--timings": nested_alias}))
        self.require_rejection(*self.invoke(changes={
            "--timings": self.root / "missing-parent" / "timings.csv",
        }))


if __name__ == "__main__":
    unittest.main()
