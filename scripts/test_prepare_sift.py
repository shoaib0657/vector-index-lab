"""Small correctness fixtures for the public SIFT exporter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_sift import selected_rows, write_vectors


class ExportTests(unittest.TestCase):
    def test_source_ids_keep_their_values_and_selection_order_across_chunks(self):
        source = np.array([[10, 11], [20, 21], [30, 31], [40, 41], [50, 51], [60, 61]], dtype=np.float32)
        actual = selected_rows(source, [5, 0, 3, 1], chunk_rows=2)
        np.testing.assert_array_equal(actual, [[60, 61], [10, 11], [40, 41], [20, 21]])

    def test_duplicate_and_invalid_source_ids_are_rejected(self):
        source = np.zeros((3, 2), dtype=np.float32)
        for ids in ([1, 1], [-1], [3], [0.5]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                selected_rows(source, ids)

    def test_selected_nonfinite_values_are_rejected(self):
        source = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)
        with self.assertRaises(ValueError):
            selected_rows(source, [1])

    def test_unselected_bad_rows_do_not_change_valid_selected_rows(self):
        source = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)
        np.testing.assert_array_equal(selected_rows(source, [0]), [[1, 2]])

    def test_text_export_keeps_integer_ids_order_and_float32_precision(self):
        ids = np.array([2**63 + 7, 2], dtype=np.uint64)
        values = np.array([[0.12345679, np.nextafter(np.float32(1), np.float32(2))], [-2, 3]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.txt"
            write_vectors(path, ids, values)
            self.assertTrue(path.is_file(), "the writer must produce a vector file")
            lines = path.read_text().splitlines()
            self.assertEqual(lines[0], "2 2")
            self.assertEqual([int(line.split()[0]) for line in lines[1:]], [2**63 + 7, 2])
            parsed = np.array([line.split()[1:] for line in lines[1:]], dtype=np.float32)
            np.testing.assert_array_equal(parsed, values)
            original = path.read_bytes()
            write_vectors(path, ids, values)
            self.assertEqual(path.read_bytes(), original)

    def test_conflicting_existing_export_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.txt"
            path.write_bytes(b"existing experiment\n")
            with self.assertRaises(ValueError):
                write_vectors(path, np.array([8], dtype=np.uint64), np.array([[1, 2]], dtype=np.float32))
            self.assertEqual(path.read_bytes(), b"existing experiment\n")


if __name__ == "__main__":
    unittest.main()
