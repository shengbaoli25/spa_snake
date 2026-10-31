"""Asynchronous CSV logger tests; no robot or network access."""
import csv
from pathlib import Path
import tempfile
import time
import unittest

from spa_control.trial_log import AsyncCsvWriter


class AsyncCsvWriterTests(unittest.TestCase):
    def test_fifo_rows_and_missing_values_are_preserved_before_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.csv"
            logger = AsyncCsvWriter(path, ["index", "current", "text"], capacity=16)
            logger.start()
            for index in range(12):
                row = {"index": index, "current": None, "text": 'a,"b"'}
                logger.submit(row)
                row["index"] = 999  # Caller mutation cannot alter a queued row.
            logger.finish(timeout_s=1.0)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["index"] for row in rows], [str(i) for i in range(12)])
            self.assertTrue(all(row["current"] == "" and row["text"] == 'a,"b"' for row in rows))
            self.assertEqual(logger.diagnostics(), {"submitted_rows": 12, "written_rows": 12,
                                                    "finished": True, "error": ""})
            with self.assertRaises(RuntimeError):
                logger.submit({"index": 13})

    def test_open_failure_is_reported_before_motion_can_start(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncCsvWriter(Path(directory) / "absent" / "steps.csv", ["index"])
            with self.assertRaisesRegex(RuntimeError, "CSV 日志失败"):
                logger.start(timeout_s=0.5)
            before = time.monotonic()
            with self.assertRaises(RuntimeError):
                logger.finish(timeout_s=0.2)
            self.assertLess(time.monotonic() - before, 0.2)


if __name__ == "__main__":
    unittest.main()
