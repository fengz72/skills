import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "request_metrics.py"
SPEC = importlib.util.spec_from_file_location("request_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RequestMetricsTests(unittest.TestCase):
    def rows(self, count=100):
        return [dict(request_id=str(i), phase="measure", status="success",
                     scheduled_ns=i * 10_000_000, sent_ns=i * 10_000_000,
                     completed_ns=i * 10_000_000 + 1_000_000)
                for i in range(count)]

    def run_summary(self, rows, **overrides):
        contract = dict(window_start_ns=0, window_end_ns=1_000_000_000,
                        expected_requests=100, min_requests=100,
                        p99_limit_ms=2, max_send_lag_ms=1)
        contract.update(overrides)
        return MODULE.summarize(rows, **contract)

    def test_valid_cohort_has_real_quantile_and_window_rate(self):
        result = self.run_summary(self.rows())
        self.assertTrue(result["numeric_constraints_met"])
        self.assertEqual(result["p99_ms"], 1)
        self.assertEqual(result["successful_completions_in_window_per_s"], 100)

    def test_missing_arrival_and_duplicate_ids_are_invalid(self):
        self.assertFalse(self.run_summary(self.rows()[:-1])["measurement_valid"])
        rows = self.rows()
        rows[-1]["request_id"] = rows[0]["request_id"]
        self.assertFalse(self.run_summary(rows)["measurement_valid"])

    def test_pending_tail_never_reports_fast_subset_p99(self):
        rows = self.rows()
        rows[-1]["status"] = "pending"
        rows[-1].pop("completed_ns")
        result = self.run_summary(rows)
        self.assertFalse(result["measurement_valid"])
        self.assertIsNone(result["p99_ms"])

    def test_finalized_timeouts_are_not_dropped(self):
        rows = self.rows()
        for row in rows[-2:]:
            row["status"] = "timeout"
            row["completed_ns"] = row["scheduled_ns"] + 100_000_000
        result = self.run_summary(rows)
        self.assertTrue(result["measurement_valid"])
        self.assertFalse(result["numeric_constraints_met"])
        self.assertEqual(result["p99_ms"], 100)
        self.assertEqual(result["error_rate"], .02)
        self.assertEqual(result["successful_completions_in_window_per_s"], 98)

    def test_sender_stall_invalid_even_when_sent_to_done_is_fast(self):
        rows = self.rows()
        for row in rows:
            row["sent_ns"] += 20_000_000
            row["completed_ns"] += 20_000_000
        result = self.run_summary(rows, latency_start="sent_ns")
        self.assertFalse(result["measurement_valid"])
        self.assertIsNone(result["p99_ms"])

    def test_work_after_window_is_latency_data_but_not_window_completion(self):
        rows = self.rows()
        rows[-1]["completed_ns"] = 1_010_000_000
        result = self.run_summary(rows)
        self.assertTrue(result["measurement_valid"])
        self.assertEqual(result["counts"]["outstanding_at_window_end"], 1)
        self.assertEqual(result["successful_completions_in_window_per_s"], 99)
        self.assertEqual(result["cohort_successes_per_window_s"], 100)

    def test_insufficient_smoke_data_and_bad_timestamps_invalid(self):
        self.assertFalse(self.run_summary(self.rows(5), expected_requests=5)["measurement_valid"])
        rows = self.rows()
        rows[0]["completed_ns"] = -1
        self.assertFalse(self.run_summary(rows)["measurement_valid"])

    def test_contract_rejects_nan_and_zero_duration(self):
        with self.assertRaises(ValueError):
            self.run_summary(self.rows(), p99_limit_ms=float("nan"))
        with self.assertRaises(ValueError):
            self.run_summary(self.rows(), window_end_ns=0)


if __name__ == "__main__":
    unittest.main()
