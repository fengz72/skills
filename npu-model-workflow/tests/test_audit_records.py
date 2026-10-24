import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_records.py"
SPEC = importlib.util.spec_from_file_location("audit_records", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecordCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.experiments = self.root / "experiments"
        self.experiments.mkdir()
        self.progress = self.root / "progress.md"
        self.report = self.root / "acceptance.md"
        self.progress.write_text("| OPT-001 | accepted |\n| OPT-002 | failed |\n")
        for opt in ("OPT-001", "OPT-002"):
            (self.experiments / (opt + ".md")).write_text(
                "# " + opt + "：attempt\n" + "".join("## %d. section\nresult\n" % i for i in range(1, 8)))
        self.rows = ["| [OPT-001](experiments/OPT-001.md) | reuse | base:100 | new:110 | +10% | 是 | passed |",
                     "| [OPT-002](experiments/OPT-002.md) | fusion | base:110 | 未测得 | 不适用 | 否 | compile failed |"]
        self.report.write_text("\n".join(self.rows))

    def audit(self):
        return MODULE.audit(self.progress, self.experiments, self.report)

    def test_failures_and_successes_are_both_retained(self):
        result = self.audit()
        self.assertTrue(result["record_coverage_complete"], result)
        self.assertEqual(result["allocated"], ["OPT-001", "OPT-002"])

    def test_same_row_count_cannot_hide_missing_failed_attempt(self):
        self.report.write_text("\n".join([self.rows[0], self.rows[0]]))
        result = self.audit()
        self.assertFalse(result["record_coverage_complete"])
        self.assertIn("report rows missing OPT-002", result["issues"])

    def test_record_cannot_link_to_another_optimization(self):
        self.report.write_text("\n".join(self.rows).replace("(experiments/OPT-002.md)", "(experiments/OPT-001.md)"))
        self.assertFalse(self.audit()["record_coverage_complete"])

    def test_deleted_record_or_unallocated_record_is_detected(self):
        (self.experiments / "OPT-002.md").unlink()
        self.assertFalse(self.audit()["record_coverage_complete"])

    def test_zero_attempts_requires_explicit_declaration(self):
        for path in self.experiments.iterdir():
            path.unlink()
        self.progress.write_text("# Progress\n")
        self.report.write_text("# Report\nNo optimization attempts.\n")
        self.assertFalse(self.audit()["record_coverage_complete"])
        self.progress.write_text("# Progress\n优化尝试数：0\n")
        self.assertTrue(self.audit()["record_coverage_complete"])

    def test_incomplete_summary_and_missing_record_section_are_detected(self):
        self.report.write_text("\n".join(self.rows).replace("compile failed", ""))
        self.assertFalse(self.audit()["record_coverage_complete"])
        self.report.write_text("\n".join(self.rows))
        (self.experiments / "OPT-002.md").write_text("# OPT-002：failed\n## 1. status\nfailed\n")
        self.assertFalse(self.audit()["record_coverage_complete"])


if __name__ == "__main__":
    unittest.main()
