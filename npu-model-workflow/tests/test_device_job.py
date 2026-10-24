"""CPU-only process and flock regressions; unittest discover compatible."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "device_job.py"
SPEC = importlib.util.spec_from_file_location("device_job", SCRIPT)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


@unittest.skipUnless(os.name == "posix", "POSIX process groups and flock required")
class DeviceJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="npu-device-job-test-")
        self.root = Path(self.temporary.name)
        self.jobs = []

    def tearDown(self):
        # Scope cleanup to jobs created by this test, through their cancel files.
        for caller, run_dir in self.jobs:
            if (run_dir / "request.json").exists():
                HELPER.write_json(run_dir / "cancel.json", {
                    "task_id": HELPER.read_json(run_dir / "request.json")["task_id"]})
            if caller.poll() is None:
                caller.wait(timeout=8)
            caller.communicate(timeout=1)
        for _, run_dir in self.jobs:
            if (run_dir / "status.json").exists():
                self.wait_state(run_dir, "finished", timeout=8)
        self.temporary.cleanup()

    def start(self, name, code, *, timeout=5, devices=("host/physical-0",), grace=0.15,
              ignore_sigchld=False):
        run_dir = self.root / name
        argv = [sys.executable, str(SCRIPT), "run", "--cwd", str(self.root),
                "--run-dir", str(run_dir), "--lock-root", str(self.root / "locks"),
                "--timeout", str(timeout), "--grace", str(grace)]
        for device in devices:
            argv += ["--device", device]
        argv += ["--", sys.executable, "-c", code]
        if ignore_sigchld:
            launcher = ("import os,signal,sys; "
                        "signal.signal(signal.SIGCHLD,signal.SIG_IGN); "
                        "os.execv(sys.executable,[sys.executable]+sys.argv[1:])")
            argv = [sys.executable, "-c", launcher] + argv[1:]
        caller = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.jobs.append((caller, run_dir))
        return caller, run_dir

    def wait_state(self, run_dir, wanted, *, timeout=5):
        deadline = time.monotonic() + timeout
        observed = None
        while time.monotonic() < deadline:
            try:
                observed = HELPER.read_json(run_dir / "status.json")
            except FileNotFoundError:
                pass
            if observed and observed["state"] == wanted:
                return observed
            time.sleep(0.02)
        self.fail("{} did not reach {}: {}".format(run_dir, wanted, observed))

    def finish(self, caller, run_dir, expected):
        stdout, stderr = caller.communicate(timeout=8)
        self.assertEqual(caller.returncode, expected, (stdout, stderr))
        state = self.wait_state(run_dir, "finished")
        self.assertEqual(state["exit_code"], expected)
        self.assertTrue(state["process_group_stopped"])
        self.assertFalse(state["device_release_checked"])
        return state

    def test_normal_and_failure_preserve_evidence(self):
        caller, run_dir = self.start("success", "import os; print(os.getcwd()); print('job-output')")
        state = self.finish(caller, run_dir, 0)
        self.assertEqual(state["reason"], "exited")
        self.assertEqual(state["cwd"], str(self.root.resolve()))
        self.assertIn("job-output", (run_dir / "task.log").read_text())
        self.assertNotIn("environ", state)
        failed, failed_dir = self.start("failure", "import sys; print('failure'); sys.exit(7)")
        self.assertEqual(self.finish(failed, failed_dir, 7)["task_returncode"], 7)

    def test_inherited_sigchld_ignore_preserves_real_failure(self):
        caller, run_dir = self.start("sigchld-ignored", "import sys; sys.exit(7)",
                                     ignore_sigchld=True)
        state = self.finish(caller, run_dir, 7)
        self.assertEqual(state["task_returncode"], 7)
        self.assertEqual(state["reason"], "exited")

    def test_observer_failure_does_not_delay_kill_or_allow_early_release(self):
        clock, signals, waited = [0.0], [], []

        def sleep(seconds):
            clock[0] += seconds

        def observe(_pgid):
            if clock[0] < .3:
                raise OSError("observer temporarily unavailable")
            return []

        def send(_proc, number):
            signals.append((number, clock[0]))

        proc = mock.Mock(pid=123456)
        proc.wait.side_effect = lambda: (waited.append(clock[0]) or -signal.SIGKILL)
        with mock.patch.object(HELPER.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(HELPER.time, "sleep", side_effect=sleep), \
                mock.patch.object(HELPER, "live_group_members", side_effect=observe), \
                mock.patch.object(HELPER, "send_group", side_effect=send):
            result = HELPER.stop_group(proc, .1, mock.Mock())
        self.assertEqual(result, -signal.SIGKILL)
        self.assertEqual(signals[0][0], signal.SIGTERM)
        self.assertEqual(signals[1][0], signal.SIGKILL)
        self.assertLessEqual(signals[1][1], .15)
        self.assertGreaterEqual(waited[0], .3)

    def test_same_physical_device_is_exclusive(self):
        first, first_dir = self.start("first", "import time; time.sleep(0.55)")
        self.wait_state(first_dir, "running")
        second, second_dir = self.start("second", "from pathlib import Path; Path('second-started').touch()")
        self.wait_state(second_dir, "waiting_for_locks")
        time.sleep(0.15)
        self.assertFalse((self.root / "second-started").exists())
        self.finish(first, first_dir, 0)
        self.finish(second, second_dir, 0)
        self.assertTrue((self.root / "second-started").exists())

    def test_timeout_kills_term_ignoring_process_and_grandchild(self):
        grandchild = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)"
        code = ("import subprocess,sys,time,signal; from pathlib import Path; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "p=subprocess.Popen([sys.executable,'-c'," + repr(grandchild) + "]); "
                "Path('grandchild-pid').write_text(str(p.pid)); time.sleep(20)")
        caller, run_dir = self.start("timeout", code, timeout=0.7)
        state = self.wait_state(run_dir, "running")
        final = self.finish(caller, run_dir, 124)
        self.assertEqual(final["reason"], "timeout")
        self.assertEqual(HELPER.live_group_members(state["pgid"]), [])
        self.assertTrue((self.root / "grandchild-pid").exists())

    def test_overlapping_device_sets_wait_but_independent_devices_run(self):
        first, first_dir = self.start("two-devices", "import time; time.sleep(0.8)",
                                      devices=("physical-B", "physical-A"))
        self.wait_state(first_dir, "running")
        second, second_dir = self.start("overlap", "from pathlib import Path; Path('overlap-started').touch()",
                                        devices=("physical-C", "physical-B"))
        self.wait_state(second_dir, "waiting_for_locks")
        independent, independent_dir = self.start("independent", "from pathlib import Path; Path('independent-started').touch()",
                                                  devices=("physical-D",))
        self.finish(independent, independent_dir, 0)
        self.assertTrue((self.root / "independent-started").exists())
        self.assertFalse((self.root / "overlap-started").exists())
        self.finish(first, first_dir, 0)
        self.finish(second, second_dir, 0)

    def test_term_and_hup_cancel_before_unlock(self):
        for number in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=number):
                caller, run_dir = self.start("signal-{}".format(number), "import time; time.sleep(20)")
                state = self.wait_state(run_dir, "running")
                caller.send_signal(number)
                final = self.finish(caller, run_dir, 128 + number)
                self.assertEqual(final["reason"], "signal_{}".format(number))
                self.assertEqual(HELPER.live_group_members(state["pgid"]), [])

    def test_hard_killed_caller_does_not_drop_supervisor_lock(self):
        caller, run_dir = self.start("killed-caller", "import time; time.sleep(20)", timeout=1.1)
        state = self.wait_state(run_dir, "running")
        caller.kill()
        caller.wait(timeout=2)
        second, second_dir = self.start("after-killed-caller", "from pathlib import Path; Path('after-hard-kill').touch()")
        self.wait_state(second_dir, "waiting_for_locks")
        time.sleep(0.15)
        self.assertFalse((self.root / "after-hard-kill").exists())
        final = self.wait_state(run_dir, "finished")
        self.assertEqual(final["reason"], "timeout")
        self.assertEqual(HELPER.live_group_members(state["pgid"]), [])
        self.finish(second, second_dir, 0)

    def test_hard_killed_supervisor_leaves_lock_in_running_direct_child(self):
        # The direct child's inherited flock is an extra safeguard, not a claim
        # that arbitrary daemonized descendants survive wrapper failure safely.
        caller, run_dir = self.start("killed-supervisor", "import time; time.sleep(0.9)")
        state = self.wait_state(run_dir, "running")
        os.kill(state["supervisor"]["pid"], signal.SIGKILL)
        stdout, stderr = caller.communicate(timeout=3)
        self.assertEqual(caller.returncode, 70, (stdout, stderr))
        # Mark this deliberately unconfirmed run separately for tearDown.
        self.jobs.remove((caller, run_dir))
        second, second_dir = self.start("after-killed-supervisor", "from pathlib import Path; Path('after-supervisor-kill').touch()")
        self.wait_state(second_dir, "waiting_for_locks")
        time.sleep(0.1)
        self.assertFalse((self.root / "after-supervisor-kill").exists())
        self.finish(second, second_dir, 0)
        self.assertEqual(HELPER.live_group_members(state["pgid"]), [])
        self.assertEqual(HELPER.read_json(run_dir / "status.json")["state"], "running")

    def test_cancel_request_never_signals_pid_from_metadata(self):
        caller, run_dir = self.start("cancel", "import time; time.sleep(20)")
        self.wait_state(run_dir, "running")
        # Corrupt only the PID evidence; cancellation is via the task's run folder.
        stored = HELPER.read_json(run_dir / "status.json")
        stored["supervisor"]["pid"] = os.getpid()
        HELPER.write_json(run_dir / "status.json", stored)
        result = subprocess.run([sys.executable, str(SCRIPT), "cancel", "--run-dir", str(run_dir)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["cancel_requested"])
        final = self.finish(caller, run_dir, 130)
        self.assertEqual(final["reason"], "cancel_requested")

    def test_background_descendant_is_cleaned_and_not_reported_as_success(self):
        code = ("import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])")
        caller, run_dir = self.start("background", code)
        final = self.finish(caller, run_dir, 125)
        self.assertEqual(final["reason"], "background_children")
        self.assertEqual(final["task_returncode"], 0)
        self.assertEqual(HELPER.live_group_members(final["pgid"]), [])

    def test_waiting_job_times_out_without_starting_command(self):
        first, first_dir = self.start("holder", "import time; time.sleep(0.8)")
        self.wait_state(first_dir, "running")
        waiting, waiting_dir = self.start("wait-timeout", "from pathlib import Path; Path('must-not-start').touch()", timeout=0.15)
        final = self.finish(waiting, waiting_dir, 124)
        self.assertIsNone(final["task_returncode"])
        self.assertNotIn("task", final)
        self.assertFalse((self.root / "must-not-start").exists())
        self.finish(first, first_dir, 0)


if __name__ == "__main__":
    unittest.main()
