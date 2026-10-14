import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_lifecycle import stop_worker_thread


class WorkerLifecycleTests(unittest.TestCase):
    def test_stops_worker_before_quitting_and_waiting_for_thread(self):
        events = []

        class Worker:
            def stop(self):
                events.append("worker.stop")

        class Thread:
            def quit(self):
                events.append("thread.quit")

            def wait(self, *args):
                events.append(("thread.wait", args))

        stop_worker_thread(Worker(), Thread())

        self.assertEqual(
            events,
            ["worker.stop", "thread.quit", ("thread.wait", ())],
        )

    def test_allows_missing_worker_or_thread(self):
        events = []

        class Worker:
            def stop(self):
                events.append("worker.stop")

        class Thread:
            def quit(self):
                events.append("thread.quit")

            def wait(self):
                events.append("thread.wait")

        stop_worker_thread(None, Thread())
        stop_worker_thread(Worker(), None)

        self.assertEqual(
            events,
            ["thread.quit", "thread.wait", "worker.stop"],
        )


if __name__ == "__main__":
    unittest.main()