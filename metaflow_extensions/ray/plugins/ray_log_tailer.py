import os
import sys
import time
import glob
import threading
from pathlib import Path


class RayLogTailer:
    """
    Tails Ray log files and streams them to stdout.
    """

    def __init__(self, ray_temp_dir, poll_interval, include_patterns):
        self.ray_temp_dir = ray_temp_dir
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()
        self.thread = None
        self.file_positions = {}
        # Only capture worker stdout/stderr (actor output)
        # worker-*.out/err excludes system components like raylet.out, gcs_server.out
        self.include_patterns = include_patterns

    def _find_ray_session_dir(self):
        """Find the Ray session directory using the session_latest symlink."""
        if not self.ray_temp_dir:
            return None

        # Ray creates a session_latest symlink to the current session directory
        session_latest = os.path.join(self.ray_temp_dir, "session_latest")
        if os.path.exists(session_latest):
            return session_latest
        return None

    def _tail_file(self, filepath):
        """Tail a single log file and print new lines."""
        try:
            # Get current position or start from beginning
            if filepath not in self.file_positions:
                self.file_positions[filepath] = 0

            with open(filepath, "r") as f:
                f.seek(self.file_positions[filepath])
                new_lines = f.readlines()

                if new_lines:
                    # Print with prefix to identify source
                    filename = os.path.basename(filepath)
                    for line in new_lines:
                        print(f"[RAY:{filename}] {line.rstrip()}")

                # Update position
                self.file_positions[filepath] = f.tell()
        except Exception as e:
            # File might not exist yet or might be rotated
            pass

    def _tail_loop(self):
        """Main loop that tails all Ray log files."""
        print("[RAY_LOG_TAILER] Starting Ray log tailer...")

        # Find the session directory once before starting the loop
        session_dir = self._find_ray_session_dir()
        if not session_dir:
            print(
                "[RAY_LOG_TAILER] Ray session directory not found. Cannot tail logs.",
                file=sys.stderr,
            )
            return

        logs_dir = os.path.join(session_dir, "logs")
        if not os.path.exists(logs_dir):
            print(
                f"[RAY_LOG_TAILER] Ray logs directory not found at {logs_dir}. Cannot tail logs.",
                file=sys.stderr,
            )
            return

        # Tail worker output files in the loop
        while not self.stop_event.is_set():
            for pattern in self.include_patterns:
                full_pattern = os.path.join(logs_dir, pattern)
                for log_file in glob.glob(full_pattern):
                    self._tail_file(log_file)

            # Wait for poll_interval or until stop_event is set
            self.stop_event.wait(self.poll_interval)

        print("[RAY_LOG_TAILER] Stopped Ray log tailer")

    def start(self):
        """Start tailing Ray logs in a background thread."""
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._tail_loop, daemon=True)
        self.thread.start()
        print("[RAY_LOG_TAILER] Ray log tailer thread started")

    def stop(self):
        """Stop tailing Ray logs."""
        if not self.thread or not self.thread.is_alive():
            return

        self.stop_event.set()
        self.thread.join(timeout=5)
        print("[RAY_LOG_TAILER] Ray log tailer stopped")
