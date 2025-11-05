import os
import time
import glob
import threading
from pathlib import Path


class RayLogTailer:
    """
    Tails Ray log files and streams them to stdout so they appear in CloudWatch logs.
    """

    def __init__(self, ray_session_dir=None, poll_interval=1.0, include_patterns=None):
        self.ray_session_dir = ray_session_dir
        self.poll_interval = poll_interval
        self.running = False
        self.thread = None
        self.file_positions = {}
        # Default: only capture worker stdout/stderr (actor output)
        # worker-*.out/err excludes system components like raylet.out, gcs_server.out
        self.include_patterns = include_patterns or ["worker-*.out", "worker-*.err"]

    def _find_ray_session_dir(self):
        """Find the Ray session directory."""
        if self.ray_session_dir:
            return self.ray_session_dir

        # Ray typically creates session dirs in /tmp/ray/session_*
        ray_tmp_dirs = glob.glob("/tmp/ray/session_*")
        if ray_tmp_dirs:
            # Get the most recent session directory
            return max(ray_tmp_dirs, key=os.path.getmtime)
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

        while self.running:
            session_dir = self._find_ray_session_dir()

            if session_dir:
                logs_dir = os.path.join(session_dir, "logs")
                if os.path.exists(logs_dir):
                    # Tail worker output files only (worker-*.out, worker-*.err)
                    for pattern in self.include_patterns:
                        full_pattern = os.path.join(logs_dir, pattern)
                        for log_file in glob.glob(full_pattern):
                            self._tail_file(log_file)

            time.sleep(self.poll_interval)

        print("[RAY_LOG_TAILER] Stopped Ray log tailer")

    def start(self):
        """Start tailing Ray logs in a background thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._tail_loop, daemon=True)
        self.thread.start()
        print("[RAY_LOG_TAILER] Ray log tailer thread started")

    def stop(self):
        """Stop tailing Ray logs."""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("[RAY_LOG_TAILER] Ray log tailer stopped")
