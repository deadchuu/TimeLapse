# progress.py (новий файл у components/)
import sys
import time
import threading

class ProgressBar:
    def __init__(self, total_frames, poll_interval=0.2):
        self.total_frames = total_frames
        self.poll_interval = poll_interval
        self.current = 0
        self.stop_flag = False
        self._lock = threading.Lock()

    def update(self, current):
        with self._lock:
            self.current = current

    def start(self):
        def run():
            while not self.stop_flag:
                self._render()
                time.sleep(self.poll_interval)
            self._render(final=True)
        t = threading.Thread(target=run, daemon=True)
        t.start()

    def stop(self):
        self.stop_flag = True

    def _render(self, final=False):
        progress = self.current / self.total_frames
        bar_len = 40
        filled = int(bar_len * progress)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = f"\r[{bar}] {self.current}/{self.total_frames} ({progress*100:.1f}%)"
        if final:
            msg += "\n"
        sys.stdout.write(msg)
        sys.stdout.flush()
