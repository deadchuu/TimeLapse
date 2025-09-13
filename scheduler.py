# scheduler.py
import psutil
import time
import threading
from .utils import get_logger
logger = get_logger()

class DispatcherThrottle:
    """
    Simple dispatcher throttle: before submitting a job, call wait_for_capacity().
    Keeps CPU usage below max_cpu_percent (e.g., 90.0 => leaves ~10% free).
    """
    def __init__(self, max_cpu_percent=90.0, check_interval=0.3):
        self.max_cpu = float(max_cpu_percent)
        self.check_interval = float(check_interval)

    def wait_for_capacity(self):
        # spin until cpu_percent < max_cpu
        while True:
            try:
                cpu = psutil.cpu_percent(interval=self.check_interval)
            except Exception:
                cpu = 0.0
            if cpu < self.max_cpu:
                return
            # else sleep a bit and re-check
            time.sleep(0.05)

class WorkerController:
    """
    Manage a set of worker processes/threads via a pause Event.
    Worker loops should check pause_event.is_set() and pause cooperatively.
    """
    def __init__(self):
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()

    def pause(self):
        logger.info("Pausing workers via WorkerController.")
        self.pause_event.set()

    def resume(self):
        logger.info("Resuming workers via WorkerController.")
        self.pause_event.clear()

    def stop(self):
        logger.info("Stopping workers.")
        self.stop_event.set()
