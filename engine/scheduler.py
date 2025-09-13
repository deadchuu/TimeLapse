# engine/scheduler.py
import psutil
import time
import threading
from engine.utils import get_logger
logger = get_logger()

class DispatcherThrottle:
    def __init__(self, max_cpu_percent=90.0, check_interval=0.3):
        self.max_cpu = float(max_cpu_percent)
        self.check_interval = float(check_interval)

    def wait_for_capacity(self):
        while True:
            try:
                cpu = psutil.cpu_percent(interval=self.check_interval)
            except Exception:
                cpu = 0.0
            if cpu < self.max_cpu:
                return
            time.sleep(0.05)

class WorkerController:
    def __init__(self):
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()

    def pause(self):
        logger.info("Pausing workers.")
        self.pause_event.set()

    def resume(self):
        logger.info("Resuming workers.")
        self.pause_event.clear()

    def stop(self):
        logger.info("Stopping workers.")
        self.stop_event.set()
