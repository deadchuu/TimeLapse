# utils.py
import logging
import os
from tqdm import tqdm as _tqdm
import sys

_logger = None

def setup_logging(log_file="timelapse.log", level=logging.INFO):
    global _logger
    logger = logging.getLogger("timelapse")
    logger.setLevel(level)
    # file handler only -> console is reserved for tqdm
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(fmt)
    # remove existing handlers to avoid duplicates
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    logger.addHandler(fh)
    _logger = logger
    logger.info("Logging initialized. Log file: %s", os.path.abspath(log_file))
    return logger

def get_logger():
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger

# tqdm wrapper: use real tqdm; ensure only tqdm prints to console
def get_tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs)

# convenience vprint that writes to logger.debug
def vprint(*args, **kwargs):
    get_logger().debug(" ".join(str(a) for a in args))
