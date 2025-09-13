# io_utils.py
import os
import shutil
import tempfile
import threading
import atexit
import signal
from utils import get_logger, vprint

_logger = get_logger()

_created = []
_lock = threading.Lock()

def register_temp_path(path):
    with _lock:
        if path not in _created:
            _created.append(path)
            vprint("Registered temp path:", path)

def unregister_temp_path(path):
    with _lock:
        if path in _created:
            _created.remove(path)
            vprint("Unregistered temp path:", path)

def cleanup_all_temp_paths():
    with _lock:
        paths = list(_created)
    for p in reversed(paths):
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
                vprint("Removed temp folder:", p)
        except Exception as e:
            _logger.exception("Failed to cleanup temp folder %s: %s", p, e)
        finally:
            unregister_temp_path(p)

# register cleanup at exit and signals
atexit.register(cleanup_all_temp_paths)
def _signal_handler(sig, frame):
    _logger.info("Signal %s received -> cleaning up temporaries.", sig)
    cleanup_all_temp_paths()
for s in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(s, _signal_handler)
    except Exception:
        pass

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def basename(path):
    return os.path.basename(os.path.normpath(path))

def list_png_files(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")])

# Try link/copy helpers
def create_link_or_copy(src, dest):
    try:
        os.link(src, dest)
        return True
    except Exception:
        pass
    try:
        os.symlink(src, dest)
        return True
    except Exception:
        pass
    try:
        shutil.copy2(src, dest)
        return True
    except Exception:
        return False

def create_links_for_batch(src_files, temp_folder):
    ensure_dir(temp_folder)
    saved = []
    mapping = {}
    for i, src in enumerate(src_files):
        name = f"{i:06d}.png"
        dest = os.path.join(temp_folder, name)
        ok = create_link_or_copy(src, dest)
        if not ok:
            _logger.warning("Failed to link/copy %s -> %s", src, dest)
            continue
        saved.append(dest)
        mapping[dest] = src
    return saved, mapping

def make_unique_tempdir(prefix="tmp_", base_dir=None):
    tmp = tempfile.mkdtemp(prefix=prefix, dir=base_dir)
    register_temp_path(tmp)
    return tmp
