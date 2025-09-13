#!/usr/bin/env python3
"""
timelapse_fullsize_auto_gpu.py - patched version

Key changes:
- Unique temp folders using tempfile.mkdtemp and global registry for cleanup
- atexit + signal handlers to ensure temp cleanup on exit
- More robust ffmpeg image2pipe streaming via explicit '-vcodec png' and writer thread
- submit_segment_stream_or_fallback registers fallback temp folder and removes it after use
- concat_segments_ffmpeg merges segments and only deletes them after successful merge
- Passes fps into select_frames modes
- Optional SUBTASKS_PER_BATCH to split batch selection into parallel subtasks
- Minimal behavioral changes otherwise
"""
import os
import sys
import time
import shutil
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import platform
import ctypes
import importlib.util
import tempfile
import io
import threading
import atexit
import signal
import uuid

# Optional: Pillow and pyvips for pixel-accurate diffs
try:
    from PIL import Image, ImageChops
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import pyvips
    VIPS_AVAILABLE = True
except Exception:
    VIPS_AVAILABLE = False

# tqdm wrapper: use real tqdm if available, else graceful fallback that prints one warning
try:
    from tqdm import tqdm as _tqdm
    TQDM_AVAILABLE = True
except Exception:
    _tqdm = None
    TQDM_AVAILABLE = False

def get_tqdm(iterable=None, **kwargs):
    """
    Return tqdm(iterator) if tqdm installed, otherwise return the original iterable.
    Prints a one-time notice when tqdm is not available.
    Use get_tqdm(...) everywhere instead of calling tqdm(...) directly.
    """
    if _tqdm:
        return _tqdm(iterable, **kwargs)
    if not getattr(get_tqdm, "_warned", False):
        print("Note: tqdm not installed; progress bars disabled. Install: pip install tqdm")
        get_tqdm._warned = True
    if iterable is None:
        return []
    return iterable

# ---------- CONFIG ----------
# base output folder: create 'timelaps' in current working directory (where script is executed)
BASE_OUTPUT_DIR = os.path.join(os.path.abspath(os.getcwd()), "timelaps")
BATCH_SIZE = 1000               # unchanged default
BATCH_WINDOW = 3                # process groups of 3 batches at a time
DIFF_PIXEL_THRESHOLD = 0.0002
DIFF_VIPS_THRESHOLD = 0.0002   # average pixel difference threshold for pyvips
FFMPEG_CRF = 18
FFMPEG_PRESET = "medium"
MAX_FFMPEG_PROCS = 2            # limit concurrent ffmpeg processes to 2
WORKERS = max(1, multiprocessing.cpu_count() - 1)
THREAD_QUEUE_SIZE = 1024
VERBOSE = False                 # set True for internal debug prints
MODES_FOLDER = os.path.join(os.path.dirname(__file__), "modes")

# Optional subtask split: split each BATCH into SUBTASKS_PER_BATCH subranges for parallel selection
SUBTASKS_PER_BATCH = 1  # set to 4 to split each batch into 4 subtasks (e.g., 4 x 250 for BATCH_SIZE 1000)
# -----------------------------

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# ---------- reliable temp cleanup registry ----------
created_temp_paths_global = []
_created_temp_lock = threading.Lock()

def _register_temp_path(p):
    try:
        with _created_temp_lock:
            if p not in created_temp_paths_global:
                created_temp_paths_global.append(p)
    except Exception:
        pass

def _unregister_temp_path(p):
    try:
        with _created_temp_lock:
            if p in created_temp_paths_global:
                created_temp_paths_global.remove(p)
    except Exception:
        pass

def cleanup_all_temp_paths():
    # remove in reverse order to be safe
    with _created_temp_lock:
        paths = list(created_temp_paths_global)
    for p in reversed(paths):
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
                vprint("Removed temp folder:", p)
            with _created_temp_lock:
                if p in created_temp_paths_global:
                    created_temp_paths_global.remove(p)
        except Exception as e:
            vprint("Failed to cleanup temp folder:", p, e)

def _signal_handler(sig, frame):
    vprint(f"Signal received: {sig}. Cleaning temporary files.")
    cleanup_all_temp_paths()
    # Re-raise KeyboardInterrupt for SIGINT to allow default exit
    try:
        if sig == signal.SIGINT:
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass

# register handlers
atexit.register(cleanup_all_temp_paths)
for s in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(s, _signal_handler)
    except Exception:
        pass

# ---------- helpers ----------
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def check_ffprobe():
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def list_png_files(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")])

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def basename(path):
    return os.path.basename(os.path.normpath(path))

# ---------- Link / copy helpers ----------
def create_link_or_copy(src, dest):
    """Try hardlink -> symlink -> copy. Return True on success."""
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
    it = enumerate(src_files)
    it = get_tqdm(it, desc="Creating links", unit="file")
    for i, src in it:
        name = f"{i:06d}.png"
        dest = os.path.join(temp_folder, name)
        ok = create_link_or_copy(src, dest)
        if not ok:
            vprint(f"Warning: failed to link/copy '{src}' -> '{dest}' (suppressed).")
            continue
        saved.append(dest)
        mapping[dest] = src
    return saved, mapping

# ---------- Full-size comparisons ----------
def compare_pair_pillow_full(path_a, path_b, region=None):
    try:
        a = Image.open(path_a).convert("RGBA")
        b = Image.open(path_b).convert("RGBA")
    except Exception as e:
        vprint("Pillow open failed:", e)
        return float('inf')
    if region:
        x1, y1, x2, y2 = region
        a = a.crop((x1, y1, x2, y2))
        b = b.crop((x1, y1, x2, y2))
    if a.size != b.size:
        return float('inf')
    diff = ImageChops.difference(a, b)
    diff_rgb = diff.convert("RGB")
    diff_count = 0
    for px in diff_rgb.getdata():
        if px[0] != 0 or px[1] != 0 or px[2] != 0:
            diff_count += 1
    return diff_count

def compare_pair_vips_full(path_a, path_b, region=None):
    try:
        a = pyvips.Image.new_from_file(path_a, access='sequential')
        b = pyvips.Image.new_from_file(path_b, access='sequential')
        if region:
            x1, y1, x2, y2 = region
            w = x2 - x1
            h = y2 - y1
            a = a.crop(x1, y1, w, h)
            b = b.crop(x1, y1, w, h)
        if a.hasalpha():
            a = a[:3]
        if b.hasalpha():
            b = b[:3]
        diff = (a - b).abs()
        avg = diff.avg()
        return float(avg)
    except Exception as e:
        vprint("pyvips compare failed:", e)
        return float('inf')

def files_are_byte_equal(path_a, path_b):
    try:
        if os.path.getsize(path_a) != os.path.getsize(path_b):
            return False
        with open(path_a, 'rb') as fa, open(path_b, 'rb') as fb:
            while True:
                ba = fa.read(65536)
                bb = fb.read(65536)
                if not ba and not bb:
                    return True
                if ba != bb:
                    return False
    except Exception as e:
        vprint("byte compare failed:", e)
        return False

def is_frame_changed(prev, curr, region=None):
    if PIL_AVAILABLE:
        diff = compare_pair_pillow_full(prev, curr, region=region)
        if diff == float('inf'):
            return True
        # threshold meaning: treat DIFF_PIXEL_THRESHOLD as absolute pixel count when >1, else fraction
        if DIFF_PIXEL_THRESHOLD >= 1:
            return diff > DIFF_PIXEL_THRESHOLD
        else:
            # fraction of pixels
            try:
                w,h = get_image_size(prev)
                total = w*h
                return (diff / float(total)) > DIFF_PIXEL_THRESHOLD
            except Exception:
                return diff > 0
    elif VIPS_AVAILABLE:
        avg = compare_pair_vips_full(prev, curr, region=region)
        if avg == float('inf'):
            return True
        return avg > DIFF_VIPS_THRESHOLD
    else:
        return not files_are_byte_equal(prev, curr)

# ---------- multiprocessing compare ----------
def _compare_worker_full(args):
    prev, curr, region = args
    try:
        if is_frame_changed(prev, curr, region):
            return curr
    except Exception:
        return curr
    return None

def filter_changed_frames_multicore(saved_paths, region=None, prev_last_original=None):
    if not saved_paths:
        return []
    selected = []
    first = saved_paths[0]
    if prev_last_original:
        try:
            if is_frame_changed(prev_last_original, first, region):
                selected.append(first)
        except Exception:
            selected.append(first)
    else:
        selected.append(first)
    tasks = []
    for i in range(1, len(saved_paths)):
        tasks.append((saved_paths[i-1], saved_paths[i], region))
    if not tasks:
        return selected
    results = []
    with multiprocessing.Pool(processes=WORKERS) as pool:
        imap = pool.imap_unordered(_compare_worker_full, tasks)
        for res in get_tqdm(imap, total=len(tasks), desc="Comparing frames", unit="cmp"):
            if res:
                results.append(res)
    def idx_of(p):
        try:
            return int(os.path.splitext(os.path.basename(p))[0])
        except Exception:
            return 0
    results_sorted = sorted(set(results), key=idx_of)
    selected.extend(results_sorted)
    return selected

# ---------- helpers ----------
def get_image_size(path):
    if VIPS_AVAILABLE:
        try:
            img = pyvips.Image.new_from_file(path, access='sequential')
            return img.width, img.height
        except Exception:
            pass
    if PIL_AVAILABLE:
        try:
            with Image.open(path) as im:
                return im.width, im.height
        except Exception:
            pass
    if check_ffprobe():
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", path
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            if "," in out:
                w, h = out.split(",")
            elif "x" in out:
                w, h = out.split("x")
            else:
                parts = out.split()
                if parts:
                    w, h = parts[0].split(",")
                else:
                    raise Exception("Cannot parse ffprobe output")
            return int(w), int(h)
        except Exception as e:
            vprint("ffprobe parse failed:", e)
            pass
    raise RuntimeError("Cannot determine image size (install pyvips or Pillow, or ensure ffprobe is available).")

# ---------- GPU detection ----------
def try_load_nvcuda():
    if platform.system() != "Windows":
        return False
    try:
        ctypes.WinDLL("nvcuda.dll")
        return True
    except Exception:
        return False

def list_video_controllers_windows():
    """Return list of video controller names on Windows via wmic."""
    try:
        out = subprocess.check_output(["wmic", "path", "win32_VideoController", "get", "name"], stderr=subprocess.DEVNULL).decode(errors='ignore')
        lines = [l.strip() for l in out.splitlines() if l.strip() and 'Name' not in l]
        return lines
    except Exception as e:
        vprint("wmic failed:", e)
        return []

def detect_ffmpeg_hw_encoders():
    """Inspect ffmpeg encoders and return encoders text."""
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT).decode('utf-8', errors='ignore')
        return out
    except Exception as e:
        vprint("ffmpeg encoders query failed:", e)
        return ""

def auto_detect_best_hw_encoder():
    encoders_text = detect_ffmpeg_hw_encoders()
    gpu_names = []
    if platform.system() == "Windows":
        gpu_names = list_video_controllers_windows()
        vprint("Detected GPUs via wmic:", gpu_names)
    if "h264_nvenc" in encoders_text:
        nvcuda_ok = try_load_nvcuda()
        nvidia_present = any("NVIDIA" in name.upper() for name in gpu_names)
        if nvcuda_ok or nvidia_present:
            return "h264_nvenc"
    if "h264_qsv" in encoders_text:
        intel_present = any("INTEL" in name.upper() for name in gpu_names) or ("Intel" in platform.processor() or "GenuineIntel" in platform.platform())
        if intel_present or platform.system() != "Windows":
            return "h264_qsv"
    if "h264_vaapi" in encoders_text:
        return "h264_vaapi"
    return None

# Pre-detect available hw encoder (but user decides whether to use)
HW_ENCODER_DETECTED = None
if check_ffmpeg():
    HW_ENCODER_DETECTED = auto_detect_best_hw_encoder()
    if HW_ENCODER_DETECTED:
        print(f"Detected hardware encoder candidate: {HW_ENCODER_DETECTED}")
    else:
        print("No hardware encoder detected by ffmpeg (NVENC/QSV/VAAPI). Software encoding (libx264) will be used unless forced.")
else:
    print("ffmpeg not found in PATH - please install ffmpeg and add to PATH.")
    HW_ENCODER_DETECTED = None

# ---------- Modes loader ----------
def load_modes_folder(modes_folder):
    modes = []
    if not os.path.isdir(modes_folder):
        vprint("Modes folder does not exist:", modes_folder)
        return modes
    for fname in sorted(os.listdir(modes_folder)):
        if not fname.lower().endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(modes_folder, fname)
        try:
            spec = importlib.util.spec_from_file_location(f"modes.{os.path.splitext(fname)[0]}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "get_mode"):
                m = mod.get_mode()
                # expected keys: id, name, desc, enable_selection (bool), build_ffmpeg_cmd
                if all(k in m for k in ("id", "name", "desc", "build_ffmpeg_cmd")):
                    modes.append(m)
                else:
                    vprint(f"Mode {fname} missing required keys, skipping.")
            else:
                vprint(f"No get_mode() in {fname}, skipping.")
        except Exception as e:
            vprint(f"Failed to load mode {fname}:", e)
    return modes

# ---------- ffmpeg helpers (pattern-based) ----------
def make_segment_from_images_ffmpeg(temp_img_folder, fps, out_segment_path, encoder_hint=None, mode=None):
    """
    Create one encoded segment from images based on pattern (temp_img_folder/%06d.png).
    This is the original (fallback) method.
    """
    first_img = os.path.join(temp_img_folder, "000000.png")
    if not os.path.exists(first_img):
        files = sorted([os.path.join(temp_img_folder, f) for f in os.listdir(temp_img_folder) if f.lower().endswith(".png")])
        if not files:
            vprint("No images found for ffmpeg segment.")
            return False
        first_img = files[0]
    try:
        w, h = get_image_size(first_img)
    except Exception as e:
        vprint("Cannot determine image size:", e)
        return False

    encoder = encoder_hint if encoder_hint else (HW_ENCODER_DETECTED if HW_ENCODER_DETECTED else "libx264")

    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=white:s={w}x{h}:r={int(fps)}",
        "-framerate", str(int(fps)),
        "-thread_queue_size", str(THREAD_QUEUE_SIZE),
        "-start_number", "0",
        "-i", os.path.join(temp_img_folder, "%06d.png"),
    ]

    if encoder == "libx264" or encoder is None:
        cmd = base + [
            "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", str(FFMPEG_CRF),
            "-pix_fmt", "yuv420p", out_segment_path
        ]
    else:
        if encoder == "h264_nvenc":
            cmd = base + [
                "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=nv12", 
                "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr_hq", "-cq", "19", "-pix_fmt", "yuv420p",
                out_segment_path
            ]
        elif encoder == "h264_qsv":
            cmd = base + [
                "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
                "-c:v", "h264_qsv", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                out_segment_path
            ]
        elif encoder == "h264_vaapi":
            vaapi_dev = "/dev/dri/renderD128"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-vaapi_device", vaapi_dev,
                "-f", "lavfi", "-i", f"color=white:s={w}x{h}:r={int(fps)}",
                "-framerate", str(int(fps)),
                "-start_number", "0",
                "-i", os.path.join(temp_img_folder, "%06d.png"),
                "-vf", "format=nv12,hwupload,overlay=shortest=1",
                "-c:v", "h264_vaapi",
                out_segment_path
            ]
        else:
            cmd = base + [
                "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
                "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", str(FFMPEG_CRF),
                "-pix_fmt", "yuv420p", out_segment_path
            ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        vprint("ffmpeg failed with encoder", encoder, "->", e)
        # fallback to libx264 quietly
        if encoder != "libx264":
            try:
                fallback = base + [
                    "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
                    "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", str(FFMPEG_CRF),
                    "-pix_fmt", "yuv420p", out_segment_path
                ]
                subprocess.run(fallback, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e2:
                vprint("Fallback libx264 failed:", e2)
                return False
        return False
    except Exception as ex:
        vprint("ffmpeg unknown error:", ex)
        return False

# ---------- ffmpeg helpers (streaming) ----------
def _writer_thread_for_proc(proc, image_paths, stop_event):
    """
    Writer thread: writes each PNG file bytes to proc.stdin.
    stop_event can be used to signal early stop.
    """
    try:
        for path in image_paths:
            if stop_event.is_set():
                break
            try:
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        proc.stdin.write(chunk)
            except BrokenPipeError:
                break
            except Exception as e:
                vprint("Writer thread read/write error:", e)
                continue
        try:
            proc.stdin.close()
        except Exception:
            pass
    except Exception as e:
        vprint("Writer thread fatal error:", e)
        try:
            proc.stdin.close()
        except Exception:
            pass

def make_segment_from_images_ffmpeg_stream(image_paths, fps, out_segment_path, encoder_hint=None, mode=None, timeout=None):
    """
    Stream PNG images to ffmpeg using image2pipe. Returns True on success.
    - Uses writer thread to avoid blocking main thread.
    - Uses '-vcodec png' to make image2pipe input explicit.
    """
    if not image_paths:
        vprint("No image paths provided to stream.")
        return False

    first_img = image_paths[0]
    try:
        w, h = get_image_size(first_img)
    except Exception as e:
        vprint("Cannot determine image size for stream:", e)
        return False

    encoder = encoder_hint if encoder_hint else (HW_ENCODER_DETECTED if HW_ENCODER_DETECTED else "libx264")

    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=white:s={w}x{h}:r={int(fps)}",
        "-framerate", str(int(fps)),
        "-f", "image2pipe", "-vcodec", "png", "-i", "-",  # read PNG images from stdin
    ]

    if encoder == "libx264" or encoder is None:
        cmd = base + [
            "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", str(FFMPEG_CRF),
            "-pix_fmt", "yuv420p", out_segment_path
        ]
    else:
        if encoder == "h264_nvenc":
            cmd = base + [
                "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=nv12",
                "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr_hq", "-cq", "19", "-pix_fmt", "yuv420p",
                out_segment_path
            ]
        elif encoder == "h264_qsv":
            cmd = base + [
                "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
                "-c:v", "h264_qsv", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                out_segment_path
            ]
        elif encoder == "h264_vaapi":
            vprint("VAAPI streaming not supported reliably; falling back to pattern method.")
            return False
        else:
            cmd = base + [
                "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
                "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", str(FFMPEG_CRF),
                "-pix_fmt", "yuv420p", out_segment_path
            ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except Exception as e:
        vprint("Failed to start ffmpeg process for streaming:", e)
        return False

    stop_event = threading.Event()
    writer = threading.Thread(target=_writer_thread_for_proc, args=(proc, image_paths, stop_event), daemon=True)
    writer.start()

    try:
        if timeout:
            ret = proc.wait(timeout=timeout)
        else:
            ret = proc.wait()
        writer.join(timeout=1.0)
        if ret == 0:
            return True
        else:
            vprint("ffmpeg (stream) exited with code:", ret)
            return False
    except subprocess.TimeoutExpired:
        vprint("ffmpeg stream timed out; killing.")
        stop_event.set()
        try:
            proc.kill()
        except Exception:
            pass
        return False
    except Exception as ex:
        vprint("Streaming error:", ex)
        stop_event.set()
        try:
            proc.kill()
        except Exception:
            pass
        return False

def submit_segment_stream_or_fallback(image_list, temp_folder_pattern, seg_path, encoder, mode, fps_param):
    """
    Try to stream image_list to ffmpeg. If it fails, create temp_folder_pattern with sequential files
    and call pattern-based encoder. The fallback temp folder is removed before returning.
    """
    ok = make_segment_from_images_ffmpeg_stream(image_list, fps_param, seg_path, encoder_hint=encoder, mode=mode)
    if ok:
        return True

    tmp = None
    try:
        # make unique tmp under same dir as temp_folder_pattern if possible
        base_dir = os.path.dirname(temp_folder_pattern) or None
        tmp = tempfile.mkdtemp(prefix="fallback_", dir=base_dir)
        _register_temp_path(tmp)
        # link/copy into sequential names
        for i, src in enumerate(image_list):
            dest = os.path.join(tmp, f"{i:06d}.png")
            create_link_or_copy(src, dest)
        ok2 = make_segment_from_images_ffmpeg(tmp, fps_param, seg_path, encoder_hint=encoder, mode=mode)
    except Exception as e:
        vprint("Fallback pattern encoding failed:", e)
        ok2 = False

    # cleanup fallback temp folder if we created it
    try:
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp)
            _unregister_temp_path(tmp)
    except Exception as e:
        vprint("Failed to remove fallback temp folder:", tmp, e)
    return ok2

# ---------- concat helper ----------
def concat_segments_ffmpeg(segment_paths, out_final):
    if not segment_paths:
        return False
    list_file = None
    try:
        list_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", dir=os.path.dirname(out_final) or None)
        for p in segment_paths:
            # escape single quotes by replacing ' with '"'"'
            safe = p.replace("'", "'\"'\"'")
            list_file.write(f"file '{safe}'\n")
        list_file.flush()
        list_file.close()
        # try fast concat
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list_file.name, "-c", "copy", out_final]
        try:
            subprocess.run(cmd, check=True)
            try:
                os.remove(list_file.name)
            except Exception:
                pass
            return True
        except subprocess.CalledProcessError:
            vprint("Fast concat failed; falling back to re-encode concat.")
            # fallback: re-encode with concat filter
            inputs = []
            for p in segment_paths:
                inputs += ["-i", p]
            n = len(segment_paths)
            cmd2 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + inputs + ["-filter_complex", f"concat=n={n}:v=1:a=0", "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", str(FFMPEG_CRF), out_final]
            subprocess.run(cmd2, check=True)
            try:
                os.remove(list_file.name)
            except Exception:
                pass
            return True
    except Exception as e:
        vprint("Concat failed:", e)
        try:
            if list_file:
                os.remove(list_file.name)
        except Exception:
            pass
        return False

# ---------- main pipeline ----------
def main():
    # Load modes
    modes = load_modes_folder(MODES_FOLDER)

    # Print modes first
    print("=== AVAILABLE MODES ===")
    if not modes:
        print("No modes found in:", MODES_FOLDER)
        print("Create a 'modes' folder and add python modules with get_mode().")
    else:
        for i, m in enumerate(modes):
            print(f"[{i}] {m['name']} - {m.get('desc','')}")
    print("=======================")

    # Ask mode index (if any)
    chosen_mode = None
    if modes:
        mode_sel = input(f"Select mode index [default 0]: ").strip()
        try:
            idx = int(mode_sel) if mode_sel != "" else 0
            if 0 <= idx < len(modes):
                chosen_mode = modes[idx]
            else:
                print("Invalid mode index, using default (0).")
                chosen_mode = modes[0]
        except Exception:
            print("Invalid input, using default mode 0.")
            chosen_mode = modes[0]

    # The three parameters that must always be asked / shown
    source_folder = input("Enter path to folder with PNG files: ").strip()
    if not os.path.isdir(source_folder):
        print("Specified folder does not exist.")
        return

    fps_in = input("Enter desired FPS (e.g. 30) [default 30]: ").strip()
    try:
        fps = float(fps_in) if fps_in else 30.0
    except Exception:
        fps = 30.0

    # hardware acceleration decision (same as before)
    use_hw = False
    if HW_ENCODER_DETECTED:
        resp = input(f"Use hardware acceleration with detected encoder '{HW_ENCODER_DETECTED}'? (y/N): ").strip().lower()
        use_hw = resp == 'y' or resp == 'yes'
    else:
        resp = input("No hw encoder detected automatically. Do you want to try hardware acceleration anyway? (y/N): ").strip().lower()
        use_hw = resp == 'y' or resp == 'yes'
    if use_hw and HW_ENCODER_DETECTED:
        chosen_encoder = HW_ENCODER_DETECTED
        print(f"Hardware acceleration requested -> using encoder: {chosen_encoder}")
    else:
        chosen_encoder = None
        if use_hw and not HW_ENCODER_DETECTED:
            print("Hardware acceleration requested but no encoder detected -> will try software libx264 (or ffmpeg fallback).")
        else:
            print("Hardware acceleration disabled -> using software encoding (libx264).")

    # If chosen mode supports selection, ask ROI but it's optional now.
    region = None
    do_selected = False
    if chosen_mode and chosen_mode.get("enable_selection"):
        print("Enter ROI coordinates for change detection (x1 y1 x2 y2). Press Enter to skip -> whole image will be used.")
        coords = input("ROI or press Enter to use whole image: ").strip()
        if coords == "":
            region = None
            do_selected = True
            print("No ROI provided -> selection will analyze whole image.")
        else:
            try:
                x1, y1, x2, y2 = map(int, coords.split())
                region = (x1, y1, x2, y2)
                do_selected = True
                print(f"ROI set to: {region}")
            except Exception:
                print("Invalid coordinates format. Proceeding with whole-image selection (region=None).")
                region = None
                do_selected = True
    else:
        do_selected = False
        region = None

    # continue original pipeline with chosen_mode passed into ffmpeg builder calls
    if not check_ffmpeg():
        print("ffmpeg is not found in PATH. Please install ffmpeg and add it to PATH.")
        return

    if not (PIL_AVAILABLE or VIPS_AVAILABLE):
        print("Warning: Pillow and pyvips are not installed. Falling back to byte-wise comparison. For pixel-accurate diffs, install Pillow: pip install pillow")

    files = list_png_files(source_folder)
    if not files:
        print("No PNG files found in source folder.")
        return

    # Setup global overall frames progress bar
    total_frames = len(files)
    overall_frames_pbar = None
    if TQDM_AVAILABLE:
        overall_frames_pbar = _tqdm(total=total_frames, desc="Overall frames", unit="frame")
    else:
        overall_frames_pbar = None

    source_name = basename(source_folder)
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, source_name)
    ensure_dir(OUTPUT_DIR)

    mode_id = chosen_mode["id"] if chosen_mode else "default"
    print(f"Found {len(files)} frames. Output directory: {OUTPUT_DIR}")
    print(f"Mode: {mode_id}  |  Using {WORKERS} processes for comparisons; up to {MAX_FFMPEG_PROCS} concurrent ffmpeg encoders (batch window {BATCH_WINDOW}).")

    total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
    prev_last_original = None
    segment_paths = []   # will contain either all-segments or selected-segments depending on do_selected

    overall_start = time.time()
    ffmpeg_executor = ThreadPoolExecutor(max_workers=MAX_FFMPEG_PROCS)
    batch_iter_outer = range(0, total_batches, BATCH_WINDOW)
    batch_iter_outer = get_tqdm(batch_iter_outer, desc="Batch groups", unit="group")

    for group_start in batch_iter_outer:
        group = list(range(group_start, min(group_start + BATCH_WINDOW, total_batches)))
        group_info = {}
        temp_to_cleanup_for_group = []

        # Create links and submit encodes for the chosen "type" (all vs selected)
        for batch_idx in group:
            s = batch_idx * BATCH_SIZE
            e = min(s + BATCH_SIZE, len(files))
            batch_src = files[s:e]
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({s}..{e-1})")

            # create unique temp folder for this batch
            temp_batch = tempfile.mkdtemp(prefix=f"temp_batch_{batch_idx:03d}_", dir=OUTPUT_DIR)
            _register_temp_path(temp_batch)
            temp_to_cleanup_for_group.append(temp_batch)

            # create numbered temp copies for compat (used by comparisons). We still keep original batch_src for streaming.
            saved_paths, mapping = create_links_for_batch(batch_src, temp_batch)

            seg_name = os.path.join(OUTPUT_DIR, f"segment_{mode_id}_{batch_idx:03d}.mp4")
            sel_name = os.path.join(OUTPUT_DIR, f"segment_selected_{mode_id}_{batch_idx:03d}.mp4")
            group_info[batch_idx] = {
                "batch_src": batch_src,
                "temp_batch": temp_batch,
                "saved_paths": saved_paths,
                "mapping": mapping,
                "seg_all": seg_name,
                "seg_sel": sel_name,
                "selected_temp": None,
                "created_temp_folder_by_mode": None
            }
            # Submit an 'all' encode only if not doing selection flow; otherwise we'll create selected later
            if not do_selected:
                future_all = ffmpeg_executor.submit(
                    submit_segment_stream_or_fallback,
                    batch_src, temp_batch, seg_name, chosen_encoder, chosen_mode, fps
                )
                group_info[batch_idx]["future_all"] = future_all
                segment_paths.append(seg_name)
            else:
                # If doing selection, we don't start sel segment yet; will after selection created.
                pass

        # For each batch: do comparisons / selection if needed and submit selected encode
        for batch_idx in group:
            info = group_info[batch_idx]
            saved_paths = info["saved_paths"]
            mapping = info["mapping"]
            if not saved_paths:
                if info["batch_src"]:
                    prev_last_original = info["batch_src"][-1]
                continue

            if do_selected:
                # support optional splitting of batch into subtasks for parallel selection
                selected_temp = None
                created_temp_folder_by_mode = None

                # If mode provides select_frames -> call it; else fallback to default filter_changed_frames_multicore
                if chosen_mode and callable(chosen_mode.get("select_frames")):
                    try:
                        # pass fps to modes so they can scale life by fps
                        selected_result = chosen_mode["select_frames"](
                            saved_paths=saved_paths,
                            mapping=mapping,
                            region=region,
                            prev_last_original=prev_last_original,
                            output_dir=OUTPUT_DIR,
                            batch_idx=batch_idx,
                            fps=fps
                        )
                    except TypeError:
                        # older signature without kwargs
                        try:
                            selected_result = chosen_mode["select_frames"](saved_paths, mapping, region, prev_last_original, OUTPUT_DIR, batch_idx, fps)
                        except Exception:
                            # fallback
                            selected_result = chosen_mode["select_frames"](saved_paths, mapping, region, prev_last_original)
                    if isinstance(selected_result, dict):
                        temp_sel_folder = selected_result.get("temp_sel_folder")
                        sel_paths = selected_result.get("selected_paths", [])
                        if temp_sel_folder and os.path.isdir(temp_sel_folder) and sel_paths:
                            selected_temp = sel_paths
                            created_temp_folder_by_mode = temp_sel_folder
                            # remember for cleanup
                            temp_to_cleanup_for_group.append(temp_sel_folder)
                            _register_temp_path(temp_sel_folder)
                        else:
                            selected_temp = filter_changed_frames_multicore(saved_paths, region=region, prev_last_original=prev_last_original)
                    elif isinstance(selected_result, (list, tuple)):
                        selected_temp = list(selected_result)
                    else:
                        selected_temp = filter_changed_frames_multicore(saved_paths, region=region, prev_last_original=prev_last_original)
                else:
                    # default selection
                    if SUBTASKS_PER_BATCH is None or SUBTASKS_PER_BATCH <= 1:
                        selected_temp = filter_changed_frames_multicore(saved_paths, region=region, prev_last_original=prev_last_original)
                    else:
                        # split saved_paths into subtasks and run filter_changed_frames_multicore in parallel
                        n = SUBTASKS_PER_BATCH
                        chunk_size = max(1, len(saved_paths) // n)
                        sub_ranges = [saved_paths[i:i+chunk_size] for i in range(0, len(saved_paths), chunk_size)]
                        results_acc = []
                        with multiprocessing.Pool(processes=min(len(sub_ranges), WORKERS)) as pool:
                            tasks = []
                            firsts = []
                            # For accurate boundaries, pass prev_last_original only to first chunk
                            for si, chunk in enumerate(sub_ranges):
                                if not chunk:
                                    continue
                                prv = None
                                if si == 0:
                                    prv = prev_last_original
                                # We'll map each chunk into a simple comparator chain inside worker by calling filter_changed_frames_multicore
                                tasks.append(pool.apply_async(filter_changed_frames_multicore, (chunk, region, prv)))
                            for t in tasks:
                                try:
                                    res = t.get()
                                    if res:
                                        results_acc.extend(res)
                                except Exception:
                                    pass
                        # results_acc contains selected file paths (from saved_paths) possibly unordered
                        # sort by index filename
                        def idx_of(p):
                            try:
                                return int(os.path.splitext(os.path.basename(p))[0])
                            except Exception:
                                return 0
                        selected_temp = sorted(set(results_acc), key=idx_of)

                print(f"Batch {batch_idx}: selected {len(selected_temp)} / {len(saved_paths)} frames.")
                info["selected_temp"] = selected_temp
                info["created_temp_folder_by_mode"] = created_temp_folder_by_mode

                if selected_temp:
                    if created_temp_folder_by_mode:
                        # Mode prepared the frames already in tmp folder; submit that folder to ffmpeg via streaming
                        sel_list_sorted = sorted(selected_temp)
                        future_sel = ffmpeg_executor.submit(
                            submit_segment_stream_or_fallback,
                            sel_list_sorted, os.path.join(OUTPUT_DIR, f"temp_selected_{batch_idx:03d}"),
                            info["seg_sel"], chosen_encoder, chosen_mode, fps
                        )
                        group_info[batch_idx]["future_sel"] = future_sel
                        segment_paths.append(info["seg_sel"])
                    else:
                        # selected_temp is a list of saved_paths (in temp_batch) - find originals from mapping where possible
                        selected_originals = []
                        for sp in selected_temp:
                            orig = mapping.get(sp)
                            if orig:
                                selected_originals.append(orig)
                            else:
                                selected_originals.append(sp)
                        # submit streaming from original selected images (with fallback)
                        future_sel = ffmpeg_executor.submit(
                            submit_segment_stream_or_fallback,
                            selected_originals, os.path.join(OUTPUT_DIR, f"temp_selected_{batch_idx:03d}"),
                            info["seg_sel"], chosen_encoder, chosen_mode, fps
                        )
                        group_info[batch_idx]["future_sel"] = future_sel
                        segment_paths.append(info["seg_sel"])
                else:
                    vprint(f"Batch {batch_idx}: no changes detected.")

            # update prev_last_original for next batch
            last_temp = saved_paths[-1]
            prev_last_original = mapping.get(last_temp, info["batch_src"][-1])

        # Wait for group's ffmpeg futures to finish (either all or sel)
        futures = []
        for batch_idx in group:
            info = group_info[batch_idx]
            if "future_all" in info:
                futures.append((batch_idx, "all", info["future_all"], info.get("seg_all"), info.get("temp_batch")))
            if "future_sel" in info:
                futures.append((batch_idx, "sel", info["future_sel"], info.get("seg_sel"), os.path.join(OUTPUT_DIR, f"temp_selected_{batch_idx:03d}")))
        if futures:
            pbar = _tqdm(total=len(futures), desc=f"Encoding group {group_start}-{group_start+len(group)-1}", unit="seg") if TQDM_AVAILABLE else None
            for batch_idx, kind, fut, segpath, temppath in futures:
                try:
                    ok = fut.result()
                    if not ok:
                        print(f"Warning: ffmpeg task for {segpath} reported failure.")
                except Exception as e:
                    print(f"ffmpeg task for {segpath} raised exception: {e}")
                # cleanup any temp folder created specifically for selected fallback (we used unique folder names)
                if temppath and os.path.isdir(temppath):
                    try:
                        shutil.rmtree(temppath)
                        _unregister_temp_path(temppath)
                    except Exception:
                        vprint("Failed to remove temp folder:", temppath)
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()

        # update global overall_frames_pbar by number of source frames in this group
        group_frame_count = sum(len(group_info[batch_idx]["batch_src"]) for batch_idx in group if group_info.get(batch_idx))
        if TQDM_AVAILABLE and overall_frames_pbar is not None:
            overall_frames_pbar.update(group_frame_count)

        # Always attempt to clean temp folders created by create_links_for_batch or by modes for this group
        for t in list(set(temp_to_cleanup_for_group)):
            try:
                if os.path.isdir(t):
                    shutil.rmtree(t)
                    _unregister_temp_path(t)
            except Exception as e:
                vprint("Failed to cleanup temp folder:", t, e)

    ffmpeg_executor.shutdown(wait=True)

    # Merge per-batch segments into single final video
    print("\nMerging segments into final video...")
    final_name = os.path.join(OUTPUT_DIR, f"timelapse_{mode_id}.mp4")
    if not concat_segments_ffmpeg(segment_paths, final_name):
        print("Failed to concatenate segments. Segments are left in", OUTPUT_DIR)
    else:
        print(f"Created: {final_name}")
        # cleanup per-batch segments only after successful merge
        for p in segment_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    # Do NOT save selected_frames.txt as requested

    # close overall frames pbar if it's a real tqdm
    if TQDM_AVAILABLE and overall_frames_pbar is not None:
        overall_frames_pbar.close()

    total_time = time.time() - overall_start
    print("\nAll done.")
    print(f"Time elapsed: {total_time:.2f} s")
    print(f"Output video: {final_name}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        cleanup_all_temp_paths()
        sys.exit(1)
