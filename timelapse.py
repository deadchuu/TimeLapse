#!/usr/bin/env python3
"""
timelapse.py - entry point wrapper with CLI, WorkerController integration and throttle.

Usage example:
  python timelapse.py --source /path/to/pngs --fps 30 --max-cpu 90 --subtasks 1 --no-fallback --log-file timelapse.log

Requires the modular files in the same folder:
 utils.py, io_utils.py, compare.py, encoder.py, scheduler.py, config.py, modes/
"""
import os
import sys
import time
import argparse
import shutil
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor

# try to import project modules (these should exist in same folder)
try:
    from engine.utils import setup_logging, get_tqdm, get_logger
    from io_utils import ensure_dir, basename, list_png_files, make_unique_tempdir, create_links_for_batch, register_temp_path, unregister_temp_path
    from compare import batch_compare_pairs, detect_best_backend
    from encoder import submit_segment_stream_or_fallback, concat_segments_ffmpeg
    from scheduler import DispatcherThrottle
    import config
except Exception as e:
    print("Failed to import project modules. Make sure utils.py, io_utils.py, compare.py, encoder.py, scheduler.py, config.py are present.")
    print("Error:", e)
    sys.exit(1)

logger = get_logger()

# Try to import psutil (if not present, fallback to basic dispatcher throttle)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available — dynamic pause/resume will be limited. Install with: pip install psutil")

# Worker process function for selection; uses compare.batch_compare_pairs on single pair for simplicity
def _selection_worker(task_q, result_q, pause_event, stop_event, backend_hint=None, threshold_pixels=0):
    """
    Worker process: reads (prev_path, curr_path) pairs from task_q.
    If the pair is considered changed, puts curr_path to result_q.
    Worker checks pause_event periodically and stops when stop_event is set.
    """
    # Import inside process to get fresh modules (and avoid pickling heavy objects)
    from compare import batch_compare_pairs
    while not stop_event.is_set():
        if pause_event.is_set():
            # cooperative pause
            time.sleep(0.1)
            continue
        try:
            pair = task_q.get(timeout=0.5)
        except Exception:
            continue
        if pair is None:
            # sentinel
            break
        prev, curr = pair
        try:
            # call batch_compare_pairs with single pair - backend autodetect unless hint provided
            changed = batch_compare_pairs([(prev, curr)], backend=backend_hint, threshold_pixels=threshold_pixels)
            if changed:
                # batch_compare_pairs returns list of curr_paths that differ
                result_q.put(curr)
        except Exception as e:
            # on failure, treat as changed to be safe
            result_q.put(curr)

def filter_changed_frames_controlled(saved_paths, mapping=None, region=None, prev_last_original=None,
                                     max_workers=4, pause_event=None, stop_event=None,
                                     backend_hint=None, threshold_pixels=0):
    """
    Controlled selection: spawns worker processes that can be paused via pause_event.
    Returns selected list of saved_paths (paths from saved_paths list) that differ vs previous frame.
    - saved_paths: list of temp paths (in temp batch folder)
    - mapping: dict temp_path -> original path (optional)
    - prev_last_original: path to previous original image (for first comparison)
    """
    selected = []

    if not saved_paths:
        return selected

    # Build list of pairs
    pairs = []
    if prev_last_original:
        pairs.append((prev_last_original, saved_paths[0]))
    for i in range(1, len(saved_paths)):
        pairs.append((saved_paths[i-1], saved_paths[i]))

    if not pairs:
        return selected

    # Multiprocessing primitives
    task_q = multiprocessing.Queue(maxsize=1024)
    result_q = multiprocessing.Queue()
    mgr = multiprocessing.Manager()
    if pause_event is None:
        pause_event = mgr.Event()
    if stop_event is None:
        stop_event = mgr.Event()

    workers = []
    nworkers = max(1, min(max_workers, multiprocessing.cpu_count()))
    for i in range(nworkers):
        p = multiprocessing.Process(target=_selection_worker, args=(task_q, result_q, pause_event, stop_event, backend_hint, threshold_pixels))
        p.daemon = True
        p.start()
        workers.append(p)

    # push tasks
    for pr in pairs:
        # wait until there's room in queue (avoids flooding)
        while True:
            try:
                task_q.put(pr, timeout=0.5)
                break
            except Exception:
                if stop_event.is_set():
                    break
                time.sleep(0.05)

    # send sentinel to stop workers after tasks are done
    for _ in workers:
        try:
            task_q.put(None, timeout=1.0)
        except Exception:
            pass

    # collect results: because result_q only contains curr paths, gather until all tasks processed
    # We'll count tasks processed by checking workers alive and timeout
    remaining = len(pairs)
    start = time.time()
    timeout_seconds = max(5, 60)  # total wait guard (fallback)
    received_set = set()
    while remaining > 0:
        try:
            item = result_q.get(timeout=1.0)
            if item:
                received_set.add(item)
            remaining -= 1
        except Exception:
            # no result this second - check if workers alive or we hit overall timeout
            alive = any(p.is_alive() for p in workers)
            if not alive:
                # no workers left - drain queue
                break
            if (time.time() - start) > (timeout_seconds + len(pairs)*0.01):
                break
            continue

    # ask workers to stop
    stop_event.set()
    # join workers
    for p in workers:
        try:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass

    # convert received_set (which contains temp paths or original depending on backend) into list in order
    # prefer original mapping when possible
    def idx_of(p):
        try:
            name = os.path.basename(p)
            return int(os.path.splitext(name)[0])
        except Exception:
            return 0

    # map potentially returned temp paths to originals via mapping if provided
    selected_list = []
    for p in sorted(received_set, key=idx_of):
        if mapping and p in mapping:
            selected_list.append(mapping[p])
        else:
            # might be original already, or temp not mapped
            selected_list.append(p)

    return selected_list


def launch_cpu_monitor(pause_event, stop_event, max_cpu_percent=90.0, check_interval=1.0):
    """
    Background thread: monitors CPU usage (psutil) and sets/clears pause_event accordingly.
    Uses a multiprocessing-compatible Event (passed from main).
    """
    if not PSUTIL_AVAILABLE:
        logger.info("psutil not available — CPU monitor disabled.")
        return None

    def _monitor():
        logger.info("CPU monitor started (max_cpu=%s%%).", max_cpu_percent)
        while not stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=check_interval)
            except Exception:
                cpu = 0.0
            if cpu >= max_cpu_percent:
                if not pause_event.is_set():
                    pause_event.set()
                    logger.info("CPU %.1f%% >= %.1f%% -> pausing selection workers.", cpu, max_cpu_percent)
            else:
                if pause_event.is_set():
                    pause_event.clear()
                    logger.info("CPU %.1f%% < %.1f%% -> resuming selection workers.", cpu, max_cpu_percent)
        logger.info("CPU monitor stopped.")
    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return t

def parse_args():
    parser = argparse.ArgumentParser(description="Timelapse generator with GPU/CPU selection and controlled resource usage.")
    parser.add_argument("--source", "-s", required=True, help="Path to folder with PNG files.")
    parser.add_argument("--fps", type=float, default=getattr(config, "DEFAULT_FPS", 30.0), help="Output FPS (default from config).")
    parser.add_argument("--mode-index", type=int, default=0, help="Index of mode from modes/ folder (default 0).")
    parser.add_argument("--max-cpu", type=float, default=90.0, help="Maximum allowed CPU percent before pausing selection workers (default 90).")
    parser.add_argument("--subtasks", type=int, default=getattr(config, "SUBTASKS_PER_BATCH", 1), help="Split each batch into this many subtasks for selection (default from config).")
    parser.add_argument("--no-fallback", action="store_true", help="Disable pattern-based fallback for ffmpeg streaming (use streaming-only).")
    parser.add_argument("--log-file", default=getattr(config, "LOG_FILE", "timelapse.log"), help="Log file path (default from config).")
    parser.add_argument("--max-ffmpeg", type=int, default=getattr(config, "MAX_FFMPEG_PROCS", 2), help="Max parallel ffmpeg processes.")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of worker processes for selection (default cpu_count-1).")
    return parser.parse_args()

def main():
    args = parse_args()
    # (re)configure logging
    setup_logging(args.log_file)
    logger.info("timelapse.py started. Source=%s FPS=%s", args.source, args.fps)

    # load modes
    modes_dir = os.path.join(os.path.dirname(__file__), "modes")
    modes = []
    if os.path.isdir(modes_dir):
        import importlib.util
        for fname in sorted(os.listdir(modes_dir)):
            if not fname.lower().endswith(".py") or fname.startswith("_"):
                continue
            path = os.path.join(modes_dir, fname)
            try:
                spec = importlib.util.spec_from_file_location(f"modes.{fname}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "get_mode"):
                    m = mod.get_mode()
                    modes.append(m)
            except Exception as e:
                logger.exception("Failed to load mode %s: %s", fname, e)
    if not modes:
        logger.warning("No modes loaded from %s", modes_dir)

    # choose mode
    chosen_mode = None
    if modes:
        idx = args.mode_index
        if 0 <= idx < len(modes):
            chosen_mode = modes[idx]
        else:
            chosen_mode = modes[0]
    logger.info("Chosen mode: %s", chosen_mode["id"] if chosen_mode else "default")

    # Validate source
    source_folder = args.source
    if not os.path.isdir(source_folder):
        print("Source folder does not exist:", source_folder)
        return

    fps = float(args.fps)
    files = list_png_files(source_folder)
    if not files:
        print("No PNG files found in source folder.")
        return

    # Create output dir
    cwd = os.path.abspath(os.getcwd())
    output_base = getattr(config, "BASE_OUTPUT_DIR", os.path.join(cwd, "timelaps"))
    ensure_dir(output_base)
    source_name = basename(source_folder)
    OUTPUT_DIR = os.path.join(output_base, source_name)
    ensure_dir(OUTPUT_DIR)
    logger.info("Output directory: %s", OUTPUT_DIR)

    # Prepare progress bar - console only
    total_frames = len(files)
    overall_pbar = get_tqdm(total=total_frames, desc="Overall frames", unit="frame")

    # prepare threadpool for ffmpeg tasks
    ffmpeg_executor = ThreadPoolExecutor(max_workers=args.max_ffmpeg)

    # prepare throttle and CPU monitor + worker control events
    throttle = DispatcherThrottle(max_cpu_percent=args.max_cpu)
    # Use multiprocessing.Event for cross-process signaling
    mgr = multiprocessing.Manager()
    pause_event = mgr.Event()
    stop_event = mgr.Event()
    # launch monitor if psutil available
    monitor_thread = launch_cpu_monitor(pause_event, stop_event, max_cpu_percent=args.max_cpu, check_interval=1.0)

    prev_last_original = None
    segment_paths = []
    total_batches = (len(files) + config.BATCH_SIZE - 1) // config.BATCH_SIZE

    for batch_idx in range(total_batches):
        s = batch_idx * config.BATCH_SIZE
        e = min(s + config.BATCH_SIZE, len(files))
        batch_src = files[s:e]
        logger.info("Processing batch %d (%d..%d)", batch_idx, s, e-1)

        # create unique batch temp folder
        temp_batch = make_unique_tempdir(prefix=f"batch_{batch_idx:03d}_", base_dir=OUTPUT_DIR)
        # create numbered links
        saved_paths, mapping = create_links_for_batch(batch_src, temp_batch)

        seg_all = os.path.join(OUTPUT_DIR, f"segment_{chosen_mode['id'] if chosen_mode else 'default'}_{batch_idx:03d}.mp4")
        seg_sel = os.path.join(OUTPUT_DIR, f"segment_selected_{chosen_mode['id'] if chosen_mode else 'default'}_{batch_idx:03d}.mp4")

        # If selection disabled, just submit full encode
        do_selected = bool(chosen_mode and chosen_mode.get("enable_selection"))
        if not do_selected:
            throttle.wait_for_capacity()
            future_all = ffmpeg_executor.submit(submit_segment_stream_or_fallback, batch_src, temp_batch, seg_all, None, fps, not args.no_fallback)
            segment_paths.append(seg_all)
        else:
            # call mode's select_frames if provided (pass fps)
            sel_func = chosen_mode.get("select_frames") if chosen_mode else None
            selected_list = None
            created_temp_folder_by_mode = None
            if sel_func:
                try:
                    # try kw-arg style
                    selected_result = sel_func(saved_paths=saved_paths, mapping=mapping, region=None, prev_last_original=prev_last_original, output_dir=OUTPUT_DIR, batch_idx=batch_idx, fps=fps)
                except TypeError:
                    try:
                        selected_result = sel_func(saved_paths, mapping, None, prev_last_original)
                    except Exception as e:
                        logger.exception("Mode select_frames failed: %s", e)
                        selected_result = None
                if isinstance(selected_result, dict):
                    created_temp_folder_by_mode = selected_result.get("temp_sel_folder")
                    selected_list = selected_result.get("selected_paths", [])
                elif isinstance(selected_result, (list,tuple)):
                    selected_list = list(selected_result)
                else:
                    selected_list = None

            if selected_list is None:
                # fallback: use controlled selection workers (this uses pause_event to pause/resume)
                # We will compute selected frames from saved_paths (which are temp sequential files)
                logger.info("Using controlled selection for batch %d (workers=%d)", batch_idx, args.workers)
                selected_list = filter_changed_frames_controlled(saved_paths, mapping=mapping, region=None, prev_last_original=prev_last_original,
                                                               max_workers=args.workers, pause_event=pause_event, stop_event=stop_event,
                                                               backend_hint=None, threshold_pixels=0)
            else:
                logger.info("Mode provided %d selected frames for batch %d", len(selected_list), batch_idx)

            if selected_list:
                # submit selected encode
                throttle.wait_for_capacity()
                future_sel = ffmpeg_executor.submit(submit_segment_stream_or_fallback, selected_list, os.path.join(OUTPUT_DIR, f"temp_selected_{batch_idx:03d}"), seg_sel, None, fps, not args.no_fallback)
                segment_paths.append(seg_sel)
            else:
                logger.info("No selected frames for batch %d", batch_idx)

        # update prev_last_original
        if saved_paths:
            prev_last_original = mapping.get(saved_paths[-1], batch_src[-1])

        # update progress bar
        overall_pbar.update(len(batch_src))

        # cleanup temp batch links to free space
        try:
            if os.path.isdir(temp_batch):
                shutil.rmtree(temp_batch)
                unregister_temp_path(temp_batch)
        except Exception as e:
            logger.exception("Failed to remove temp batch %s: %s", temp_batch, e)

    # wait for ffmpeg tasks
    ffmpeg_executor.shutdown(wait=True)
    overall_pbar.close()

    # stop monitor
    stop_event.set()
    time.sleep(0.1)

    # merge segments
    final_out = os.path.join(OUTPUT_DIR, f"timelapse_{chosen_mode['id'] if chosen_mode else 'default'}.mp4")
    if concat_segments_ffmpeg(segment_paths, final_out):
        logger.info("Created final movie: %s", final_out)
        # remove intermediate segments
        for p in segment_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    else:
        logger.error("Failed to merge segments. Segments kept in %s", OUTPUT_DIR)

    logger.info("All done. Output: %s", final_out)

if __name__ == "__main__":
    main()
