# engine/core.py
"""
Main orchestration. Provides run_interactive() which the top-level Tlapse.py calls.
"""
import os
import math
import time
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from engine.utils import setup_logging, get_tqdm, get_logger
from engine.io_utils import ensure_dir, basename, list_png_files, make_unique_tempdir, create_links_for_batch, register_temp_path, unregister_temp_path
from engine.compare import batch_compare_pairs, detect_best_backend
from engine.encoder import submit_segment_stream_or_fallback, concat_segments_ffmpeg
from engine.scheduler import DispatcherThrottle
import engine.config as config

logger = get_logger()

def run_interactive():
    # configure logging file
    setup_logging(config.LOG_FILE)
    logger.info("Starting interactive timelapse run")

    # load modes from modes/ folder
    modes_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modes")
    modes = []
    if os.path.isdir(modes_folder):
        import importlib.util
        for fname in sorted(os.listdir(modes_folder)):
            if not fname.lower().endswith(".py") or fname.startswith("_"):
                continue
            path = os.path.join(modes_folder, fname)
            try:
                spec = importlib.util.spec_from_file_location(f"modes.{fname}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "get_mode"):
                    m = mod.get_mode()
                    modes.append(m)
            except Exception as e:
                logger.exception("Failed to load mode %s: %s", fname, e)
    # print modes to console (user wants interactive selection)
    print("=== AVAILABLE MODES ===")
    if not modes:
        print("No modes found in 'modes' folder. Place your mode files there.")
    else:
        for i,m in enumerate(modes):
            print(f"[{i}] {m['name']} - {m.get('desc','')}")
    print("=======================")

    # choose mode
    chosen_mode = None
    if modes:
        mode_sel = input("Select mode index [default 0]: ").strip()
        try:
            idx = int(mode_sel) if mode_sel != "" else 0
            if 0 <= idx < len(modes):
                chosen_mode = modes[idx]
            else:
                print("Invalid mode index, using default 0.")
                chosen_mode = modes[0]
        except Exception:
            print("Invalid input, using default mode 0.")
            chosen_mode = modes[0]

    # Ask source folder interactively (as you requested)
    source_folder = input("Enter path to folder with PNG files: ").strip()
    if not os.path.isdir(source_folder):
        print("Specified folder does not exist.")
        return

    fps_in = input(f"Enter desired FPS (e.g. 30) [default {config.DEFAULT_FPS}]: ").strip()
    try:
        fps = float(fps_in) if fps_in else config.DEFAULT_FPS
    except Exception:
        fps = config.DEFAULT_FPS

    # hardware acceleration question (simple)
    use_hw = False
    resp = input("Try hardware acceleration if available? (y/N): ").strip().lower()
    use_hw = resp in ("y", "yes")
    if use_hw:
        logger.info("User requested hardware acceleration if available.")
    else:
        logger.info("User disabled hardware acceleration.")

    # ensure ffmpeg
    from shutil import which
    if which("ffmpeg") is None:
        print("ffmpeg not found in PATH. Please install ffmpeg and add to PATH.")
        return

    files = list_png_files(source_folder)
    if not files:
        print("No PNG files found in source folder.")
        return

    # progress bar setup (console only)
    total_frames = len(files)
    overall_pbar = get_tqdm(total=total_frames, desc="Overall frames", unit="frame")

    # output dir
    cwd = os.path.abspath(os.getcwd())
    output_base = config.BASE_OUTPUT_DIR if config.BASE_OUTPUT_DIR else os.path.join(cwd, "timelapse")
    ensure_dir(output_base)
    source_name = basename(source_folder)
    OUTPUT_DIR = os.path.join(output_base, source_name)
    ensure_dir(OUTPUT_DIR)

    do_selected = bool(chosen_mode and chosen_mode.get("enable_selection"))
    mode_id = chosen_mode["id"] if chosen_mode else "default"
    logger.info("Processing %d frames. Output dir: %s Mode: %s", total_frames, OUTPUT_DIR, mode_id)

    total_batches = math.ceil(len(files)/config.BATCH_SIZE)
    prev_last_original = None
    segment_paths = []

    ffmpeg_executor = ThreadPoolExecutor(max_workers=config.MAX_FFMPEG_PROCS)
    throttle = DispatcherThrottle(max_cpu_percent=90.0)

    for batch_idx in range(total_batches):
        s = batch_idx * config.BATCH_SIZE
        e = min(s + config.BATCH_SIZE, len(files))
        batch_src = files[s:e]
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({s}..{e-1})")
        logger.info("Processing batch %d (%d..%d)", batch_idx, s, e-1)

        temp_batch = make_unique_tempdir(prefix=f"batch_{batch_idx:03d}_", base_dir=OUTPUT_DIR)
        saved_paths, mapping = create_links_for_batch(batch_src, temp_batch)

        seg_name = os.path.join(OUTPUT_DIR, f"segment_{mode_id}_{batch_idx:03d}.mp4")
        sel_name = os.path.join(OUTPUT_DIR, f"segment_selected_{mode_id}_{batch_idx:03d}.mp4")

        if not do_selected:
            throttle.wait_for_capacity()
            future_all = ffmpeg_executor.submit(submit_segment_stream_or_fallback, batch_src, temp_batch, seg_name, None, fps, True)
            segment_paths.append(seg_name)
        else:
            # selection path: if mode has select_frames -> call with fps
            selected_temp = None
            created_temp_folder_by_mode = None
            if chosen_mode and callable(chosen_mode.get("select_frames")):
                try:
                    selected_result = chosen_mode["select_frames"](
                        saved_paths=saved_paths,
                        mapping=mapping,
                        region=None,
                        prev_last_original=prev_last_original,
                        output_dir=OUTPUT_DIR,
                        batch_idx=batch_idx,
                        fps=fps
                    )
                except TypeError:
                    try:
                        selected_result = chosen_mode["select_frames"](saved_paths, mapping, None, prev_last_original, OUTPUT_DIR, batch_idx, fps)
                    except Exception as ex:
                        logger.exception("Mode select_frames raised: %s", ex)
                        selected_result = None
                if isinstance(selected_result, dict):
                    created_temp_folder_by_mode = selected_result.get("temp_sel_folder")
                    selected_temp = selected_result.get("selected_paths", [])
                elif isinstance(selected_result, (list,tuple)):
                    selected_temp = list(selected_result)
                else:
                    selected_temp = None

            # fallback generic selection if mode did not produce selected frames
            if not selected_temp:
                # use generic compare (batch_compare_pairs)
                pairs = []
                if prev_last_original:
                    pairs.append((prev_last_original, saved_paths[0]))
                for i in range(1, len(saved_paths)):
                    pairs.append((saved_paths[i-1], saved_paths[i]))
                if pairs:
                    backend = detect_best_backend()
                    changed = batch_compare_pairs(pairs, backend=backend)
                else:
                    changed = []
                # map changed entries back to original paths
                selected_originals = []
                for cp in changed:
                    orig = mapping.get(cp)
                    selected_originals.append(orig or cp)
                if selected_originals:
                    throttle.wait_for_capacity()
                    future_sel = ffmpeg_executor.submit(submit_segment_stream_or_fallback, selected_originals, os.path.join(OUTPUT_DIR, f"temp_selected_{batch_idx:03d}"), sel_name, None, fps, True)
                    segment_paths.append(sel_name)
                else:
                    logger.info("No changes detected in batch %d", batch_idx)
            else:
                # selected_temp contains either temp folder produced by mode or list of paths
                if created_temp_folder_by_mode:
                    sel_list_sorted = sorted(selected_temp)
                    throttle.wait_for_capacity()
                    future_sel = ffmpeg_executor.submit(submit_segment_stream_or_fallback, sel_list_sorted, os.path.join(OUTPUT_DIR, f"temp_selected_{batch_idx:03d}"), sel_name, None, fps, True)
                    segment_paths.append(sel_name)
                else:
                    # map back to original if possible
                    selected_originals = []
                    for sp in selected_temp:
                        selected_originals.append(mapping.get(sp, sp))
                    throttle.wait_for_capacity()
                    future_sel = ffmpeg_executor.submit(submit_segment_stream_or_fallback, selected_originals, os.path.join(OUTPUT_DIR, f"temp_selected_{batch_idx:03d}"), sel_name, None, fps, True)
                    segment_paths.append(sel_name)

        # update prev_last_original
        if saved_paths:
            prev_last_original = mapping.get(saved_paths[-1], batch_src[-1])

        overall_pbar.update(len(batch_src))

        # cleanup created temp links
        try:
            if os.path.isdir(temp_batch):
                shutil.rmtree(temp_batch)
                unregister_temp_path(temp_batch)
        except Exception as e:
            logger.exception("Failed to cleanup temp batch %s: %s", temp_batch, e)

    ffmpeg_executor.shutdown(wait=True)
    overall_pbar.close()

    # merge segments
    print("\nMerging segments into final video...")
    final_name = os.path.join(OUTPUT_DIR, f"timelapse_{mode_id}.mp4")
    if concat_segments_ffmpeg(segment_paths, final_name):
        print("Created:", final_name)
        logger.info("Created final video: %s", final_name)
        # remove per-batch segments
        for p in segment_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    else:
        logger.error("Failed to concatenate segments. Segments left in %s", OUTPUT_DIR)
    print("Done.")
