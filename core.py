# core.py
import os
import time
import math
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from .utils import get_logger, setup_logging, get_tqdm
from .io_utils import ensure_dir, basename, list_png_files, make_unique_tempdir, create_links_for_batch, register_temp_path, unregister_temp_path
from .compare import batch_compare_pairs, detect_best_backend
from .encoder import submit_segment_stream_or_fallback, concat_segments_ffmpeg
from .scheduler import DispatcherThrottle
from . import config

logger = get_logger()

def main():
    # configure paths and logging
    cwd = os.path.abspath(os.getcwd())
    if config.BASE_OUTPUT_DIR is None:
        output_base = os.path.join(cwd, "timelaps")
    else:
        output_base = config.BASE_OUTPUT_DIR
    ensure_dir(output_base)
    setup_logging(config.LOG_FILE)
    logger.info("Starting timelapse pipeline. Output base: %s", output_base)

    # load modes folder dynamically like before (kept simple here)
    from importlib import util
    modes_folder = os.path.join(os.path.dirname(__file__), "modes")
    modes = []
    if os.path.isdir(modes_folder):
        for fname in sorted(os.listdir(modes_folder)):
            if not fname.lower().endswith(".py") or fname.startswith("_"):
                continue
            path = os.path.join(modes_folder, fname)
            try:
                spec = util.spec_from_file_location(f"modes.{fname}", path)
                mod = util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "get_mode"):
                    m = mod.get_mode()
                    modes.append(m)
            except Exception as e:
                logger.exception("Failed to load mode %s: %s", fname, e)

    # print available modes to logger
    logger.info("Available modes: %s", [m.get("id") for m in modes])

    # choose mode via input (console only for short prompt)
    print("=== AVAILABLE MODES ===")
    if not modes:
        print("No modes found. Put modes modules into modes/ folder.")
        chosen_mode = None
    else:
        for i,m in enumerate(modes):
            print(f"[{i}] {m['name']} - {m.get('desc','')}")
        sel = input("Select mode index [default 0]: ").strip()
        try:
            idx = int(sel) if sel != "" else 0
            if 0 <= idx < len(modes):
                chosen_mode = modes[idx]
            else:
                chosen_mode = modes[0]
        except Exception:
            chosen_mode = modes[0]
    print("=======================")

    source_folder = input("Enter path to folder with PNG files: ").strip()
    if not os.path.isdir(source_folder):
        print("Folder does not exist.")
        return

    fps_in = input(f"Enter desired FPS [default {config.DEFAULT_FPS}]: ").strip()
    try:
        fps = float(fps_in) if fps_in else config.DEFAULT_FPS
    except Exception:
        fps = config.DEFAULT_FPS

    # encoder selection prompt simplified
    use_hw = False
    encoder_choice = None
    # (we keep encoder decision simple; core doesn't detect hw encoder in this module)
    hwq = input("Try hardware acceleration (yes/no) [default no]: ").strip().lower()
    if hwq in ("y","yes"):
        use_hw = True
        encoder_choice = None  # let encoder.py default to libx264; advanced: pass detected encoder

    files = list_png_files(source_folder)
    if not files:
        print("No PNG files found.")
        return

    total_frames = len(files)
    overall_pbar = get_tqdm(total=total_frames, desc="Overall frames", unit="frame")

    source_name = basename(source_folder)
    OUTPUT_DIR = os.path.join(output_base, source_name)
    ensure_dir(OUTPUT_DIR)

    do_selected = bool(chosen_mode and chosen_mode.get("enable_selection"))
    mode_id = chosen_mode["id"] if chosen_mode else "default"
    logger.info("Starting processing: %d frames, mode=%s", total_frames, mode_id)

    total_batches = math.ceil(len(files)/config.BATCH_SIZE)
    prev_last_original = None
    segment_paths = []

    ffmpeg_executor = ThreadPoolExecutor(max_workers=config.MAX_FFMPEG_PROCS)
    batch_iter = range(total_batches)

    # prepare throttle
    throttle = DispatcherThrottle(max_cpu_percent=90.0)

    for batch_idx in batch_iter:
        s = batch_idx * config.BATCH_SIZE
        e = min(s + config.BATCH_SIZE, len(files))
        batch_src = files[s:e]
        logger.info("Processing batch %d (%d..%d)", batch_idx, s, e-1)
        temp_batch = make_unique_tempdir(prefix=f"batch_{batch_idx:03d}_", base_dir=OUTPUT_DIR)
        # create numbered links
        saved_paths, mapping = create_links_for_batch(batch_src, temp_batch)

        seg_all = os.path.join(OUTPUT_DIR, f"segment_{mode_id}_{batch_idx:03d}.mp4")
        seg_sel = os.path.join(OUTPUT_DIR, f"segment_selected_{mode_id}_{batch_idx:03d}.mp4")

        if not do_selected:
            # submit full encode
            throttle.wait_for_capacity()
            future_all = ffmpeg_executor.submit(submit_segment_stream_or_fallback, batch_src, temp_batch, seg_all, encoder_choice, fps)
            segment_paths.append(seg_all)
        else:
            # selection: use mode's select_frames if present; pass fps
            sel_func = None
            if chosen_mode and callable(chosen_mode.get("select_frames")):
                sel_func = chosen_mode["select_frames"]
            if sel_func:
                try:
                    selected_result = sel_func(saved_paths=saved_paths, mapping=mapping, region=None, prev_last_original=prev_last_original, output_dir=OUTPUT_DIR, batch_idx=batch_idx, fps=fps)
                except TypeError:
                    # older signature
                    selected_result = sel_func(saved_paths, mapping, None, prev_last_original)
                if isinstance(selected_result, dict):
                    temp_sel = selected_result.get("temp_sel_folder")
                    sel_paths = selected_result.get("selected_paths", [])
                    if temp_sel and sel_paths:
                        throttle.wait_for_capacity()
                        future_sel = ffmpeg_executor.submit(submit_segment_stream_or_fallback, sel_paths, temp_sel, seg_sel, encoder_choice, fps)
                        segment_paths.append(seg_sel)
                    else:
                        # fallback to generic compare (batch_compare_pairs)
                        pairs = []
                        if prev_last_original:
                            pairs.append((prev_last_original, saved_paths[0]))
                        for i in range(1,len(saved_paths)):
                            pairs.append((saved_paths[i-1], saved_paths[i]))
                        changed = batch_compare_pairs(pairs)
                        # build list of original paths for changed frames
                        sel_originals = []
                        for cp in changed:
                            orig = mapping.get(cp)
                            sel_originals.append(orig or cp)
                        if sel_originals:
                            throttle.wait_for_capacity()
                            future_sel = ffmpeg_executor.submit(submit_segment_stream_or_fallback, sel_originals, os.path.join(OUTPUT_DIR, f"temp_selected_{batch_idx:03d}"), seg_sel, encoder_choice, fps)
                            segment_paths.append(seg_sel)
                elif isinstance(selected_result, (list,tuple)):
                    sel_originals = []
                    for sp in selected_result:
                        orig = mapping.get(sp)
                        sel_originals.append(orig or sp)
                    if sel_originals:
                        throttle.wait_for_capacity()
                        future_sel = ffmpeg_executor.submit(submit_segment_stream_or_fallback, sel_originals, os.path.join(OUTPUT_DIR, f"temp_selected_{batch_idx:03d}"), seg_sel, encoder_choice, fps)
                        segment_paths.append(seg_sel)
                else:
                    # no frames selected
                    pass
            else:
                # mode doesn't provide selection - fallback to generic method
                pass

        # update prev_last_original
        if saved_paths:
            last_temp = saved_paths[-1]
            prev_last_original = mapping.get(last_temp, batch_src[-1])

        # update overall progress (we mark batch frames as processed)
        overall_pbar.update(len(batch_src))

        # cleanup temp batch (links) immediately to free space
        try:
            if os.path.isdir(temp_batch):
                shutil.rmtree(temp_batch)
                unregister_temp_path(temp_batch)
        except Exception as e:
            logger.exception("Failed remove temp batch %s: %s", temp_batch, e)

    ffmpeg_executor.shutdown(wait=True)
    overall_pbar.close()

    # merge segments
    final_out = os.path.join(OUTPUT_DIR, f"timelapse_{mode_id}.mp4")
    if concat_segments_ffmpeg(segment_paths, final_out):
        logger.info("Created final movie: %s", final_out)
        # remove segments after success
        for p in segment_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    else:
        logger.error("Failed to merge segments. Segments are kept in %s", OUTPUT_DIR)

    logger.info("All done.")

if __name__ == "__main__":
    main()
