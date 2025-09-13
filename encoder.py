# encoder.py
import os
import subprocess
import threading
import shutil
from .utils import get_logger, vprint
from .io_utils import register_temp_path, unregister_temp_path, make_unique_tempdir
logger = get_logger()

FFMPEG_CRF = 18
FFMPEG_PRESET = "medium"
THREAD_QUEUE_SIZE = 1024

def _writer_thread_for_proc(proc, image_paths, stop_event):
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
                logger.debug("Writer thread error: %s", e)
                continue
        try:
            proc.stdin.close()
        except Exception:
            pass
    except Exception as e:
        logger.exception("Writer thread fatal: %s", e)
        try:
            proc.stdin.close()
        except Exception:
            pass

def make_segment_from_images_ffmpeg_stream(image_paths, fps, out_segment_path, encoder_hint=None, timeout=None):
    if not image_paths:
        logger.debug("No image paths to stream.")
        return False
    first = image_paths[0]
    # get image size via PIL
    from PIL import Image
    try:
        with Image.open(first) as im:
            w,h = im.width, im.height
    except Exception as e:
        logger.debug("Cannot determine image size: %s", e)
        return False

    encoder = encoder_hint or "libx264"
    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=white:s={w}x{h}:r={int(fps)}",
        "-framerate", str(int(fps)),
        "-f", "image2pipe", "-vcodec", "png", "-i", "-",
    ]
    if encoder == "libx264":
        cmd = base + [
            "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", str(FFMPEG_CRF),
            "-pix_fmt", "yuv420p", out_segment_path
        ]
    else:
        # for non-libx264 we fallback to libx264 for safety in this module,
        # leaving hardware specific tuning to caller if desired
        cmd = base + [
            "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", str(FFMPEG_CRF),
            "-pix_fmt", "yuv420p", out_segment_path
        ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except Exception as e:
        logger.exception("Failed to start ffmpeg: %s", e)
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
        return ret == 0
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg stream timed out; killing.")
        stop_event.set()
        try:
            proc.kill()
        except Exception:
            pass
        return False
    except Exception as e:
        logger.exception("Streaming error: %s", e)
        stop_event.set()
        try:
            proc.kill()
        except Exception:
            pass
        return False

def make_segment_from_images_ffmpeg(temp_img_folder, fps, out_segment_path, encoder_hint=None):
    # pattern-based method
    # find first image
    first_img = os.path.join(temp_img_folder, "000000.png")
    if not os.path.exists(first_img):
        files = sorted([os.path.join(temp_img_folder,f) for f in os.listdir(temp_img_folder) if f.lower().endswith(".png")])
        if not files:
            logger.debug("No images for pattern encoding.")
            return False
        first_img = files[0]
    from PIL import Image
    try:
        with Image.open(first_img) as im:
            w,h = im.width, im.height
    except Exception as e:
        logger.debug("Cannot determine image size: %s", e)
        return False
    encoder = encoder_hint or "libx264"
    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=white:s={w}x{h}:r={int(fps)}",
        "-framerate", str(int(fps)),
        "-thread_queue_size", str(THREAD_QUEUE_SIZE),
        "-start_number", "0",
        "-i", os.path.join(temp_img_folder, "%06d.png"),
    ]
    cmd = base + [
        "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
        "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", str(FFMPEG_CRF),
        "-pix_fmt", "yuv420p", out_segment_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        logger.exception("Pattern ffmpeg failed: %s", e)
        return False

def submit_segment_stream_or_fallback(image_list, temp_folder_pattern, seg_path, encoder, fps_param, allow_pattern_fallback=True):
    ok = make_segment_from_images_ffmpeg_stream(image_list, fps_param, seg_path, encoder_hint=encoder)
    if ok:
        return True
    if not allow_pattern_fallback:
        logger.warning("Streaming failed and pattern fallback disabled.")
        return False
    tmp = make_unique_tempdir(prefix="fallback_")
    try:
        # link images
        for i, src in enumerate(image_list):
            dest = os.path.join(tmp, f"{i:06d}.png")
            try:
                os.link(src, dest)
            except Exception:
                try:
                    shutil.copy2(src,dest)
                except Exception:
                    logger.debug("copy failed for %s", src)
        ok2 = make_segment_from_images_ffmpeg(tmp, fps_param, seg_path, encoder_hint=encoder)
    finally:
        try:
            if os.path.isdir(tmp):
                shutil.rmtree(tmp)
                unregister_temp_path(tmp)
        except Exception as e:
            logger.exception("Failed cleanup fallback tmp %s: %s", tmp, e)
    return ok2

def concat_segments_ffmpeg(segment_paths, out_final):
    if not segment_paths:
        return False
    import tempfile
    list_file = None
    try:
        list_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", dir=os.path.dirname(out_final) or None)
        for p in segment_paths:
            safe = p.replace("'", "'\"'\"'")
            list_file.write(f"file '{safe}'\n")
        list_file.flush()
        list_file.close()
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list_file.name, "-c", "copy", out_final]
        try:
            subprocess.run(cmd, check=True)
            try:
                os.remove(list_file.name)
            except Exception:
                pass
            return True
        except subprocess.CalledProcessError:
            logger.debug("Fast concat failed; trying re-encode concat")
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
        logger.exception("Concat failed: %s", e)
        try:
            if list_file:
                os.remove(list_file.name)
        except Exception:
            pass
        return False
