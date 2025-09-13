# modes/any_pixel_roi.py
from PIL import Image, ImageChops

def get_mode():
    return {
        "id": "any_pixel_roi",
        "name": "Any-pixel change (ROI)",
        "desc": "Select frames if at least one pixel changed inside ROI.",
        "enable_selection": True,
        "build_ffmpeg_cmd": build_ffmpeg_cmd,
        "select_frames": select_frames_any_pixel_roi
    }

def build_ffmpeg_cmd(first_img, temp_pattern, w, h, fps, out_segment_path, encoder, thread_queue_size, ffmpeg_preset, ffmpeg_crf):
    # reuse overlay pipeline from default
    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=white:s={w}x{h}:r={int(fps)}",
        "-framerate", str(int(fps)),
        "-thread_queue_size", str(thread_queue_size),
        "-start_number", "0",
        "-i", temp_pattern,
    ]
    cmd = base + [
        "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
        "-c:v", "libx264", "-preset", ffmpeg_preset, "-crf", str(ffmpeg_crf),
        "-pix_fmt", "yuv420p", out_segment_path
    ]
    return cmd

def frames_differ_any(prev_path, curr_path, region=None):
    try:
        a = Image.open(prev_path).convert("RGBA")
        b = Image.open(curr_path).convert("RGBA")
    except Exception:
        return True
    if region:
        x1, y1, x2, y2 = region
        a = a.crop((x1, y1, x2, y2))
        b = b.crop((x1, y1, x2, y2))
    if a.size != b.size:
        return True
    diff = ImageChops.difference(a, b)
    bbox = diff.getbbox()
    return bbox is not None

def select_frames_any_pixel_roi(saved_paths, mapping, region, prev_last_original, *args, **kwargs):
    if region is None:
        # If no region provided, behave like full-image any-pixel
        return select_frames_any_pixel_full(saved_paths, mapping, None, prev_last_original)
    # same as full but pass region to frames_differ_any
    selected = []
    if not saved_paths:
        return selected
    first = saved_paths[0]
    if prev_last_original:
        try:
            if frames_differ_any(prev_last_original, first, region):
                selected.append(first)
        except Exception:
            selected.append(first)
    else:
        selected.append(first)
    for i in range(1, len(saved_paths)):
        prev = saved_paths[i-1]
        curr = saved_paths[i]
        try:
            if frames_differ_any(prev, curr, region):
                selected.append(curr)
        except Exception:
            selected.append(curr)
    return selected

# helper: reuse from full mode
def select_frames_any_pixel_full(saved_paths, mapping, region, prev_last_original):
    # minimal copy to avoid import cycle
    selected = []
    if not saved_paths:
        return selected
    first = saved_paths[0]
    if prev_last_original:
        try:
            if frames_differ_any(prev_last_original, first, None):
                selected.append(first)
        except Exception:
            selected.append(first)
    else:
        selected.append(first)
    for i in range(1, len(saved_paths)):
        prev = saved_paths[i-1]
        curr = saved_paths[i]
        try:
            if frames_differ_any(prev, curr, None):
                selected.append(curr)
        except Exception:
            selected.append(curr)
    return selected
