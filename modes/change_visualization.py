# modes/change_visualization.py
import os
import shutil
import tempfile
from math import ceil
from PIL import Image

# Optional acceleration
try:
    import numpy as _np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

try:
    import pyvips as _pyvips
    VIPS_AVAILABLE = True
except Exception:
    VIPS_AVAILABLE = False

def get_mode():
    return {
        "id": "change_visualization",
        "name": "Change visualization (purple fades)",
        "desc": "Visualize changed pixels on black background as purple that fades over time. "
                "Pixel life is scaled by FPS (so lifetime in seconds is approximately constant).",
        "enable_selection": True,
        "build_ffmpeg_cmd": build_ffmpeg_cmd,
        "select_frames": select_frames_change_visualization
    }

def build_ffmpeg_cmd(first_img, temp_pattern, w, h, fps, out_segment_path, encoder, thread_queue_size, ffmpeg_preset, ffmpeg_crf):
    """
    Encode frames that are already 'final' visualization frames.
    We assume temp_pattern is like /path/to/temp/%06d.png (no overlay needed).
    """
    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(int(fps)),
        "-thread_queue_size", str(thread_queue_size),
        "-start_number", "0",
        "-i", temp_pattern,
    ]
    # encode directly
    cmd = base + [
        "-c:v", "libx264", "-preset", ffmpeg_preset, "-crf", str(ffmpeg_crf),
        "-pix_fmt", "yuv420p", out_segment_path
    ]
    return cmd

# --- configuration for visualization lifetime ---
# Interpret as: BASE_LIFE_FRAMES frames at BASELINE_FPS correspond to desired lifetime in seconds.
BASE_LIFE_FRAMES = 5
BASELINE_FPS = 30
DEFAULT_FPS = 30
PURPLE_RGB = (255, 0, 255)

def _get_fps_from_kwargs(kwargs):
    fps = None
    if kwargs is None:
        return DEFAULT_FPS
    fps = kwargs.get("fps") or kwargs.get("fps_param") or kwargs.get("fps_actual")
    if fps is None:
        envv = os.environ.get("TIMELAPSE_FPS")
        if envv:
            try:
                fps = float(envv)
            except Exception:
                fps = None
    try:
        fps = float(fps) if fps is not None else DEFAULT_FPS
    except Exception:
        fps = DEFAULT_FPS
    if fps <= 0:
        fps = DEFAULT_FPS
    return fps

def _compute_life_frames_for_fps(fps, base_frames=BASE_LIFE_FRAMES, baseline_fps=BASELINE_FPS):
    life_seconds = float(base_frames) / float(baseline_fps)
    life_frames = max(1, int(round(life_seconds * float(fps))))
    return life_frames

def select_frames_change_visualization(saved_paths, mapping=None, region=None, prev_last_original=None,
                                      output_dir=None, batch_idx=0, *args, **kwargs):
    """
    Build visualized frames into a temp folder and return:
        {'temp_sel_folder': temp_folder, 'selected_paths': [list of png paths in that folder]}
    - Accepts 'fps' in kwargs (float)
    - Optionally accepts 'life_seconds' or 'life_frames' override
    - Uses numpy/pyvips if available for speed
    """
    if not saved_paths:
        return {'temp_sel_folder': None, 'selected_paths': []}

    fps_actual = _get_fps_from_kwargs(kwargs)
    life_frames = _compute_life_frames_for_fps(fps_actual)

    # allow overrides
    if 'life_frames' in kwargs and isinstance(kwargs['life_frames'], int) and kwargs['life_frames'] > 0:
        life_frames = kwargs['life_frames']
    if 'life_seconds' in kwargs:
        try:
            lfsec = float(kwargs['life_seconds'])
            if lfsec > 0:
                life_frames = max(1, int(round(lfsec * fps_actual)))
        except Exception:
            pass

    # create tmp folder
    tmp_folder = tempfile.mkdtemp(prefix=f"vis_sel_{batch_idx:03d}_", dir=(output_dir if output_dir else None))
    selected_out_paths = []

    # load baseline
    first_img_path = saved_paths[0]
    try:
        if prev_last_original and os.path.exists(prev_last_original):
            baseline_img = Image.open(prev_last_original).convert("RGBA")
        else:
            baseline_img = Image.open(first_img_path).convert("RGBA")
    except Exception:
        try:
            baseline_img = Image.open(first_img_path).convert("RGBA")
        except Exception:
            try:
                shutil.rmtree(tmp_folder)
            except Exception:
                pass
            return {'temp_sel_folder': None, 'selected_paths': []}

    w, h = baseline_img.size
    total_pixels = w * h

    # baseline pixels: as list of tuples
    baseline_pixels = list(baseline_img.getdata())
    # life array
    life = [0] * total_pixels

    # Fast path using numpy if available
    if NUMPY_AVAILABLE:
        # Represent baseline as uint32 packed values for fast comparison
        def rgba_to_uint32(arr):
            a = (_np.array(arr, dtype=_np.uint8)).reshape(-1,4)
            return (a[:,0].astype(_np.uint32) << 24) | (a[:,1].astype(_np.uint32) << 16) | (a[:,2].astype(_np.uint32) << 8) | a[:,3].astype(_np.uint32)
        baseline_u = rgba_to_uint32(baseline_pixels)

        out_index = 0
        for path in saved_paths:
            try:
                curr_img = Image.open(path).convert("RGBA")
            except Exception:
                continue
            if curr_img.size != (w,h):
                continue
            curr_pixels = list(curr_img.getdata())
            curr_u = rgba_to_uint32(curr_pixels)
            diff_mask = (curr_u != baseline_u)
            if not diff_mask.any():
                continue
            # update baseline and reset life
            changed_idx = _np.nonzero(diff_mask)[0].tolist()
            for i in changed_idx:
                baseline_u[i] = curr_u[i]
                life[i] = life_frames
            # build output array (RGB)
            # we will build as uint8 (H*W,3)
            out_rgb = _np.zeros((total_pixels,3), dtype=_np.uint8)
            # find indices with life>0
            life_arr = _np.array(life, dtype=_np.int32)
            active_idx = _np.nonzero(life_arr > 0)[0]
            if active_idx.size > 0:
                # compute brightness scaled
                vals = ( (_np.clip( (life_arr[active_idx].astype(_np.float32) / float(life_frames)) * 255.0, 0, 255)).astype(_np.uint8) )
                out_rgb[active_idx,0] = vals
                out_rgb[active_idx,2] = vals
            # create image via frombytes
            out_img = Image.fromarray(out_rgb.reshape((h,w,3)), mode="RGB")
            out_path = os.path.join(tmp_folder, f"{out_index:06d}.png")
            try:
                out_img.save(out_path, "PNG")
                selected_out_paths.append(out_path)
                out_index += 1
            except Exception:
                pass
            # decrement life
            for i in active_idx:
                life[i] -= 1
        # end numpy path
    else:
        # pure-Pillow path (optimized with putdata)
        out_index = 0
        for path in saved_paths:
            try:
                curr_img = Image.open(path).convert("RGBA")
            except Exception:
                continue
            if curr_img.size != (w,h):
                continue
            curr_pixels = list(curr_img.getdata())

            # compute changed indices
            changed_indices = [i for i,(b,c) in enumerate(zip(baseline_pixels, curr_pixels)) if b != c]
            if not changed_indices:
                continue
            for i in changed_indices:
                baseline_pixels[i] = curr_pixels[i]
                life[i] = life_frames

            # create out data
            black_pixel = (0,0,0)
            out_data = [black_pixel] * total_pixels
            lf = life_frames
            for i in range(total_pixels):
                li = life[i]
                if li > 0:
                    f = float(li) / float(lf)
                    val = int(round(255.0 * f))
                    if val < 0: val = 0
                    elif val > 255: val = 255
                    out_data[i] = (val,0,val)
            out_img = Image.new("RGB", (w,h))
            out_img.putdata(out_data)
            out_path = os.path.join(tmp_folder, f"{out_index:06d}.png")
            try:
                out_img.save(out_path, "PNG")
                selected_out_paths.append(out_path)
                out_index += 1
            except Exception:
                pass

            # decrement life
            for i in range(total_pixels):
                if life[i] > 0:
                    life[i] -= 1

    if not selected_out_paths:
        try:
            shutil.rmtree(tmp_folder)
        except Exception:
            pass
        return {'temp_sel_folder': None, 'selected_paths': []}

    return {'temp_sel_folder': tmp_folder, 'selected_paths': selected_out_paths}
