# modes/heatmap_accumulation.py
import os
import shutil
import tempfile
from PIL import Image

# optional acceleration
try:
    import numpy as _np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

def get_mode():
    return {
        "id": "heatmap_accumulation",
        "name": "Heatmap accumulation",
        "desc": "Accumulate change locations as a heatmap and render a heatmap video (useful for analytics).",
        "enable_selection": True,
        "build_ffmpeg_cmd": build_ffmpeg_cmd,
        "select_frames": select_frames_heatmap_accumulation
    }

def build_ffmpeg_cmd(first_img, temp_pattern, w, h, fps, out_segment_path, encoder, thread_queue_size, ffmpeg_preset, ffmpeg_crf):
    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(int(fps)),
        "-thread_queue_size", str(thread_queue_size),
        "-start_number", "0",
        "-i", temp_pattern,
    ]
    cmd = base + [
        "-c:v", "libx264", "-preset", ffmpeg_preset, "-crf", str(ffmpeg_crf),
        "-pix_fmt", "yuv420p", out_segment_path
    ]
    return cmd

# simple colormap: list of (position, (r,g,b))
_DEFAULT_COLORMAP = [
    (0.0, (0, 0, 128)),   # dark blue
    (0.2, (0, 0, 255)),   # blue
    (0.4, (0, 255, 255)), # cyan
    (0.6, (0, 255, 0)),   # green
    (0.8, (255, 255, 0)), # yellow
    (1.0, (255, 0, 0)),   # red
]

def _apply_colormap_to_norm(norm_arr, colormap=_DEFAULT_COLORMAP):
    """
    norm_arr: 1D numpy float array in [0,1] length N
    returns: uint8 array shape (N,3)
    """
    stops_pos = [p for p,c in colormap]
    stops_col = [c for p,c in colormap]
    # linear interpolation between stops
    out = _np.zeros((norm_arr.shape[0], 3), dtype=_np.uint8)
    for i,val in enumerate(norm_arr):
        if _np.isnan(val):
            val = 0.0
        v = float(val)
        if v <= stops_pos[0]:
            r,g,b = stops_col[0]
        elif v >= stops_pos[-1]:
            r,g,b = stops_col[-1]
        else:
            # find segment
            for j in range(len(stops_pos)-1):
                p0 = stops_pos[j]; p1 = stops_pos[j+1]
                if p0 <= v <= p1:
                    t = (v - p0) / (p1 - p0) if (p1 - p0) != 0 else 0.0
                    c0 = stops_col[j]; c1 = stops_col[j+1]
                    r = int(round(c0[0] + (c1[0]-c0[0])*t))
                    g = int(round(c0[1] + (c1[1]-c0[1])*t))
                    b = int(round(c0[2] + (c1[2]-c0[2])*t))
                    break
        out[i,0] = r; out[i,1] = g; out[i,2] = b
    return out

def select_frames_heatmap_accumulation(saved_paths, mapping=None, region=None, prev_last_original=None,
                                       output_dir=None, batch_idx=0, *args, **kwargs):
    """
    Build heatmap frames:
      - Count number of times each pixel changed in the batch (accumulation).
      - Render a sequence of heatmap frames showing progressive accumulation.
    kwargs supported:
      - max_accum: int, cap for accumulation normalization (default: dynamic)
      - frames_per_update: int, how many heatmap frames to output while accumulating (default  max(1, len(saved_paths)//10))
      - colormap: custom colormap list (pos, (r,g,b)) -- if provided and numpy available, will be used.
    """
    if not saved_paths:
        return {'temp_sel_folder': None, 'selected_paths': []}

    # params
    max_accum = kwargs.get('max_accum', None)
    frames_per_update = kwargs.get('frames_per_update', max(1, max(1, len(saved_paths)//10)))
    colormap = kwargs.get('colormap', None)
    if colormap and NUMPY_AVAILABLE:
        cmap = colormap
    else:
        cmap = _DEFAULT_COLORMAP

    tmp_folder = tempfile.mkdtemp(prefix=f"heatmap_{batch_idx:03d}_", dir=(output_dir if output_dir else None))
    out_paths = []

    # baseline image to get dimensions
    try:
        first = prev_last_original if (prev_last_original and os.path.exists(prev_last_original)) else saved_paths[0]
        base_img = Image.open(first).convert("RGBA")
    except Exception:
        try:
            base_img = Image.open(saved_paths[0]).convert("RGBA")
        except Exception:
            try:
                shutil.rmtree(tmp_folder)
            except Exception:
                pass
            return {'temp_sel_folder': None, 'selected_paths': []}

    w,h = base_img.size
    total = w*h

    if NUMPY_AVAILABLE:
        # use uint32 packing to compare quickly
        def rgba_to_u32_array(img):
            arr = _np.asarray(img.convert("RGBA"), dtype=_np.uint8)
            A = arr.reshape(-1,4)
            return (A[:,0].astype(_np.uint32) << 24) | (A[:,1].astype(_np.uint32) << 16) | (A[:,2].astype(_np.uint32) << 8) | A[:,3].astype(_np.uint32)

        baseline = rgba_to_u32_array(base_img)
        accum = _np.zeros(total, dtype=_np.int32)

        step = max(1, int(len(saved_paths) / frames_per_update))
        out_index = 0
        for idx, path in enumerate(saved_paths):
            try:
                cur = Image.open(path).convert("RGBA")
            except Exception:
                continue
            if cur.size != (w,h):
                continue
            cur_u = rgba_to_u32_array(cur)
            diff_mask = (cur_u != baseline)
            if diff_mask.any():
                # increment counts where changed, and update baseline
                changed_idx = _np.nonzero(diff_mask)[0]
                accum[changed_idx] += 1
                baseline[changed_idx] = cur_u[changed_idx]
            # output at steps
            if (idx % step == 0) or (idx == len(saved_paths)-1):
                # normalize accum to [0,1]
                if max_accum is None:
                    denom = float(accum.max() if accum.max() > 0 else 1)
                else:
                    denom = float(max_accum)
                norm = _np.clip(accum.astype(_np.float32) / denom, 0.0, 1.0)
                rgb = _apply_colormap_to_norm(norm, colormap=cmap)
                # convert to HxWx3 to save
                img_arr = rgb.reshape((h, w, 3))
                out_img = Image.fromarray(img_arr, mode="RGB")
                out_path = os.path.join(tmp_folder, f"{out_index:06d}.png")
                try:
                    out_img.save(out_path, "PNG")
                    out_paths.append(out_path)
                    out_index += 1
                except Exception:
                    pass

        if not out_paths:
            try:
                shutil.rmtree(tmp_folder)
            except Exception:
                pass
            return {'temp_sel_folder': None, 'selected_paths': []}
        return {'temp_sel_folder': tmp_folder, 'selected_paths': out_paths}

    else:
        # pure-Pillow fallback: compute accum as Python list (slower)
        baseline_pixels = list(base_img.getdata())
        accum = [0] * total
        out_index = 0
        step = max(1, int(len(saved_paths) / frames_per_update))
        for idx, path in enumerate(saved_paths):
            try:
                cur = Image.open(path).convert("RGBA")
            except Exception:
                continue
            if cur.size != (w,h):
                continue
            cur_pixels = list(cur.getdata())
            changed_idx = [i for i,(b,c) in enumerate(zip(baseline_pixels, cur_pixels)) if b != c]
            if changed_idx:
                for i in changed_idx:
                    accum[i] += 1
                    baseline_pixels[i] = cur_pixels[i]
            if (idx % step == 0) or (idx == len(saved_paths)-1):
                max_val = max(accum) if max_accum is None else max_accum
                if max_val <= 0: max_val = 1
                norm = [min(1.0, float(v)/float(max_val)) for v in accum]
                # map norm to rgb via simple colormap mapping
                rgb_flat = []
                for v in norm:
                    # reuse simple interpolation from _DEFAULT_COLORMAP
                    # (inefficient but ok for fallback)
                    for j in range(len(cmap)-1):
                        p0, c0 = cmap[j]; p1, c1 = cmap[j+1]
                        if p0 <= v <= p1:
                            t = (v - p0) / (p1 - p0) if (p1 - p0) != 0 else 0.0
                            r = int(round(c0[0] + (c1[0]-c0[0])*t))
                            g = int(round(c0[1] + (c1[1]-c0[1])*t))
                            b = int(round(c0[2] + (c1[2]-c0[2])*t))
                            rgb_flat.append((r,g,b))
                            break
                    else:
                        rgb_flat.append(cmap[-1][1])
                out_img = Image.new("RGB", (w,h))
                out_img.putdata(rgb_flat)
                out_path = os.path.join(tmp_folder, f"{out_index:06d}.png")
                try:
                    out_img.save(out_path, "PNG")
                    out_paths.append(out_path)
                    out_index += 1
                except Exception:
                    pass

        if not out_paths:
            try:
                shutil.rmtree(tmp_folder)
            except Exception:
                pass
            return {'temp_sel_folder': None, 'selected_paths': []}
        return {'temp_sel_folder': tmp_folder, 'selected_paths': out_paths}
