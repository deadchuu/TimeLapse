# modes/temporal_compositing.py
import os
import shutil
import tempfile
from math import exp
from PIL import Image

# optional acceleration
try:
    import numpy as _np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

def get_mode():
    return {
        "id": "temporal_compositing",
        "name": "Temporal compositing (decay curves)",
        "desc": "Temporal compositing with selectable decay curves (linear, exp, power) and color maps.",
        "enable_selection": True,
        "build_ffmpeg_cmd": build_ffmpeg_cmd,
        "select_frames": select_frames_temporal_compositing
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

# simple color maps
_COLORMAPS = {
    'purple': [(0.0,(0,0,0)), (1.0,(255,0,255))],
    'hot': [(0.0,(0,0,0)), (0.5,(255,128,0)), (1.0,(255,0,0))],
    'blue': [(0.0,(0,0,0)), (1.0,(0,0,255))],
}

def _map_intensity_to_rgb(intensity_arr, cmap):
    # intensity_arr: 1D numpy float array [0,1]
    # cmap: list of (pos, (r,g,b))
    stops_pos = [p for p,c in cmap]
    stops_col = [c for p,c in cmap]
    out = _np.zeros((intensity_arr.shape[0],3), dtype=_np.uint8)
    for i,v in enumerate(intensity_arr):
        if v <= stops_pos[0]:
            r,g,b = stops_col[0]
        elif v >= stops_pos[-1]:
            r,g,b = stops_col[-1]
        else:
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

def select_frames_temporal_compositing(saved_paths, mapping=None, region=None, prev_last_original=None,
                                       output_dir=None, batch_idx=0, *args, **kwargs):
    """
    Temporal compositing:
      kwargs:
        - fps: float
        - life_seconds or life_frames: how long the signal should persist (default 5s)
        - curve: 'linear'|'exp'|'power' (default 'exp')
        - power: float (for power curve, default 2.0)
        - colormap: 'purple'|'hot'|'blue' or custom list
    """
    if not saved_paths:
        return {'temp_sel_folder': None, 'selected_paths': []}

    fps = float(kwargs.get('fps', kwargs.get('fps_param', 30.0)))
    life_seconds = float(kwargs.get('life_seconds', 5.0))
    life_frames = kwargs.get('life_frames', max(1, int(round(life_seconds * fps))))
    curve = kwargs.get('curve', 'exp')
    power = float(kwargs.get('power', 2.0))
    cmap_name = kwargs.get('colormap', 'purple')
    cmap = _COLORMAPS.get(cmap_name, _COLORMAPS['purple']) if isinstance(cmap_name, str) else cmap_name

    tmp_folder = tempfile.mkdtemp(prefix=f"temporal_{batch_idx:03d}_", dir=(output_dir if output_dir else None))
    out_paths = []

    # baseline for size
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

    # prepare arrays
    if NUMPY_AVAILABLE:
        def rgba_to_u32(img):
            arr = _np.asarray(img.convert("RGBA"), dtype=_np.uint8)
            A = arr.reshape(-1,4)
            return (A[:,0].astype(_np.uint32) << 24) | (A[:,1].astype(_np.uint32) << 16) | (A[:,2].astype(_np.uint32) << 8) | A[:,3].astype(_np.uint32)

        baseline = rgba_to_u32(base_img)
        life = _np.zeros(total, dtype=_np.float32)  # 0..1
        out_index = 0

        # compute decay factor per frame for exponential so that after life_frames it decays near 0.01
        if curve == 'exp':
            decay_factor = float(max(0.0, min(0.9999, _np.exp(_np.log(0.01) / float(max(1, life_frames))))))
        else:
            decay_factor = None
        linear_decrement = 1.0 / float(max(1, life_frames))

        for path in saved_paths:
            try:
                cur = Image.open(path).convert("RGBA")
            except Exception:
                continue
            if cur.size != (w,h):
                continue
            cur_u = rgba_to_u32(cur)
            diff_mask = (cur_u != baseline)
            if diff_mask.any():
                idxs = _np.nonzero(diff_mask)[0]
                baseline[idxs] = cur_u[idxs]
                life[idxs] = 1.0  # reset to full on change

            # build intensity from life according to curve
            if curve == 'linear':
                intensity = _np.clip(life, 0.0, 1.0)
            elif curve == 'exp':
                intensity = _np.clip(life, 0.0, 1.0)
            elif curve == 'power':
                intensity = _np.clip(life ** power, 0.0, 1.0)
            else:
                intensity = _np.clip(life, 0.0, 1.0)

            # map to rgb
            rgb = _map_intensity_to_rgb(intensity, cmap)
            img_arr = rgb.reshape((h, w, 3))
            out_img = Image.fromarray(img_arr, mode="RGB")
            out_path = os.path.join(tmp_folder, f"{out_index:06d}.png")
            try:
                out_img.save(out_path, "PNG")
                out_paths.append(out_path)
                out_index += 1
            except Exception:
                pass

            # decay life
            if curve == 'exp' and decay_factor is not None:
                life *= decay_factor
            else:
                # linear or power (we decrement life linearly)
                life -= linear_decrement
                life = _np.clip(life, 0.0, 1.0)

        if not out_paths:
            try:
                shutil.rmtree(tmp_folder)
            except Exception:
                pass
            return {'temp_sel_folder': None, 'selected_paths': []}
        return {'temp_sel_folder': tmp_folder, 'selected_paths': out_paths}

    else:
        # Pillow fallback: use Python lists (slower)
        baseline_pixels = list(base_img.getdata())
        life = [0.0] * total
        out_index = 0
        linear_decrement = 1.0 / float(max(1, life_frames))
        for path in saved_paths:
            try:
                cur = Image.open(path).convert("RGBA")
            except Exception:
                continue
            if cur.size != (w,h):
                continue
            cur_pixels = list(cur.getdata())
            changed = [i for i,(b,c) in enumerate(zip(baseline_pixels, cur_pixels)) if b != c]
            if changed:
                for i in changed:
                    baseline_pixels[i] = cur_pixels[i]
                    life[i] = 1.0
            # compute intensity
            intensity = [0.0]*total
            if curve == 'linear':
                for i in range(total):
                    intensity[i] = max(0.0, min(1.0, life[i]))
            elif curve == 'power':
                for i in range(total):
                    intensity[i] = max(0.0, min(1.0, life[i] ** power))
            else:  # exp or default -> approximate with power fallback
                for i in range(total):
                    intensity[i] = max(0.0, min(1.0, life[i]))
            # map intensity to rgb via cmap
            rgb_flat = []
            for v in intensity:
                # linear interpolation between cmap stops
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
            # decay
            if curve == 'exp':
                # approximate exponential as linear decrement fallback
                for i in range(total):
                    life[i] -= linear_decrement
                    if life[i] < 0: life[i] = 0.0
            else:
                for i in range(total):
                    life[i] -= linear_decrement
                    if life[i] < 0: life[i] = 0.0

        if not out_paths:
            try:
                shutil.rmtree(tmp_folder)
            except Exception:
                pass
            return {'temp_sel_folder': None, 'selected_paths': []}
        return {'temp_sel_folder': tmp_folder, 'selected_paths': out_paths}
