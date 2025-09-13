# engine/compare.py
"""
Compare backends + auto-detection.
Provides batch_compare_pairs(pairs, backend=None, threshold_pixels=0, device=None, region=None)
"""
from engine.utils import get_logger
logger = get_logger()

# optional libs
_TORCH = None
_CV2 = None
_PYOPENCL = None
_PYVIPS = None
_PIL = None

try:
    import torch
    _TORCH = torch
except Exception:
    pass

try:
    import cv2
    _CV2 = cv2
except Exception:
    pass

try:
    import pyopencl as cl
    _PYOPENCL = cl
except Exception:
    pass

try:
    import pyvips
    _PYVIPS = pyvips
except Exception:
    pass

try:
    from PIL import Image, ImageChops
    _PIL = True
except Exception:
    _PIL = False

import numpy as _np
import os

def detect_best_backend():
    if _TORCH is not None:
        try:
            if _TORCH.cuda.is_available():
                logger.info("Using backend: torch_cuda")
                return "torch_cuda"
        except Exception:
            pass
    if _CV2 is not None:
        try:
            if hasattr(_CV2, "cuda"):
                try:
                    dev_count = _CV2.cuda.getCudaEnabledDeviceCount()
                    if dev_count > 0:
                        logger.info("Using backend: opencv_cuda")
                        return "opencv_cuda"
                except Exception:
                    pass
        except Exception:
            pass
    if _PYOPENCL is not None:
        try:
            plats = _PYOPENCL.get_platforms()
            if plats:
                logger.info("Using backend: pyopencl")
                return "pyopencl"
        except Exception:
            pass
    if _PYVIPS is not None:
        logger.info("Using backend: pyvips")
        return "pyvips"
    logger.info("Using backend: cpu")
    if _TORCH is None:
        logger.debug("Optional: install torch for GPU.")
    if _CV2 is None:
        logger.debug("Optional: install opencv (with cuda) for GPU diffs.")
    if _PYOPENCL is None:
        logger.debug("Optional: install pyopencl for OpenCL on Intel/AMD.")
    return "cpu"

# CPU Pillow compare
def _compare_pair_pillow_count(prev, curr, region=None):
    try:
        from PIL import Image, ImageChops
        a = Image.open(prev).convert("RGBA")
        b = Image.open(curr).convert("RGBA")
    except Exception as e:
        logger.debug("Pillow open failed: %s", e)
        return float('inf')
    if region:
        x1,y1,x2,y2 = region
        a = a.crop((x1,y1,x2,y2))
        b = b.crop((x1,y1,x2,y2))
    if a.size != b.size:
        return float('inf')
    diff = ImageChops.difference(a,b)
    diff_rgb = diff.convert("RGB")
    diff_count = 0
    for px in diff_rgb.getdata():
        if px[0] != 0 or px[1] != 0 or px[2] != 0:
            diff_count += 1
    return diff_count

def _files_are_byte_equal(a,b):
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        with open(a,'rb') as fa, open(b,'rb') as fb:
            while True:
                ba = fa.read(65536)
                bb = fb.read(65536)
                if not ba and not bb:
                    return True
                if ba != bb:
                    return False
    except Exception as e:
        logger.debug("Byte compare failed: %s", e)
        return False

# Torch batch compare
def _batch_compare_torch(pairs, device='cuda', threshold_pixels=0):
    changed = []
    if not pairs:
        return changed
    ts_prev = []
    ts_curr = []
    idxs = []
    for i,(p,c) in enumerate(pairs):
        try:
            from PIL import Image
            a = Image.open(p).convert("RGB")
            b = Image.open(c).convert("RGB")
            arr_a = _np.asarray(a, dtype=_np.uint8)
            arr_b = _np.asarray(b, dtype=_np.uint8)
            if arr_a.shape != arr_b.shape:
                changed.append(c)
                continue
            ta = _TORCH.from_numpy(arr_a).permute(2,0,1).to(device=device, dtype=_TORCH.int16)
            tb = _TORCH.from_numpy(arr_b).permute(2,0,1).to(device=device, dtype=_TORCH.int16)
            ts_prev.append(ta)
            ts_curr.append(tb)
            idxs.append(i)
        except Exception as e:
            logger.debug("Torch load error for pair %s,%s: %s", p, c, e)
            changed.append(c)
    if not ts_prev:
        return changed
    prev_batch = _TORCH.stack(ts_prev, dim=0)
    curr_batch = _TORCH.stack(ts_curr, dim=0)
    diff = (prev_batch - curr_batch).abs()
    diff_any = (diff.sum(dim=1) > 0)
    counts = diff_any.view(diff_any.shape[0], -1).sum(dim=1).cpu().numpy()
    for j, cnt in enumerate(counts):
        if cnt > threshold_pixels:
            changed.append(pairs[idxs[j]][1])
    return changed

# OpenCV CUDA per pair
def _pair_compare_opencv_cuda(prev, curr, threshold_pixels=0):
    try:
        a = _CV2.imread(prev, _CV2.IMREAD_UNCHANGED)
        b = _CV2.imread(curr, _CV2.IMREAD_UNCHANGED)
        if a is None or b is None:
            return True
        if a.shape != b.shape:
            return True
        ga = _CV2.cuda_GpuMat()
        gb = _CV2.cuda_GpuMat()
        ga.upload(a)
        gb.upload(b)
        gdiff = _CV2.cuda.absdiff(ga, gb)
        ggray = _CV2.cuda.cvtColor(gdiff, _CV2.COLOR_BGR2GRAY)
        _, gth = _CV2.cuda.threshold(ggray, 0, 255, _CV2.THRESH_BINARY)
        th_cpu = gth.download()
        cnt = (th_cpu > 0).sum()
        return cnt > threshold_pixels
    except Exception as e:
        logger.debug("opencv.cuda compare failed: %s", e)
        return True

# pyopencl fallback (simple CPU computation for now)
def _batch_compare_pyopencl(pairs, threshold_pixels=0):
    changed = []
    for p,c in pairs:
        try:
            from PIL import Image
            a = Image.open(p).convert("RGB")
            b = Image.open(c).convert("RGB")
            A = _np.asarray(a, dtype=_np.int16)
            B = _np.asarray(b, dtype=_np.int16)
            if A.shape != B.shape:
                changed.append(c)
                continue
            if (_np.abs(A - B)).sum() > 0:
                changed.append(c)
        except Exception:
            changed.append(c)
    return changed

# pyvips avg
def _pair_compare_pyvips(prev, curr, region=None):
    try:
        a = _PYVIPS.Image.new_from_file(prev, access='sequential')
        b = _PYVIPS.Image.new_from_file(curr, access='sequential')
        if region:
            x1,y1,x2,y2 = region
            w = x2 - x1
            h = y2 - y1
            a = a.crop(x1,y1,w,h)
            b = b.crop(x1,y1,w,h)
        if a.hasalpha():
            a = a[:3]
        if b.hasalpha():
            b = b[:3]
        diff = (a - b).abs()
        avg = diff.avg()
        return float(avg)
    except Exception as e:
        logger.debug("pyvips compare failed: %s", e)
        return float('inf')

def batch_compare_pairs(pairs, backend=None, threshold_pixels=0, device=None, region=None):
    if backend is None:
        backend = detect_best_backend()
    logger.debug("batch_compare_pairs backend=%s device=%s", backend, device)
    if backend == "torch_cuda" and _TORCH is not None:
        dev = device or ("cuda" if _TORCH.cuda.is_available() else "cpu")
        try:
            return _batch_compare_torch(pairs, device=dev, threshold_pixels=threshold_pixels)
        except Exception as e:
            logger.exception("Torch compare failed: %s", e)
            backend = "cpu"
    if backend == "opencv_cuda" and _CV2 is not None:
        changed = []
        for prev,curr in pairs:
            try:
                if _pair_compare_opencv_cuda(prev, curr, threshold_pixels=threshold_pixels):
                    changed.append(curr)
            except Exception:
                changed.append(curr)
        return changed
    if backend == "pyopencl" and _PYOPENCL is not None:
        return _batch_compare_pyopencl(pairs, threshold_pixels=threshold_pixels)
    if backend == "pyvips" and _PYVIPS is not None:
        changed = []
        for prev,curr in pairs:
            try:
                avg = _pair_compare_pyvips(prev,curr,region=region)
                if avg == float('inf') or avg > 0:
                    changed.append(curr)
            except Exception:
                changed.append(curr)
        return changed
    # CPU fallback (Pillow or byte)
    out = []
    if _PIL:
        from PIL import Image, ImageChops
        for prev,curr in pairs:
            try:
                a = Image.open(prev).convert("RGBA")
                b = Image.open(curr).convert("RGBA")
                if region:
                    x1,y1,x2,y2 = region
                    a = a.crop((x1,y1,x2,y2))
                    b = b.crop((x1,y1,x2,y2))
                if a.size != b.size:
                    out.append(curr)
                    continue
                diff = ImageChops.difference(a,b)
                bbox = diff.getbbox()
                if bbox is not None:
                    out.append(curr)
            except Exception:
                out.append(curr)
        return out
    else:
        for prev,curr in pairs:
            try:
                if not _files_are_byte_equal(prev,curr):
                    out.append(curr)
            except Exception:
                out.append(curr)
        return out
