def get_mode():
    """
    Mode: ROI filter
    - requires ROI selection (enable_selection = True)
    - identical ffmpeg command building to default
    """
    return {
        "id": "roi_filter",
        "name": "ROI filter (select changed frames in region)",
        "desc": "Filter frames by ROI changes and produce selected timelapse (plus always produce 'all' video).",
        "enable_selection": True,
        "build_ffmpeg_cmd": build_ffmpeg_cmd
    }

def build_ffmpeg_cmd(first_img, temp_pattern, w, h, fps, out_segment_path, encoder, thread_queue_size, ffmpeg_preset, ffmpeg_crf):
    # same as default overlay behavior
    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=white:s={w}x{h}:r={int(fps)}",
        "-framerate", str(int(fps)),
        "-thread_queue_size", str(thread_queue_size),
        "-start_number", "0",
        "-i", temp_pattern,
    ]

    if encoder == "libx264" or encoder is None:
        cmd = base + [
            "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
            "-c:v", "libx264", "-preset", ffmpeg_preset, "-crf", str(ffmpeg_crf),
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
                "-i", temp_pattern,
                "-vf", "format=nv12,hwupload,overlay=shortest=1",
                "-c:v", "h264_vaapi",
                out_segment_path
            ]
        else:
            cmd = base + [
                "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto",
                "-c:v", "libx264", "-preset", ffmpeg_preset, "-crf", str(ffmpeg_crf),
                "-pix_fmt", "yuv420p", out_segment_path
            ]
    return cmd
