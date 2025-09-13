# config.py
BATCH_SIZE = 1000
BATCH_WINDOW = 3
DIFF_PIXEL_THRESHOLD = 0.0002
DIFF_VIPS_THRESHOLD = 0.0002
FFMPEG_CRF = 18
FFMPEG_PRESET = "medium"
MAX_FFMPEG_PROCS = 2
THREAD_QUEUE_SIZE = 1024
VERBOSE = False
SUBTASKS_PER_BATCH = 1  # set 4 to split each batch into 4 subtasks for selection
BASE_OUTPUT_DIR = None  # will be set by core at runtime (cwd/timelaps)
LOG_FILE = "timelapse.log"
DEFAULT_FPS = 30.0
