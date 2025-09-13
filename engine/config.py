# engine/config.py
# Configuration constants (tweak as needed)

BASE_OUTPUT_DIR = None  # if None -> will be created as ./timelapse in working dir
BATCH_SIZE = 1000
BATCH_WINDOW = 3
DIFF_PIXEL_THRESHOLD = 10
DIFF_VIPS_THRESHOLD = 0.01
FFMPEG_CRF = 18
FFMPEG_PRESET = "medium"
MAX_FFMPEG_PROCS = 2
THREAD_QUEUE_SIZE = 1024
VERBOSE = False
SUBTASKS_PER_BATCH = 1
LOG_FILE = "timelapse.log"
DEFAULT_FPS = 30.0
