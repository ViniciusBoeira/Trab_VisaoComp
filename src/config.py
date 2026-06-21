from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

MEDIAPIPE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
DATASET_OUTPUT_CSV = BASE_DIR / "data" / "processed" / "features_all.csv"
RUNTIME_LOG_PATH = BASE_DIR / "data" / "logs" / "runtime_predictions.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "model_skip5_window5_extratrees.pkl"

DATASET_CONFIG = [
    {
        "source": "uta",
        "path": BASE_DIR / "data" / "raw" / "uta" / "normal",
        "label": "NORMAL",
    },
    {
        "source": "uta",
        "path": BASE_DIR / "data" / "raw" / "uta" / "drowsiness",
        "label": "SONOLENTO",
    },
]

VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Extração offline dos vídeos.
FRAME_SKIP = 5
WINDOW_SECONDS = 5
MAX_WINDOWS_PER_VIDEO = 40
DEBUG_SHOW = False

# Limiar inicial baseado nas características do seu dataset.
# Esses valores afetam PERCLOS e eventos de boca aberta.
EAR_THRESHOLD = 0.15
MAR_THRESHOLD = 0.12

# Webcam/runtime.
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# WEBCAM CONFIGS
PROCESS_EVERY_N_FRAMES = 5
RUNTIME_WINDOW_SECONDS = 5
RUNTIME_WINDOW_SIZE = int((CAMERA_FPS / PROCESS_EVERY_N_FRAMES) * RUNTIME_WINDOW_SECONDS)

USE_SAFETY_RULE = True
ENABLE_CSV_LOG = True

USE_ARDUINO = True
SERIAL_PORT = "COM3"
BAUD_RATE = 9600
