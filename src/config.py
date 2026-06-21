from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

MEDIAPIPE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
DATASET_OUTPUT_CSV = BASE_DIR / "data" / "processed" / "features_all.csv"
RUNTIME_LOG_PATH = BASE_DIR / "data" / "logs" / "runtime_predictions.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "random_forest_drowsiness.pkl"

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
FRAME_SKIP = 10
WINDOW_SECONDS = 10
MAX_WINDOWS_PER_VIDEO = 20
DEBUG_SHOW = False

# Limiar inicial baseado nas características do seu dataset.
# Esses valores afetam PERCLOS e eventos de boca aberta.
EAR_THRESHOLD = 0.15
MAR_THRESHOLD = 0.12

# Webcam/runtime.
CAMERA_INDEX = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Para ficar coerente com o treino: 30 medições em ~10 segundos.
# 30 FPS / 10 = 3 medições por segundo; 3 * 10s = 30 medições.
PROCESS_EVERY_N_FRAMES = 5
RUNTIME_WINDOW_SECONDS = 10
RUNTIME_WINDOW_SIZE = int((CAMERA_FPS / PROCESS_EVERY_N_FRAMES) * RUNTIME_WINDOW_SECONDS)

USE_SAFETY_RULE = True
ENABLE_CSV_LOG = True

USE_ARDUINO = False
SERIAL_PORT = "COM3"
BAUD_RATE = 9600