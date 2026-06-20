import os

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"


BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode


def create_landmarker():
    """
    Cria o detector de landmarks faciais usando a API nova do MediaPipe Tasks.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}\n"
            "Baixe o arquivo face_landmarker.task e coloque dentro da pasta models/."
        )

    if MODEL_PATH.stat().st_size == 0:
        raise ValueError(
            f"O arquivo {MODEL_PATH} está vazio. "
            "Você precisa baixar o modelo real face_landmarker.task."
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return FaceLandmarker.create_from_options(options)


def extract_landmarks_from_frame(landmarker, frame, timestamp_ms):
    """
    Recebe um frame BGR do OpenCV e retorna os landmarks em coordenadas de pixel.

    Retorno:
        None, se nenhum rosto for detectado.
        Lista de tuplas [(x, y), ...], se detectar rosto.
    """

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return None

    face_landmarks = result.face_landmarks[0]

    landmarks = []

    for lm in face_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append((x, y))

    return landmarks