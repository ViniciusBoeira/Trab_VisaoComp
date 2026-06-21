import cv2
import numpy as np


def analyze_frame_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings = []

    if brightness < 60:
        warnings.append("Iluminacao baixa")

    if contrast < 25:
        warnings.append("Baixo contraste")

    if sharpness < 80:
        warnings.append("Imagem borrada")

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "warnings": warnings,
    }


def preprocess_frame_for_landmarks(frame, quality=None, enabled=False):
    """
    Pré-processamento opcional antes do MediaPipe.

    Por padrão fica desativado para manter coerência com o treino.
    Se quiser testar CLAHE como entrada do MediaPipe, ligue enabled=True
    tanto no extractor quanto no worker_camera.
    """
    if not enabled:
        return frame

    if quality is not None:
        has_bad_light = quality.get("brightness", 999) < 60
        has_low_contrast = quality.get("contrast", 999) < 25

        if not has_bad_light and not has_low_contrast:
            return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_frame


def extract_roi(frame, landmarks, indexes, padding=20):
    points = np.array([landmarks[i] for i in indexes])

    x_min = int(max(np.min(points[:, 0]) - padding, 0))
    y_min = int(max(np.min(points[:, 1]) - padding, 0))
    x_max = int(min(np.max(points[:, 0]) + padding, frame.shape[1]))
    y_max = int(min(np.max(points[:, 1]) + padding, frame.shape[0]))

    roi = frame[y_min:y_max, x_min:x_max]

    return roi, (x_min, y_min, x_max, y_max)


def transform_roi_for_visualization(roi):
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 80, 160)

    return {
        "gray": gray,
        "enhanced": enhanced,
        "edges": edges,
    }
