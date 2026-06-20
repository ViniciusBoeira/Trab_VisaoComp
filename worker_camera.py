from pathlib import Path
import time
import csv
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd

from src.feature_extractor import (
    create_landmarker,
    extract_landmarks_from_frame,
    extract_ear_mar_from_landmarks,
    calculate_window_features,
    LEFT_EYE,
    RIGHT_EYE,
    MOUTH_ROI,
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_drowsiness.pkl"
LOG_PATH = BASE_DIR / "data" / "logs" / "runtime_predictions.csv"

WINDOW_SIZE = 30
PROCESS_EVERY_N_FRAMES = 5

USE_SAFETY_RULE = True
ENABLE_CSV_LOG = True

USE_ARDUINO = False
SERIAL_PORT = "COM3"
BAUD_RATE = 9600


def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera no índice {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
    else:
        model = bundle
        feature_columns = [
            "mean_ear",
            "min_ear",
            "std_ear",
            "perclos",
            "longest_eye_close",
            "mean_mar",
            "max_mar",
            "std_mar",
            "mouth_open_ratio",
        ]

    return model, feature_columns


def init_arduino():
    if not USE_ARDUINO:
        return None

    try:
        import serial

        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)

        print(f"Arduino conectado em {SERIAL_PORT}")

        return arduino

    except Exception as error:
        print(f"Não foi possível conectar ao Arduino: {error}")
        return None


def send_status_to_arduino(arduino, status, last_sent_status):
    if arduino is None:
        return last_sent_status

    if status == last_sent_status:
        return last_sent_status

    try:
        if status == "SONOLENTO":
            arduino.write(b"S")
            print("Enviado para Arduino: S")
            return status

        if status == "NORMAL":
            arduino.write(b"N")
            print("Enviado para Arduino: N")
            return status

    except Exception as error:
        print(f"Erro enviando status ao Arduino: {error}")

    return last_sent_status


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


def extract_roi(frame, landmarks, indexes, padding=20):
    points = np.array([landmarks[i] for i in indexes])

    x_min = int(max(np.min(points[:, 0]) - padding, 0))
    y_min = int(max(np.min(points[:, 1]) - padding, 0))
    x_max = int(min(np.max(points[:, 0]) + padding, frame.shape[1]))
    y_max = int(min(np.max(points[:, 1]) + padding, frame.shape[0]))

    roi = frame[y_min:y_max, x_min:x_max]

    return roi, (x_min, y_min, x_max, y_max)


def draw_box(frame, box, label, color):
    x_min, y_min, x_max, y_max = box

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    cv2.putText(
        frame,
        label,
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_landmark_points(frame, landmarks, indexes, color):
    for idx in indexes:
        if idx < len(landmarks):
            x, y = landmarks[idx]
            cv2.circle(frame, (x, y), 2, color, -1)


def transform_roi_for_visualization(roi):
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 80, 160)

    return {
        "gray": gray,
        "enhanced": enhanced,
        "edges": edges,
    }


def prepare_logs():
    if not ENABLE_CSV_LOG:
        return

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if LOG_PATH.exists():
        return

    columns = [
        "timestamp",
        "status",
        "confidence",
        "mean_ear",
        "min_ear",
        "std_ear",
        "perclos",
        "longest_eye_close",
        "mean_mar",
        "max_mar",
        "std_mar",
        "mouth_open_ratio",
        "brightness",
        "contrast",
        "sharpness",
    ]

    with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)


def log_prediction(status, confidence, features, quality):
    if not ENABLE_CSV_LOG:
        return

    row = [
        datetime.now().isoformat(timespec="seconds"),
        status,
        confidence,
        features.get("mean_ear"),
        features.get("min_ear"),
        features.get("std_ear"),
        features.get("perclos"),
        features.get("longest_eye_close"),
        features.get("mean_mar"),
        features.get("max_mar"),
        features.get("std_mar"),
        features.get("mouth_open_ratio"),
        quality.get("brightness"),
        quality.get("contrast"),
        quality.get("sharpness"),
    ]

    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def apply_safety_rule(features):
    if not USE_SAFETY_RULE:
        return None, None

    if (
        features["mean_ear"] > 0.25
        and features["perclos"] < 0.20
        and features["mouth_open_ratio"] < 0.20
    ):
        return "NORMAL", 0.95

    return None, None


def predict_window(model, feature_columns, features):
    missing_columns = [col for col in feature_columns if col not in features]

    if missing_columns:
        raise ValueError(f"Features ausentes para o modelo: {missing_columns}")

    forced_prediction, forced_confidence = apply_safety_rule(features)

    if forced_prediction is not None:
        return forced_prediction, forced_confidence

    X = pd.DataFrame(
        [[features[col] for col in feature_columns]],
        columns=feature_columns,
    )

    prediction = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)])
    else:
        confidence = 1.0

    return prediction, confidence


def draw_status(frame, status, confidence, ear, mar, fps, window_count, features, quality):
    if status == "SONOLENTO":
        color = (0, 0, 255)
    elif status == "NORMAL":
        color = (0, 255, 0)
    else:
        color = (0, 255, 255)

    cv2.putText(frame, f"Status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, f"Confianca: {confidence:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    if ear is not None:
        cv2.putText(frame, f"EAR atual: {ear:.3f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if mar is not None:
        cv2.putText(frame, f"MAR atual: {mar:.3f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, f"Janela: {window_count}/{WINDOW_SIZE}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if features is not None:
        cv2.putText(frame, f"Mean EAR: {features['mean_ear']:.3f}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"PERCLOS: {features['perclos']:.3f}", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"Mean MAR: {features['mean_mar']:.3f}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"Mouth Ratio: {features['mouth_open_ratio']:.3f}", (20, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.putText(frame, f"Luz: {quality['brightness']:.1f}", (20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Contraste: {quality['contrast']:.1f}", (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Nitidez: {quality['sharpness']:.1f}", (20, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if quality["warnings"]:
        warning_text = " | ".join(quality["warnings"])
        cv2.putText(frame, warning_text, (20, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    cv2.putText(frame, "Pressione Q para sair", (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    print("Carregando modelo...")
    model, feature_columns = load_model()

    print("Features usadas pelo modelo:")
    print(feature_columns)

    prepare_logs()

    arduino = init_arduino()
    last_sent_status = None

    print("Iniciando câmera...")
    cap = open_camera(camera_index=0)

    ear_window = []
    mar_window = []

    status = "CALIBRANDO"
    confidence = 0.0

    last_ear = None
    last_mar = None
    last_features = None
    last_quality = {
        "brightness": 0.0,
        "contrast": 0.0,
        "sharpness": 0.0,
        "warnings": [],
    }

    frame_count = 0
    previous_time = time.time()
    start_time = time.time()

    with create_landmarker() as landmarker:
        while True:
            success, frame = cap.read()

            if not success:
                print("Falha ao capturar frame da câmera.")
                break

            frame_count += 1

            current_time = time.time()
            fps = 1 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))

            quality = analyze_frame_quality(frame)
            last_quality = quality

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                timestamp_ms = int((time.time() - start_time) * 1000)

                landmarks = extract_landmarks_from_frame(
                    landmarker=landmarker,
                    frame=frame,
                    timestamp_ms=timestamp_ms,
                )

                if landmarks is None:
                    status = "ROSTO NAO DETECTADO"
                else:
                    draw_landmark_points(frame, landmarks, LEFT_EYE, (0, 255, 0))
                    draw_landmark_points(frame, landmarks, RIGHT_EYE, (0, 255, 0))
                    draw_landmark_points(frame, landmarks, MOUTH_ROI, (255, 0, 0))

                    try:
                        left_eye_roi, left_box = extract_roi(frame, landmarks, LEFT_EYE, padding=20)
                        right_eye_roi, right_box = extract_roi(frame, landmarks, RIGHT_EYE, padding=20)
                        mouth_roi, mouth_box = extract_roi(frame, landmarks, MOUTH_ROI, padding=25)

                        draw_box(frame, left_box, "Olho E", (0, 255, 0))
                        draw_box(frame, right_box, "Olho D", (0, 255, 0))
                        draw_box(frame, mouth_box, "Boca", (255, 0, 0))

                        mouth_views = transform_roi_for_visualization(mouth_roi)

                        if mouth_views is not None:
                            cv2.imshow("Boca - Canny", mouth_views["edges"])

                    except Exception:
                        pass

                    ear, mar = extract_ear_mar_from_landmarks(landmarks)

                    if ear is not None and mar is not None:
                        last_ear = ear
                        last_mar = mar

                        ear_window.append(ear)
                        mar_window.append(mar)

                        if len(ear_window) > WINDOW_SIZE:
                            ear_window.pop(0)
                            mar_window.pop(0)

                        if len(ear_window) == WINDOW_SIZE:
                            features = calculate_window_features(
                                ear_values=ear_window,
                                mar_values=mar_window,
                            )

                            last_features = features

                            status, confidence = predict_window(
                                model=model,
                                feature_columns=feature_columns,
                                features=features,
                            )

                            last_sent_status = send_status_to_arduino(
                                arduino=arduino,
                                status=status,
                                last_sent_status=last_sent_status,
                            )

                            log_prediction(
                                status=status,
                                confidence=confidence,
                                features=features,
                                quality=quality,
                            )

                            print("\nFeatures da janela:")
                            for key, value in features.items():
                                if isinstance(value, float):
                                    print(f"{key}: {value:.4f}")
                                else:
                                    print(f"{key}: {value}")

                            print(f"Predicao: {status} | Confianca: {confidence:.2f}")

            draw_status(
                frame=frame,
                status=status,
                confidence=confidence,
                ear=last_ear,
                mar=last_mar,
                fps=fps,
                window_count=len(ear_window),
                features=last_features,
                quality=last_quality,
            )

            cv2.imshow("SafeDrive Vision - Webcam", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Encerrando captura...")
                break

    cap.release()

    if arduino is not None:
        arduino.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()