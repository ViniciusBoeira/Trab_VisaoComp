from pathlib import Path
import time
import csv
from datetime import datetime

import cv2
import joblib
import pandas as pd

from src.config import (
    BAUD_RATE,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    ENABLE_CSV_LOG,
    MODEL_OUTPUT_PATH,
    PROCESS_EVERY_N_FRAMES,
    RUNTIME_LOG_PATH,
    RUNTIME_WINDOW_SIZE,
    SERIAL_PORT,
    USE_ARDUINO,
    USE_SAFETY_RULE,
)
from src.facial_metrics import (
    LEFT_EYE,
    RIGHT_EYE,
    MOUTH_ROI,
    extract_ear_mar_from_landmarks,
)
from src.image_processing import (
    analyze_frame_quality,
    extract_roi,
    preprocess_frame_for_landmarks,
    transform_roi_for_visualization,
)
from src.mediapipe_detector import create_landmarker, extract_landmarks_from_frame
from src.temporal_features import FEATURE_COLUMNS, calculate_window_features


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = MODEL_OUTPUT_PATH
LOG_PATH = RUNTIME_LOG_PATH
WINDOW_SIZE = RUNTIME_WINDOW_SIZE
USE_PREPROCESS_FOR_LANDMARKS = False


def open_camera(camera_index=CAMERA_INDEX):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera no índice {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    return cap


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        model_name = bundle.get("model_name", "modelo")
        validation_accuracy = bundle.get("validation_accuracy")
        validation_macro_f1 = bundle.get("validation_macro_f1")
    else:
        model = bundle
        feature_columns = FEATURE_COLUMNS
        model_name = "modelo_antigo"
        validation_accuracy = None
        validation_macro_f1 = None

    print(f"Modelo carregado: {model_name}")

    if validation_accuracy is not None:
        print(f"Accuracy de validação: {validation_accuracy:.4f}")

    if validation_macro_f1 is not None:
        print(f"Macro F1 de validação: {validation_macro_f1:.4f}")

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

def read_arduino_event(arduino):
    if arduino is None:
        return None

    try:
        if arduino.in_waiting > 0:
            return arduino.read().decode("utf-8", errors="ignore")
    except Exception as error:
        print(f"Erro lendo Arduino: {error}")

    return None

def draw_box(frame, box, label, color):
    x_min, y_min, x_max, y_max = box

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(frame, label, (x_min, max(y_min - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_landmark_points(frame, landmarks, indexes, color):
    for idx in indexes:
        if idx < len(landmarks):
            x, y = landmarks[idx]
            cv2.circle(frame, (x, y), 2, color, -1)


def prepare_logs(feature_columns):
    if not ENABLE_CSV_LOG:
        return

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if LOG_PATH.exists():
        return

    columns = [
        "timestamp",
        "status",
        "confidence",
        "decision_source",
        *feature_columns,
        "brightness",
        "contrast",
        "sharpness",
    ]

    with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)


def log_prediction(status, confidence, decision_source, features, quality, feature_columns):
    if not ENABLE_CSV_LOG:
        return

    row = [
        datetime.now().isoformat(timespec="seconds"),
        status,
        confidence,
        decision_source,
        *[features.get(col) for col in feature_columns],
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
        return forced_prediction, forced_confidence, "REGRA"

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

    return prediction, confidence, "MODELO"


def draw_status(frame, status, confidence, decision_source, ear, mar, fps, window_count, features, quality):
    if status == "SONOLENTO":
        color = (0, 0, 255)
    elif status == "NORMAL":
        color = (0, 255, 0)
    else:
        color = (0, 255, 255)

    cv2.putText(frame, f"Status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, f"Confianca: {confidence:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"Fonte: {decision_source}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if ear is not None:
        cv2.putText(frame, f"EAR atual: {ear:.3f}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if mar is not None:
        cv2.putText(frame, f"MAR atual: {mar:.3f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    cv2.putText(frame, f"Janela: {window_count}/{WINDOW_SIZE}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if features is not None:
        cv2.putText(frame, f"Mean EAR: {features['mean_ear']:.3f}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.putText(frame, f"PERCLOS: {features['perclos']:.3f}", (20, 263), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.putText(frame, f"Eye events: {features['eye_close_events']}", (20, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.putText(frame, f"Mean MAR: {features['mean_mar']:.3f}", (20, 309), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.putText(frame, f"Yawn score: {features['yawn_score']:.3f}", (20, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    cv2.putText(frame, f"Luz: {quality['brightness']:.1f}", (20, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    cv2.putText(frame, f"Contraste: {quality['contrast']:.1f}", (20, 388), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    cv2.putText(frame, f"Nitidez: {quality['sharpness']:.1f}", (20, 411), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    if quality["warnings"]:
        warning_text = " | ".join(quality["warnings"])
        cv2.putText(frame, warning_text, (20, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2)

    cv2.putText(frame, "Pressione Q para sair", (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main():
    print("Carregando modelo...")
    model, feature_columns = load_model()

    print("Features usadas pelo modelo:")
    print(feature_columns)

    print(f"Safety rule ativa? {USE_SAFETY_RULE}")
    print(f"Janela temporal aproximada: {WINDOW_SIZE} medições")
    print(f"Processa 1 frame a cada {PROCESS_EVERY_N_FRAMES} frames")

    prepare_logs(feature_columns)

    arduino = init_arduino()
    last_sent_status = None
    cooldown_until = 0

    print("Iniciando câmera...")
    cap = open_camera(camera_index=CAMERA_INDEX)

    ear_window = []
    mar_window = []

    status = "CALIBRANDO"
    confidence = 0.0
    decision_source = "-"

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

    samples_per_second = max(CAMERA_FPS / PROCESS_EVERY_N_FRAMES, 1e-6)

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
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

            quality = analyze_frame_quality(frame)
            last_quality = quality

            arduino_event = read_arduino_event(arduino)

            if arduino_event == "B":
                print("Botão pressionado no Arduino. Pausando alertas por 10 segundos.")

                cooldown_until = time.time() + 10

                ear_window.clear()
                mar_window.clear()

                last_sent_status = None
                status = "PAUSA POS-ALERTA"
                decision_source = "ARDUINO"

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                timestamp_ms = int((time.time() - start_time) * 1000)

                frame_for_landmarks = preprocess_frame_for_landmarks(
                    frame=frame,
                    quality=quality,
                    enabled=USE_PREPROCESS_FOR_LANDMARKS,
                )

                landmarks = extract_landmarks_from_frame(
                    landmarker=landmarker,
                    frame=frame_for_landmarks,
                    timestamp_ms=timestamp_ms,
                )

                if landmarks is None:
                    status = "ROSTO NAO DETECTADO"
                    decision_source = "-"
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
                                samples_per_second=samples_per_second,
                            )

                            last_features = features

                            status, confidence, decision_source = predict_window(
                                model=model,
                                feature_columns=feature_columns,
                                features=features,
                            )

                            in_cooldown = time.time() < cooldown_until

                            if not in_cooldown:
                                last_sent_status = send_status_to_arduino(
                                    arduino=arduino,
                                    status=status,
                                    last_sent_status=last_sent_status,
                                )
                            else:
                                print("Em cooldown pós-alerta. Não enviando status ao Arduino.")

                            log_prediction(
                                status=status,
                                confidence=confidence,
                                decision_source=decision_source,
                                features=features,
                                quality=quality,
                                feature_columns=feature_columns,
                            )

                            print("\nFeatures principais da janela:")
                            for key in [
                                "mean_ear",
                                "min_ear",
                                "perclos",
                                "longest_eye_close_seconds",
                                "eye_close_events",
                                "mean_mar",
                                "max_mar",
                                "mouth_open_ratio",
                                "yawn_score",
                            ]:
                                value = features.get(key)
                                if isinstance(value, float):
                                    print(f"{key}: {value:.4f}")
                                else:
                                    print(f"{key}: {value}")

                            print(f"Predicao: {status} | Confianca: {confidence:.2f} | Fonte: {decision_source}")

            draw_status(
                frame=frame,
                status=status,
                confidence=confidence,
                decision_source=decision_source,
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
