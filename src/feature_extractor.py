import re
import time
from pathlib import Path

import cv2
import pandas as pd

from src.config import (
    BASE_DIR,
    DATASET_CONFIG,
    DATASET_OUTPUT_CSV,
    DEBUG_SHOW,
    FRAME_SKIP,
    MAX_WINDOWS_PER_VIDEO,
    VALID_VIDEO_EXTENSIONS,
    WINDOW_SECONDS,
)
from src.facial_metrics import (
    LEFT_EYE,
    RIGHT_EYE,
    MOUTH_ROI,
    extract_ear_mar_from_landmarks,
)
from src.mediapipe_detector import create_landmarker, extract_landmarks_from_frame
from src.temporal_features import calculate_window_features


__all__ = [
    "create_landmarker",
    "extract_landmarks_from_frame",
    "extract_ear_mar_from_landmarks",
    "calculate_window_features",
    "LEFT_EYE",
    "RIGHT_EYE",
    "MOUTH_ROI",
]


def infer_group_id(video_path):
    """
    Extrai o ID do participante assumindo o padrão:
    estado_NUMERO_hash (ex: drowsy_018_aa93590e.mov -> group_018)
    """
    video_path = Path(video_path)
    stem = video_path.stem.lower().strip() 
    parts = stem.split("_")

    if len(parts) >= 2:
        return f"group_{parts[1]}"

    return f"group_{stem}"


def get_video_files(folder):
    folder = Path(folder)

    print(f"Procurando vídeos em: {folder}")
    print(f"Pasta existe? {folder.exists()}")

    if not folder.exists():
        return []

    files = [
        file
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in VALID_VIDEO_EXTENSIONS
    ]

    files = sorted(files, key=lambda item: str(item).lower())

    print("Arquivos encontrados:")
    for file in files[:10]:
        print(f" - {file}")

    if len(files) > 10:
        print(f" ... mais {len(files) - 10} arquivos")

    return files


def draw_points(frame, landmarks, indexes, color):
    for idx in indexes:
        if idx < len(landmarks):
            x, y = landmarks[idx]
            cv2.circle(frame, (x, y), 2, color, -1)


def draw_debug(frame, landmarks, ear, mar, label):
    draw_points(frame, landmarks, LEFT_EYE, (0, 255, 0))
    draw_points(frame, landmarks, RIGHT_EYE, (0, 255, 0))
    draw_points(frame, landmarks, MOUTH_ROI, (255, 0, 0))

    cv2.putText(frame, f"Label: {label}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"EAR: {ear:.3f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"MAR: {mar:.3f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def process_video(video_path, label, source):
    start = time.time()

    print(f"\nProcessando: {video_path}")
    print(f"Label: {label}")

    rows = []
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERRO: não conseguiu abrir o vídeo: {video_path}")
        return rows

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 30

    samples_per_second = max(fps / FRAME_SKIP, 1e-6)
    window_size = int(WINDOW_SECONDS * samples_per_second)

    if window_size <= 0:
        window_size = 30

    print(f"FPS original: {fps:.2f}")
    print(f"Amostras por segundo: {samples_per_second:.2f}")
    print(f"Tamanho da janela: {window_size} medições")

    ear_window = []
    mar_window = []

    frame_count = 0
    total_processed = 0
    detected_frames = 0
    window_index = 0

    window_frame_start = None
    window_frame_end = None
    group_id = infer_group_id(video_path)

    with create_landmarker() as landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % FRAME_SKIP != 0:
                continue

            total_processed += 1
            timestamp_ms = int((frame_count / fps) * 1000)

            frame = cv2.resize(frame, (480, 360))

            landmarks = extract_landmarks_from_frame(
                landmarker=landmarker,
                frame=frame,
                timestamp_ms=timestamp_ms,
            )

            if landmarks is None:
                continue

            detected_frames += 1

            ear, mar = extract_ear_mar_from_landmarks(landmarks)

            if ear is None or mar is None:
                continue

            if window_frame_start is None:
                window_frame_start = frame_count

            window_frame_end = frame_count

            ear_window.append(ear)
            mar_window.append(mar)

            if len(ear_window) == window_size:
                features = calculate_window_features(
                    ear_values=ear_window,
                    mar_values=mar_window,
                    samples_per_second=samples_per_second,
                )

                if features is not None:
                    window_index += 1

                    features["label"] = label
                    features["source"] = source
                    features["video_name"] = str(video_path.relative_to(BASE_DIR))
                    features["group_id"] = group_id
                    features["window_index"] = window_index
                    features["frame_start"] = window_frame_start
                    features["frame_end"] = window_frame_end
                    features["fps"] = float(fps)
                    features["samples_per_second"] = float(samples_per_second)
                    features["window_seconds"] = float(WINDOW_SECONDS)

                    rows.append(features)

                ear_window = []
                mar_window = []
                window_frame_start = None
                window_frame_end = None

                if MAX_WINDOWS_PER_VIDEO is not None and window_index >= MAX_WINDOWS_PER_VIDEO:
                    print(f"Limite de {MAX_WINDOWS_PER_VIDEO} janelas atingido para este vídeo.")
                    break

            if DEBUG_SHOW:
                debug_frame = frame.copy()
                draw_debug(debug_frame, landmarks, ear, mar, label)
                cv2.imshow("debug", debug_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()

    if DEBUG_SHOW:
        cv2.destroyAllWindows()

    elapsed = time.time() - start

    print(f"Frames lidos: {frame_count}")
    print(f"Frames processados: {total_processed}")
    print(f"Frames com rosto detectado: {detected_frames}")
    print(f"Janelas geradas: {len(rows)}")
    print(f"Group ID: {group_id}")
    print(f"Tempo do vídeo: {elapsed / 60:.2f} minutos")

    return rows


def main():
    DATASET_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for config in DATASET_CONFIG:
        videos = get_video_files(config["path"])

        print(f"\nTotal de vídeos para {config['label']}: {len(videos)}")

        for video_path in videos:
            try:
                rows = process_video(
                    video_path=video_path,
                    label=config["label"],
                    source=config["source"],
                )

                all_rows.extend(rows)

                df_partial = pd.DataFrame(all_rows)
                df_partial.to_csv(DATASET_OUTPUT_CSV, index=False)

                print(f"Checkpoint salvo com {len(df_partial)} linhas.")

            except Exception as error:
                print(f"ERRO ao processar {video_path}: {error}")
                continue

    df = pd.DataFrame(all_rows)
    df.to_csv(DATASET_OUTPUT_CSV, index=False)

    print("\nExtração finalizada!")
    print(f"Linhas geradas: {len(df)}")
    print(f"CSV salvo em: {DATASET_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
