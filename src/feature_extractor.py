import os

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import re
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "features_all.csv"

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

FRAME_SKIP = 10
WINDOW_SECONDS = 10
MAX_WINDOWS_PER_VIDEO = 20
DEBUG_SHOW = False

EAR_THRESHOLD = 0.15
MAR_THRESHOLD = 0.12

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

MOUTH_ROI = [61, 291, 13, 14, 78, 308, 82, 312, 87, 317]


BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode


def create_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}\n"
            "Baixe o face_landmarker.task e coloque dentro da pasta models/."
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


def infer_group_id(video_name):
    """
    Ajuda no treino para evitar vazamento.

    Exemplos:
    0 (1).mp4  -> group_1
    10 (1).mp4 -> group_1
    0.mp4      -> group_base_mp4
    10.mp4     -> group_base_mp4
    0.mov      -> group_base_mov
    10.mov     -> group_base_mov

    A ideia é manter vídeos pareados da mesma pessoa no mesmo grupo.
    """
    stem = Path(video_name).stem
    suffix = Path(video_name).suffix.lower().replace(".", "")

    match = re.search(r"\((\d+)\)", stem)

    if match:
        return f"group_{match.group(1)}"

    cleaned = stem.strip()

    cleaned = re.sub(r"^(0|5|10)", "", cleaned).strip()
    cleaned = cleaned.replace(" ", "_")

    if cleaned:
        return f"group_{cleaned}_{suffix}"

    return f"group_base_{suffix}"


def get_video_files(folder):
    folder = Path(folder)

    print(f"Procurando vídeos em: {folder}")
    print(f"Pasta existe? {folder.exists()}")

    if not folder.exists():
        return []

    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    files = [
        file
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in valid_extensions
    ]

    files = sorted(files, key=lambda item: item.name)

    print("Arquivos encontrados:")
    for file in files[:10]:
        print(f" - {file}")

    if len(files) > 10:
        print(f" ... mais {len(files) - 10} arquivos")

    return files


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return None

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks):
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]

    vertical = euclidean_distance(top, bottom)
    horizontal = euclidean_distance(left, right)

    if horizontal == 0:
        return None

    return vertical / horizontal


def extract_landmarks_from_frame(landmarker, frame, timestamp_ms):
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


def extract_ear_mar_from_landmarks(landmarks):
    try:
        left_eye_points = [landmarks[i] for i in LEFT_EYE]
        right_eye_points = [landmarks[i] for i in RIGHT_EYE]

        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        mar = calculate_mar(landmarks)

        if left_ear is None or right_ear is None or mar is None:
            return None, None

        ear = (left_ear + right_ear) / 2.0

        return ear, mar

    except IndexError:
        return None, None


def longest_streak(flags):
    max_streak = 0
    current = 0

    for flag in flags:
        if flag:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return max_streak


def calculate_window_features(ear_values, mar_values):
    ear_values = np.array(ear_values)
    mar_values = np.array(mar_values)

    if len(ear_values) == 0 or len(mar_values) == 0:
        return None

    closed_flags = ear_values < EAR_THRESHOLD
    mouth_open_flags = mar_values > MAR_THRESHOLD

    features = {
        "mean_ear": float(np.mean(ear_values)),
        "min_ear": float(np.min(ear_values)),
        "std_ear": float(np.std(ear_values)),

        "perclos": float(np.mean(closed_flags)),
        "longest_eye_close": int(longest_streak(closed_flags)),

        "mean_mar": float(np.mean(mar_values)),
        "max_mar": float(np.max(mar_values)),
        "std_mar": float(np.std(mar_values)),

        "mouth_open_ratio": float(np.mean(mouth_open_flags)),
    }

    return features


def draw_points(frame, landmarks, indexes, color):
    for idx in indexes:
        if idx < len(landmarks):
            x, y = landmarks[idx]
            cv2.circle(frame, (x, y), 2, color, -1)


def draw_debug(frame, landmarks, ear, mar, label):
    draw_points(frame, landmarks, LEFT_EYE, (0, 255, 0))
    draw_points(frame, landmarks, RIGHT_EYE, (0, 255, 0))
    draw_points(frame, landmarks, MOUTH_ROI, (255, 0, 0))

    cv2.putText(
        frame,
        f"Label: {label}",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"EAR: {ear:.3f}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"MAR: {mar:.3f}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )


def process_video(video_path, label, source):
    start = time.time()

    print(f"\nProcessando: {video_path}")
    print(f"Label: {label}")

    rows = []

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERRO: não conseguiu abrir o vídeo: {video_path}")
        return rows

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 30

    effective_fps = fps / FRAME_SKIP
    window_size = int(WINDOW_SECONDS * effective_fps)

    if window_size <= 0:
        window_size = 30

    print(f"FPS original: {fps:.2f}")
    print(f"FPS efetivo: {effective_fps:.2f}")
    print(f"Tamanho da janela: {window_size} medições")

    ear_window = []
    mar_window = []

    frame_count = 0
    total_processed = 0
    detected_frames = 0
    window_index = 0

    window_frame_start = None
    window_frame_end = None

    group_id = infer_group_id(video_path.name)

    with create_landmarker() as landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % FRAME_SKIP != 0:
                continue

            total_processed += 1

            frame = cv2.resize(frame, (480, 360))

            timestamp_ms = int((frame_count / fps) * 1000)

            landmarks = extract_landmarks_from_frame(
                landmarker=landmarker,
                frame=frame,
                timestamp_ms=timestamp_ms,
            )

            if landmarks is None:
                continue

            ear, mar = extract_ear_mar_from_landmarks(landmarks)

            if ear is None or mar is None:
                continue

            detected_frames += 1

            if not ear_window:
                window_frame_start = frame_count

            window_frame_end = frame_count

            ear_window.append(ear)
            mar_window.append(mar)

            if DEBUG_SHOW:
                draw_debug(frame, landmarks, ear, mar, label)
                cv2.imshow("Debug Feature Extraction", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if len(ear_window) >= window_size:
                features = calculate_window_features(ear_window, mar_window)

                if features is not None:
                    features["label"] = label
                    features["source"] = source
                    features["video_name"] = video_path.name
                    features["group_id"] = group_id
                    features["window_index"] = window_index
                    features["frame_start"] = window_frame_start
                    features["frame_end"] = window_frame_end
                    rows.append(features)

                    window_index += 1

                ear_window = []
                mar_window = []
                window_frame_start = None
                window_frame_end = None

                if len(rows) >= MAX_WINDOWS_PER_VIDEO:
                    break

    cap.release()

    if DEBUG_SHOW:
        cv2.destroyAllWindows()

    elapsed = time.time() - start

    print(f"Frames lidos: {frame_count}")
    print(f"Frames processados: {total_processed}")
    print(f"Frames com rosto detectado: {detected_frames}")
    print(f"Janelas geradas: {len(rows)}")
    print(f"Tempo do vídeo: {elapsed / 60:.2f} minutos")
    print(f"Group ID: {group_id}")

    return rows


def main():
    all_rows = []

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    for config in DATASET_CONFIG:
        videos = get_video_files(config["path"])

        print("\n============================")
        print(f"Fonte: {config['source']}")
        print(f"Pasta: {config['path']}")
        print(f"Label: {config['label']}")
        print(f"Vídeos encontrados: {len(videos)}")
        print("============================")

        for video_path in videos:
            try:
                rows = process_video(
                    video_path=video_path,
                    label=config["label"],
                    source=config["source"],
                )

                all_rows.extend(rows)

                df_partial = pd.DataFrame(all_rows)
                df_partial.to_csv(OUTPUT_CSV, index=False)

                print(f"Checkpoint salvo com {len(df_partial)} linhas.")

            except Exception as error:
                print(f"ERRO ao processar {video_path}: {error}")
                continue

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nCSV gerado!")
    print(f"Arquivo: {OUTPUT_CSV}")

    if not df.empty:
        print("\nPrimeiras linhas:")
        print(df.head())

        print("\nDistribuição das classes:")
        print(df["label"].value_counts())

        if "group_id" in df.columns:
            print("\nDistribuição por grupo:")
            print(df.groupby(["label", "group_id"]).size())
    else:
        print("ATENÇÃO: CSV vazio. Nenhuma janela foi gerada.")


if __name__ == "__main__":
    main()