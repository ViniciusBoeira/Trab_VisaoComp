import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# Funções de tempo
def get_timestamp():
    # Retorna o timestamp atual no formato ISO 8601
    return datetime.now().isoformat()


def format_timestamp_display(timestamp_iso):
    # Converte timestamp para formato legível na tela
    try:
        dt = datetime.fromisoformat(timestamp_iso)
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return timestamp_iso

# Funções de imagem
def resize_frame(frame, width=640, height=480):
    # Redimensiona o frame mantendo a proporção se apenas uma dimensão for informada
    return cv2.resize(frame, (width, height))


def to_rgb(frame):
    # Converte frame BGR (OpenCV) para RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def to_gray(frame):
    # Converte frame BGR para escala de cinza
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def normalize_frame(frame):
    # Normaliza os valores de pixel para o intervalo [0, 1]
    return frame.astype(np.float32) / 255.0


def is_valid_frame(frame):
    # Verifica se o frame é válido (não nulo e com dimensões corretas)
    if frame is None:
        return False
    if frame.size == 0:
        return False
    if len(frame.shape) < 2:
        return False
    return True


# Funções de desenho no frame
def draw_status_box(frame, prediction, confidence, method="rules"):
    # Desenha um box colorido no frame com o status de sonolência detectado.
    # Verde = ALERTA 
    # Amarelo = ATENÇÃO 
    # Vermelho = SONOLENTO

    color_map = {
        "ALERTA": (0, 255, 0),
        "ATENÇÃO": (0, 200, 255),
        "SONOLENTO": (0, 0, 255),
    }

    color = color_map.get(prediction, (255, 255, 255))

    # Fundo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Texto do status
    cv2.putText(
        frame,
        f"Status: {prediction}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )

    # Confiança e método
    method_label = "ML" if method == "ml" else "Regras"
    cv2.putText(
        frame,
        f"Confianca: {confidence:.0%}  [{method_label}]",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )

    return frame


def draw_ear_mar(frame, ear, mar):
    # Desenha os valores de EAR e MAR
    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        f"EAR: {ear:.3f}",
        (20, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"MAR: {mar:.3f}",
        (20, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    return frame


def draw_fps(frame, fps):
    # Desenha o FPS no canto superior direito do frame
    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 130, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return frame


def draw_alert_banner(frame, prediction):
    # Desenha um banner de alerta vermelho piscante quando SONOLENTO
    if prediction != "SONOLENTO":
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame,
        "⚠ ALERTA: OPERADOR SONOLENTO!",
        (w // 2 - 250, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    return frame

# Funções de arquivo
def ensure_dir(path):
    # Garante que o diretório existe, criando se necessário
    Path(path).mkdir(parents=True, exist_ok=True)


def is_image_file(path):
    # Verifica se o arquivo é uma imagem suportada
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return Path(path).suffix.lower() in supported


def is_video_file(path):
    # Verifica se o arquivo é um vídeo suportado
    supported = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    return Path(path).suffix.lower() in supported


def load_image(path):
    # Carrega uma imagem do disco e valida se foi carregada corretamente
    frame = cv2.imread(str(path))
    if not is_valid_frame(frame):
        raise ValueError(f"Não foi possível carregar a imagem: {path}")
    return frame


# Funções de métricas
def safe_mean(values):
    # Calcula a média de uma lista, retornando 0.0 se estiver vazia
    if not values:
        return 0.0
    return float(np.mean(values))


def safe_min(values):
    # Retorna o mínimo de uma lista, retornando 0.0 se estiver vazia
    if not values:
        return 0.0
    return float(np.min(values))


def safe_max(values):
    # Retorna o máximo de uma lista, retornando 0.0 se estiver vazia
    if not values:
        return 0.0
    return float(np.max(values))


def clamp(value, min_val=0.0, max_val=1.0):
    # Limita um valor entre min_val e max_val
    return max(min_val, min(max_val, value))