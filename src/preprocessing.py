import cv2
import numpy as np

# Funções de melhoria de iluminação
def equalize_histogram(frame):
    # Equaliza o histograma do frame para melhorar contraste em ambientes escuros.
    # Funciona em escala de cinza — converte, equaliza e volta para BGR.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


def equalize_clahe(frame, clip_limit=2.0, tile_size=(8, 8)):
    # Equalização adaptativa por regiões (CLAHE).
    # Melhor que a equalização global — evita superexposição em áreas já claras.
    # Recomendada para ambientes com iluminação irregular.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Aplica CLAHE apenas no canal L (luminância)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_brightness_contrast(frame, brightness=0, contrast=1.0):
    # Ajusta brilho e contraste do frame manualmente.
    # brightness: valor inteiro entre -100 e 100
    # contrast: float, 1.0 = sem alteração, >1.0 aumenta contraste
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    return adjusted


def gamma_correction(frame, gamma=1.2):
    # Aplica correção gama para clarear frames escuros sem estourar os claros.
    # gamma < 1.0 = escurece, gamma > 1.0 = clareia
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)

    return cv2.LUT(frame, table)

# Funções de filtragem e suavização
def apply_gaussian_blur(frame, kernel_size=(5, 5)):
    # Aplica blur gaussiano para reduzir ruído no frame.
    # Útil antes de detectar bordas ou em câmeras de baixa qualidade.
    return cv2.GaussianBlur(frame, kernel_size, 0)


def apply_bilateral_filter(frame, d=9, sigma_color=75, sigma_space=75):
    # Filtro bilateral — suaviza o frame preservando as bordas.
    # Mais lento que o gaussiano, mas mantém detalhes faciais importantes.
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)


def apply_sharpening(frame):
    # Aplica um filtro de nitidez para realçar detalhes do rosto.
    # Útil quando a câmera está levemente desfocada.
    kernel = np.array([
        [0, -1,  0],
        [-1,  5, -1],
        [0, -1,  0]
    ])
    return cv2.filter2D(frame, -1, kernel)

# Funções de ROI
def extract_face_roi(frame, landmarks_points, padding=20):
    # Extrai a região do rosto a partir dos landmarks do MediaPipe.
    # Adiciona um padding ao redor para não cortar partes do rosto.
    if landmarks_points is None or len(landmarks_points) == 0:
        return frame

    x_coords = landmarks_points[:, 0]
    y_coords = landmarks_points[:, 1]

    x_min = max(0, int(np.min(x_coords)) - padding)
    y_min = max(0, int(np.min(y_coords)) - padding)
    x_max = min(frame.shape[1], int(np.max(x_coords)) + padding)
    y_max = min(frame.shape[0], int(np.max(y_coords)) + padding)

    return frame[y_min:y_max, x_min:x_max]


def extract_eye_roi(frame, eye_points, padding=10):
    # Extrai a região de um olho a partir dos seus 6 pontos landmarks.
    # Retorna None se a região for inválida.
    if eye_points is None or len(eye_points) == 0:
        return None

    x_min = max(0, int(np.min(eye_points[:, 0])) - padding)
    y_min = max(0, int(np.min(eye_points[:, 1])) - padding)
    x_max = min(frame.shape[1], int(np.max(eye_points[:, 0])) + padding)
    y_max = min(frame.shape[0], int(np.max(eye_points[:, 1])) + padding)

    roi = frame[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return None

    return roi

# Pipeline de pré-processamento
def preprocess_frame(frame, use_clahe=True, use_bilateral=False):
    # Pipeline principal de pré-processamento aplicado a cada frame.
    # Ordem: redimensiona -> melhora iluminação -> suaviza ruído
    # use_clahe: recomendado para ambientes com iluminação variável
    # use_bilateral: melhor qualidade mas mais lento (desativa em fps baixo)

    if frame is None:
        return None

    # Redimensiona para resolução padrão de processamento
    frame = cv2.resize(frame, (640, 480))

    # Melhoria de iluminação
    if use_clahe:
        frame = equalize_clahe(frame)
    else:
        frame = gamma_correction(frame, gamma=1.2)

    # Suavização de ruído
    if use_bilateral:
        frame = apply_bilateral_filter(frame)
    else:
        frame = apply_gaussian_blur(frame, kernel_size=(3, 3))

    return frame


def preprocess_for_display(frame):
    # Pré-processamento leve apenas para exibição na tela.
    # Não aplica filtros pesados para manter o FPS alto.
    if frame is None:
        return None

    frame = cv2.resize(frame, (640, 480))
    frame = equalize_clahe(frame, clip_limit=1.5)

    return frame