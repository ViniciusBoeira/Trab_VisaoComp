import numpy as np
from scipy.spatial.distance import euclidean

# Limiar abaixo do qual considera-se o olho fechado
EAR_CLOSED_THRESHOLD = 0.20

# Limiar acima do qual considera-se a boca aberta (bocejo)
MAR_OPEN_THRESHOLD = 0.6

# Número mínimo de frames com olho fechado para contar como piscada
BLINK_CONSEC_FRAMES = 2

# Número mínimo de frames com boca aberta para contar como bocejo
YAWN_CONSEC_FRAMES = 15


def calculate_ear(eye_points):
    # Calcula o Eye Aspect Ratio (EAR) de um olho.
    # EAR = (dist vertical 1 + dist vertical 2) / (2 * dist horizontal)
    # Quanto menor o EAR, mais fechado está o olho.
    # eye_points: array de 6 pontos (x, y)

    # Distâncias verticais
    vertical_1 = euclidean(eye_points[1], eye_points[5])
    vertical_2 = euclidean(eye_points[2], eye_points[4])

    # Distância horizontal
    horizontal = euclidean(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def calculate_mar(mouth_points):
    # Calcula o Mouth Aspect Ratio (MAR).
    # Quanto maior o MAR, mais aberta está a boca — indica bocejo.
    # mouth_points: array de 8 pontos da boca

    # Distâncias verticais internas da boca
    vertical_1 = euclidean(mouth_points[2], mouth_points[6])
    vertical_2 = euclidean(mouth_points[3], mouth_points[7])
    vertical_3 = euclidean(mouth_points[4], mouth_points[5])

    # Distância horizontal (canto esquerdo ao canto direito)
    horizontal = euclidean(mouth_points[0], mouth_points[1])

    if horizontal == 0:
        return 0.0

    mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
    return float(mar)


def calculate_perclos(ear_history):
    # Calcula o PERCLOS: porcentagem de frames em que os olhos estiveram fechados.
    # É um dos indicadores mais confiáveis de sonolência.
    # ear_history: lista de valores EAR dos últimos N frames

    if len(ear_history) == 0:
        return 0.0

    closed_frames = sum(1 for ear in ear_history if ear < EAR_CLOSED_THRESHOLD)
    perclos = closed_frames / len(ear_history)
    return float(perclos)


class FeatureExtractor:
    def __init__(self):
        # Histórico de EAR para calcular PERCLOS (janela de 90 frames ≈ 3 segundos a 30fps)
        self.ear_history = []
        self.ear_window_size = 90

        # Contadores de piscada
        self.blink_counter = 0       # frames consecutivos com olho fechado
        self.blink_count = 0         # total de piscadas detectadas

        # Contadores de bocejo
        self.yawn_counter = 0        # frames consecutivos com boca aberta
        self.yawn_count = 0          # total de bocejos detectados

        # Flags de estado
        self.eye_closed = False
        self.mouth_open = False

    def update(self, left_eye, right_eye, mouth_points, head_pitch):
        # Processa os pontos de um frame e retorna todas as features calculadas.
        # Deve ser chamado a cada frame capturado.

        # Calcula EAR de cada olho e tira a média
        ear_left = calculate_ear(left_eye)
        ear_right = calculate_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0

        # Calcula MAR
        mar = calculate_mar(mouth_points)

        # Atualiza histórico de EAR para PERCLOS
        self.ear_history.append(ear)
        if len(self.ear_history) > self.ear_window_size:
            self.ear_history.pop(0)

        # Calcula PERCLOS na janela atual
        perclos = calculate_perclos(self.ear_history)

        # Detecta piscadas
        if ear < EAR_CLOSED_THRESHOLD:
            self.blink_counter += 1
            self.eye_closed = True
        else:
            if self.eye_closed and self.blink_counter >= BLINK_CONSEC_FRAMES:
                self.blink_count += 1
            self.blink_counter = 0
            self.eye_closed = False

        # Detecta bocejos
        if mar > MAR_OPEN_THRESHOLD:
            self.yawn_counter += 1
            self.mouth_open = True
        else:
            if self.mouth_open and self.yawn_counter >= YAWN_CONSEC_FRAMES:
                self.yawn_count += 1
            self.yawn_counter = 0
            self.mouth_open = False

        # Monta e retorna o dicionário de features do frame atual
        features = {
            "ear": ear,
            "ear_left": ear_left,
            "ear_right": ear_right,
            "mar": mar,
            "perclos": perclos,
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "head_pitch": head_pitch,
            "eye_closed": self.eye_closed,
            "mouth_open": self.mouth_open,
        }

        return features

    def get_window_summary(self):
        # Retorna um resumo das features na janela atual.
        # Usado para gerar eventos periódicos no banco de dados.

        if len(self.ear_history) == 0:
            return None

        summary = {
            "ear_mean": float(np.mean(self.ear_history)),
            "ear_min": float(np.min(self.ear_history)),
            "perclos": calculate_perclos(self.ear_history),
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
        }

        return summary

    def reset_counts(self):
        # Zera os contadores de piscada e bocejo.
        # Útil para resetar a cada novo evento salvo no banco.

        self.blink_count = 0
        self.yawn_count = 0
        self.ear_history.clear()