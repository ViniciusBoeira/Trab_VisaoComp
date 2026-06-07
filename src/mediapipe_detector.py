import cv2
import mediapipe as mp
import numpy as np


# Índices dos landmarks do MediaPipe Face Mesh
# Olho esquerdo (da perspectiva do observador)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Olho direito (da perspectiva do observador)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Boca (pontos externos + internos verticais)
MOUTH = [61, 291, 13, 14, 17, 0, 78, 308]

# Pontos para orientação da cabeça (pitch = inclinação pra frente/trás)
NOSE_TIP = 1
CHIN = 152
LEFT_EAR_POINT = 234
RIGHT_EAR_POINT = 454
LEFT_FOREHEAD = 70
RIGHT_FOREHEAD = 300


class FaceDetector:
    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame):
        # Recebe um frame BGR (OpenCV) e retorna os landmarks detectados.
        # Retorna None se nenhum rosto for encontrado.

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        return results.multi_face_landmarks[0]  # Pega o primeiro rosto detectado

    def get_landmarks_array(self, landmarks, frame_shape):
        # Converte os landmarks normalizados do MediaPipe em coordenadas de pixel (x, y).
        # Retorna um array numpy de shape (478, 2).
        
        h, w = frame_shape[:2]
        points = []

        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        return np.array(points, dtype=np.float32)

    def get_eye_points(self, points):
        # Retorna os 6 pontos de cada olho como arrays numpy.

        left_eye = points[LEFT_EYE]
        right_eye = points[RIGHT_EYE]
        return left_eye, right_eye

    def get_mouth_points(self, points):
        # Retorna os pontos da boca como array numpy.
        
        return points[MOUTH]

    def get_head_pitch(self, points):
        # Calcula o ângulo de inclinação vertical da cabeça (pitch).
        # Positivo = cabeça inclinada para baixo (sonolência).
        # Retorna o ângulo em graus.
       
        nose = points[NOSE_TIP]
        chin = points[CHIN]

        # Vetor do nariz ao queixo
        dx = chin[0] - nose[0]
        dy = chin[1] - nose[1]

        # Ângulo em relação ao eixo vertical
        angle = np.degrees(np.arctan2(dx, dy))
        return float(angle)

    def draw_landmarks(self, frame, landmarks):
        # Desenha os pontos faciais no frame para debug/visualização.

        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    def draw_eye_contours(self, frame, points):
        # Desenha contornos simples ao redor dos olhos (útil para debug do EAR).
    
        left_eye, right_eye = self.get_eye_points(points)

        left_hull = cv2.convexHull(left_eye.astype(np.int32))
        right_hull = cv2.convexHull(right_eye.astype(np.int32))

        cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)

    def close(self):
        self.face_mesh.close()