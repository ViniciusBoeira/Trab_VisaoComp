import numpy as np


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

MOUTH_ROI = [61, 291, 13, 14, 78, 308, 82, 312, 87, 317]


def euclidean_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def calculate_ear(eye_points):
    
    #EAR = Eye Aspect Ratio.
    #Quanto menor o EAR, mais fechado o olho tende a estar.

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return None

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks):
    
    # MAR = Mouth Aspect Ratio.
    # Quanto maior o MAR, mais aberta a boca tende a estar.
    
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]

    vertical = euclidean_distance(top, bottom)
    horizontal = euclidean_distance(left, right)

    if horizontal == 0:
        return None

    return vertical / horizontal


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