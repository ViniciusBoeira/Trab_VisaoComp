import numpy as np

from src.config import EAR_THRESHOLD, MAR_THRESHOLD


FEATURE_COLUMNS = [
    # Olhos / EAR
    "mean_ear",
    "min_ear",
    "max_ear",
    "std_ear",
    "ear_q10",
    "ear_q25",
    "ear_q75",
    "ear_q90",
    "ear_range",
    "ear_slope",

    # Fechamento dos olhos
    "perclos",
    "longest_eye_close",
    "longest_eye_close_seconds",
    "mean_eye_close_duration",
    "eye_close_events",
    "eye_close_event_rate",

    # Boca / MAR
    "mean_mar",
    "min_mar",
    "max_mar",
    "std_mar",
    "mar_q75",
    "mar_q90",
    "mar_q95",
    "mar_range",
    "mar_slope",

    # Abertura da boca
    "mouth_open_ratio",
    "mouth_open_events",
    "longest_mouth_open",
    "longest_mouth_open_seconds",
    "mean_mouth_open_duration",

    # Combinações derivadas
    "eye_mouth_ratio",
    "fatigue_score",
    "yawn_score",
]


def count_events(flags):
    # Conta quantas sequências True existem em um vetor booleano.
    count = 0
    inside_event = False

    for flag in flags:
        if flag and not inside_event:
            count += 1
            inside_event = True
        elif not flag:
            inside_event = False

    return count


def streak_lengths(flags):
    # Retorna os tamanhos de todas as sequências consecutivas True.
    lengths = []
    current = 0

    for flag in flags:
        if flag:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
            current = 0

    if current > 0:
        lengths.append(current)

    return lengths


def calculate_slope(values):
    # Inclinação linear simples da sequência dentro da janela.
    values = np.array(values, dtype=float)

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, 1)[0]

    return float(slope)


def safe_mean(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def calculate_window_features(ear_values, mar_values, samples_per_second=3.0):
    
    # Transforma uma janela temporal de EAR/MAR em features numéricas.

    # samples_per_second é usado para converter durações de eventos
    # de quantidade de medições para segundos aproximados.
    

    ear_values = np.array(ear_values, dtype=float)
    mar_values = np.array(mar_values, dtype=float)

    if len(ear_values) == 0 or len(mar_values) == 0:
        return None

    eps = 1e-6
    samples_per_second = max(float(samples_per_second), eps)
    window_duration_seconds = max(len(ear_values) / samples_per_second, eps)

    closed_flags = ear_values < EAR_THRESHOLD
    mouth_open_flags = mar_values > MAR_THRESHOLD

    eye_streaks = streak_lengths(closed_flags)
    mouth_streaks = streak_lengths(mouth_open_flags)

    longest_eye_close = max(eye_streaks) if eye_streaks else 0
    longest_mouth_open = max(mouth_streaks) if mouth_streaks else 0

    mean_eye_close = safe_mean(eye_streaks)
    mean_mouth_open = safe_mean(mouth_streaks)

    eye_close_events = count_events(closed_flags)
    mouth_open_events = count_events(mouth_open_flags)

    mean_ear = float(np.mean(ear_values))
    mean_mar = float(np.mean(mar_values))

    features = {
        # Olhos / EAR
        "mean_ear": mean_ear,
        "min_ear": float(np.min(ear_values)),
        "max_ear": float(np.max(ear_values)),
        "std_ear": float(np.std(ear_values)),
        "ear_q10": float(np.percentile(ear_values, 10)),
        "ear_q25": float(np.percentile(ear_values, 25)),
        "ear_q75": float(np.percentile(ear_values, 75)),
        "ear_q90": float(np.percentile(ear_values, 90)),
        "ear_range": float(np.max(ear_values) - np.min(ear_values)),
        "ear_slope": calculate_slope(ear_values),

        # Fechamento dos olhos
        "perclos": float(np.mean(closed_flags)),
        "longest_eye_close": int(longest_eye_close),
        "longest_eye_close_seconds": float(longest_eye_close / samples_per_second),
        "mean_eye_close_duration": float(mean_eye_close / samples_per_second),
        "eye_close_events": int(eye_close_events),
        "eye_close_event_rate": float(eye_close_events / window_duration_seconds),

        # Boca / MAR
        "mean_mar": mean_mar,
        "min_mar": float(np.min(mar_values)),
        "max_mar": float(np.max(mar_values)),
        "std_mar": float(np.std(mar_values)),
        "mar_q75": float(np.percentile(mar_values, 75)),
        "mar_q90": float(np.percentile(mar_values, 90)),
        "mar_q95": float(np.percentile(mar_values, 95)),
        "mar_range": float(np.max(mar_values) - np.min(mar_values)),
        "mar_slope": calculate_slope(mar_values),

        # Abertura da boca
        "mouth_open_ratio": float(np.mean(mouth_open_flags)),
        "mouth_open_events": int(mouth_open_events),
        "longest_mouth_open": int(longest_mouth_open),
        "longest_mouth_open_seconds": float(longest_mouth_open / samples_per_second),
        "mean_mouth_open_duration": float(mean_mouth_open / samples_per_second),

        # Combinações derivadas
        "eye_mouth_ratio": float(mean_ear / (mean_mar + eps)),
        "fatigue_score": float(np.mean(closed_flags) * longest_eye_close),
        "yawn_score": float(np.max(mar_values) * np.mean(mouth_open_flags)),
    }

    return features