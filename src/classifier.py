import numpy as np
import joblib
from pathlib import Path


# Caminho do modelo treinado
MODEL_PATH = Path("models/drowsiness_model.pkl")

# Limiares para classificação por regras fixas

# PERCLOS: % de frames com olho fechado
PERCLOS_ALERT = 0.15       # acima disso = ATENÇÃO
PERCLOS_DROWSY = 0.30      # acima disso = SONOLENTO

# EAR médio da janela
EAR_ALERT = 0.22           # abaixo disso = ATENÇÃO
EAR_DROWSY = 0.18          # abaixo disso = SONOLENTO

# MAR máximo da janela (bocejo)
MAR_ALERT = 0.5            # acima disso = ATENÇÃO
MAR_DROWSY = 0.7           # acima disso = SONOLENTO

# Bocejos na janela
YAWN_ALERT = 1             # 1 bocejo = ATENÇÃO
YAWN_DROWSY = 3            # 3+ bocejos = SONOLENTO

# Inclinação da cabeça em graus
HEAD_PITCH_ALERT = 15      # acima disso = ATENÇÃO
HEAD_PITCH_DROWSY = 25     # acima disso = SONOLENTO


# Mapeamento de label para nível de risco
RISK_LEVEL = {
    "ALERTA": "LOW",
    "ATENÇÃO": "MEDIUM",
    "SONOLENTO": "HIGH",
}


def classify_by_rules(features):
    # Classifica o estado do operador usando regras fixas baseadas nos limiares.
    # Retorna: (prediction, risk_level, confidence)
    # prediction: "ALERTA", "ATENÇÃO" ou "SONOLENTO"

    ear_mean = features.get("ear_mean", 1.0)
    mar_max = features.get("mar_max", 0.0)
    perclos = features.get("perclos", 0.0)
    yawn_count = features.get("yawn_count", 0)
    head_pitch = abs(features.get("head_pitch", 0.0))

    # Conta quantos indicadores apontam para cada nível
    drowsy_score = 0
    alert_score = 0

    if perclos >= PERCLOS_DROWSY:
        drowsy_score += 2      # PERCLOS tem peso maior
    elif perclos >= PERCLOS_ALERT:
        alert_score += 1

    if ear_mean <= EAR_DROWSY:
        drowsy_score += 2
    elif ear_mean <= EAR_ALERT:
        alert_score += 1

    if mar_max >= MAR_DROWSY:
        drowsy_score += 1
    elif mar_max >= MAR_ALERT:
        alert_score += 1

    if yawn_count >= YAWN_DROWSY:
        drowsy_score += 1
    elif yawn_count >= YAWN_ALERT:
        alert_score += 1

    if head_pitch >= HEAD_PITCH_DROWSY:
        drowsy_score += 1
    elif head_pitch >= HEAD_PITCH_ALERT:
        alert_score += 1

    # Decisão final baseada nos scores
    if drowsy_score >= 2:
        prediction = "SONOLENTO"
        confidence = min(0.95, 0.60 + drowsy_score * 0.07)
    elif alert_score >= 2 or drowsy_score == 1:
        prediction = "ATENÇÃO"
        confidence = min(0.90, 0.55 + alert_score * 0.08)
    else:
        prediction = "ALERTA"
        confidence = min(0.95, 0.70 + (5 - alert_score) * 0.05)

    risk_level = RISK_LEVEL[prediction]
    return prediction, risk_level, round(confidence, 3)


def load_ml_model():
    # Carrega o modelo ML treinado do disco.
    # Retorna None se o modelo ainda não existir.

    if not MODEL_PATH.exists():
        return None

    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo ML: {e}")
        return None


def classify_by_model(model, features):
    # Classifica usando o modelo ML treinado.
    # Retorna: (prediction, risk_level, confidence)

    # Ordem das features deve bater com o treino no train_model.py
    feature_vector = np.array([[
        features.get("ear_mean", 0.0),
        features.get("ear_min", 0.0),
        features.get("mar_max", 0.0),
        features.get("perclos", 0.0),
        features.get("blink_count", 0),
        features.get("yawn_count", 0),
        features.get("head_pitch", 0.0),
    ]])

    prediction_raw = model.predict(feature_vector)[0]

    # Tenta pegar a probabilidade se o modelo suportar
    try:
        proba = model.predict_proba(feature_vector)[0]
        confidence = float(np.max(proba))
    except Exception:
        confidence = 0.75

    # Garante que o label está no formato esperado
    label_map = {
        0: "ALERTA",
        1: "ATENÇÃO",
        2: "SONOLENTO",
        "ALERTA": "ALERTA",
        "ATENÇÃO": "ATENÇÃO",
        "SONOLENTO": "SONOLENTO",
    }

    prediction = label_map.get(prediction_raw, "ALERTA")
    risk_level = RISK_LEVEL[prediction]

    return prediction, risk_level, round(confidence, 3)


class Classifier:
    def __init__(self):
        # Tenta carregar o modelo ML; se não existir, usa só regras fixas
        self.model = load_ml_model()

        if self.model:
            print("Modelo ML carregado — usando ML + regras como fallback.")
        else:
            print("Modelo ML não encontrado — usando apenas regras fixas.")

    def classify(self, features):
        # Classifica o estado do operador.
        # Usa ML se o modelo estiver disponível, senão usa regras fixas.
        # Retorna: (prediction, risk_level, confidence, method)

        if self.model is not None:
            try:
                prediction, risk_level, confidence = classify_by_model(self.model, features)
                method = "ml"
                return prediction, risk_level, confidence, method
            except Exception as e:
                print(f"Erro na classificação ML, usando regras fixas: {e}")

        prediction, risk_level, confidence = classify_by_rules(features)
        method = "rules"
        return prediction, risk_level, confidence, method

    def reload_model(self):
        # Recarrega o modelo do disco sem reiniciar o worker.
        # Útil após treinar um novo modelo com train_model.py.

        self.model = load_ml_model()
        if self.model:
            print("Modelo ML recarregado com sucesso.")
        else:
            print("Nenhum modelo encontrado após reload.")
