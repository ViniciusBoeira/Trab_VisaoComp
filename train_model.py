from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import DATASET_OUTPUT_CSV, MODEL_OUTPUT_PATH
from src.temporal_features import FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = DATASET_OUTPUT_CSV


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\nTreinando {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n{name} - Accuracy:")
    print(accuracy)

    print(f"\n{name} - Macro F1:")
    print(macro_f1)

    print(f"\n{name} - Classification report:")
    print(classification_report(y_test, y_pred))

    print(f"\n{name} - Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    if hasattr(model, "feature_importances_"):
        print(f"\n{name} - Feature importances:")
        importances = sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )

        for feature, importance in importances:
            print(f"{feature}: {importance:.4f}")

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def main():
    print("Lendo dataset em:")
    print(DATASET_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"CSV não encontrado em: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print("\nColunas encontradas:")
    print(df.columns.tolist())

    missing_columns = [
        col for col in FEATURE_COLUMNS + ["label"]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Colunas ausentes no dataset: {missing_columns}")

    print("\nShape original:")
    print(df.shape)

    print("\nDistribuição das classes:")
    print(df["label"].value_counts())

    print("\nNaN por coluna:")
    print(df[FEATURE_COLUMNS + ["label"]].isna().sum())

    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    print("\nShape após remover NaN/inf:")
    print(df.shape)

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    if "group_id" in df.columns:
        groups = df["group_id"]
        group_column = "group_id"
    else:
        groups = df["video_name"]
        group_column = "video_name"

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("\nGrupos no treino:")
    print(df.iloc[train_idx][group_column].unique())

    print("\nGrupos no teste:")
    print(df.iloc[test_idx][group_column].unique())

    print("\nDistribuição treino:")
    print(y_train.value_counts())

    print("\nDistribuição teste:")
    print(y_test.value_counts())

    candidates = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=700,
                random_state=42,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]

    results = []

    for name, model in candidates:
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))

    best = max(results, key=lambda item: item["macro_f1"])

    print("\nMelhor modelo pela Macro F1:")
    print(f"{best['name']} | accuracy={best['accuracy']:.4f} | macro_f1={best['macro_f1']:.4f}")

    # Modelo final usado na demo: mesmo algoritmo vencedor, treinado com todas as linhas disponíveis.
    final_model = clone(best["model"])
    final_model.fit(X, y)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": FEATURE_COLUMNS,
            "group_column": group_column,
            "model_name": best["name"],
            "validation_accuracy": best["accuracy"],
            "validation_macro_f1": best["macro_f1"],
        },
        MODEL_OUTPUT_PATH,
    )

    print("\nModelo final salvo em:")
    print(MODEL_OUTPUT_PATH)
    print("\nObservação: o .pkl salvo foi retreinado com TODO o dataset após a validação.")
    print("A métrica honesta é a validação mostrada acima.")


if __name__ == "__main__":
    main()