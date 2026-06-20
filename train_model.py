from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "data" / "processed" / "features_all.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "random_forest_drowsiness.pkl"

FEATURE_COLUMNS = [
    "mean_ear",
    "min_ear",
    "std_ear",
    "perclos",
    "longest_eye_close",
    "mean_mar",
    "max_mar",
    "std_mar",
    "mouth_open_ratio",
]


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

    print("\nShape após remover NaN:")
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

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        class_weight="balanced",
    )

    print("\nTreinando Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nFeature importances:")
    for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "group_column": group_column,
        },
        MODEL_OUTPUT_PATH,
    )

    print(f"\nModelo salvo em:")
    print(MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()