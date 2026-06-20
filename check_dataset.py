from pathlib import Path

import pandas as pd


print("INICIANDO CHECK_DATASET...")


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "features_all.csv"

FEATURE_COLUMNS = [
    "mean_ear",
    "min_ear",
    "std_ear",
    "perclos",
    "longest_eye_close",
    "mean_mar",
    "max_mar",
    "std_mar",
    "mouth_open_ratio"
]


def main():
    print("Lendo dataset em:")
    print(DATASET_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"CSV não encontrado em: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print("\nColunas:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nDistribuição das classes:")
    print(df["label"].value_counts())

    print("\nNaN por coluna:")
    print(df.isna().sum())

    print("\nResumo geral das features:")
    print(df[FEATURE_COLUMNS].describe())

    print("\nResumo por classe:")
    summary = df.groupby("label")[FEATURE_COLUMNS].agg(
        ["mean", "median", "min", "max", "std"]
    )
    print(summary)

    print("\nQuantidade de janelas por vídeo:")
    print(df.groupby(["label", "video_name"]).size())


if __name__ == "__main__":
    main()