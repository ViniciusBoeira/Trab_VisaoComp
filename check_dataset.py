from pathlib import Path

import pandas as pd


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)


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
    "mouth_open_ratio",
]

REQUIRED_COLUMNS = FEATURE_COLUMNS + [
    "label",
    "source",
    "video_name",
]


def main():
    print("INICIANDO CHECK_DATASET...")
    print("Lendo dataset em:")
    print(DATASET_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"CSV não encontrado em: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print("\nColunas encontradas:")
    print(df.columns.tolist())

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")

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

    print("\nMédias por classe:")
    print(df.groupby("label")[FEATURE_COLUMNS].mean().T)

    print("\nMedianas por classe:")
    print(df.groupby("label")[FEATURE_COLUMNS].median().T)

    print("\nResumo por classe:")
    summary = df.groupby("label")[FEATURE_COLUMNS].agg(
        ["mean", "median", "min", "max", "std"]
    )
    print(summary)

    print("\nQuantidade de janelas por vídeo:")
    print(df.groupby(["label", "video_name"]).size())

    if "group_id" in df.columns:
        print("\nQuantidade de janelas por grupo:")
        print(df.groupby(["label", "group_id"]).size())

        print("\nQuantidade de grupos por classe:")
        print(df.groupby("label")["group_id"].nunique())

        print("\nGrupos existentes:")
        print(sorted(df["group_id"].unique()))

        print("\nClasses presentes em cada grupo:")
        group_classes = df.groupby("group_id")["label"].unique()

        for group_id, labels in group_classes.items():
            print(f"{group_id}: {list(labels)}")

        groups_with_multiple_classes = [
            group_id
            for group_id, labels in group_classes.items()
            if len(labels) > 1
        ]

        if groups_with_multiple_classes:
            print("\nOK: grupos com mais de uma classe encontrados.")
            print("Isso é bom se o mesmo participante tem vídeo NORMAL e SONOLENTO.")
            print(groups_with_multiple_classes)
        else:
            print("\nAVISO: nenhum group_id aparece com mais de uma classe.")
            print("Isso não é necessariamente erro, mas confira se group_id está representando a pessoa/participante corretamente.")

    else:
        print("\nAVISO: coluna group_id não encontrada.")
        print("O treino poderá usar video_name como grupo, mas group_id seria melhor para evitar vazamento por pessoa.")

    if "window_index" in df.columns:
        print("\nJanelas por vídeo, usando window_index:")
        print(
            df.groupby(["label", "video_name"])["window_index"]
            .agg(["min", "max", "count"])
        )

    if "frame_start" in df.columns and "frame_end" in df.columns:
        print("\nFrames inicial/final por vídeo:")
        print(
            df.groupby(["label", "video_name"])[["frame_start", "frame_end"]]
            .agg(["min", "max"])
        )

    print("\nChecagem de duplicatas completas:")
    duplicated_rows = df.duplicated().sum()
    print(f"Linhas duplicadas: {duplicated_rows}")

    print("\nChecagem de valores infinitos/nulos nas features:")
    invalid_values = df[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], pd.NA).isna().sum()
    print(invalid_values)

    print("\nCHECK FINAL:")

    if df.empty:
        print("ERRO: dataset vazio.")
    elif df["label"].nunique() < 2:
        print("ERRO: dataset possui menos de 2 classes.")
    elif df[FEATURE_COLUMNS].isna().sum().sum() > 0:
        print("ERRO: existem NaNs nas features.")
    else:
        print("Dataset parece válido para treino.")

    print("\nPróximo passo:")
    print("python .\\train_model.py")


if __name__ == "__main__":
    main()