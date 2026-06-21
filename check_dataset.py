import pandas as pd

from src.config import DATASET_OUTPUT_CSV
from src.temporal_features import FEATURE_COLUMNS


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)


DATASET_PATH = DATASET_OUTPUT_CSV

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

    print("\nValores infinitos/nulos nas features:")
    invalid_values = df[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], pd.NA).isna().sum()
    print(invalid_values)

    print("\nResumo geral das features:")
    print(df[FEATURE_COLUMNS].describe())

    print("\nMédias por classe:")
    print(df.groupby("label")[FEATURE_COLUMNS].mean().T)

    print("\nMedianas por classe:")
    print(df.groupby("label")[FEATURE_COLUMNS].median().T)

    print("\nQuantidade de janelas por vídeo:")
    print(df.groupby(["label", "video_name"]).size())

    if "group_id" in df.columns:
        print("\nQuantidade de janelas por grupo:")
        print(df.groupby(["label", "group_id"]).size())

        print("\nQuantidade de grupos por classe:")
        print(df.groupby("label")["group_id"].nunique())

        print("\nClasses presentes em cada grupo:")
        group_classes = df.groupby("group_id")["label"].unique()

        for group_id, labels in group_classes.items():
            print(f"{group_id}: {list(labels)}")

        paired_groups = [
            group_id
            for group_id, labels in group_classes.items()
            if len(labels) > 1
        ]

        if paired_groups:
            print("\nOK: há grupos com NORMAL e SONOLENTO juntos.")
            print("Isso é bom se representam a mesma pessoa/participante.")
        else:
            print("\nAVISO: nenhum group_id possui mais de uma classe.")
            print("Se normal_001 e drowsy_001 forem a mesma pessoa, revise infer_group_id.")

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

    print("\nCHECK FINAL:")

    if df.empty:
        print("ERRO: dataset vazio.")
    elif df["label"].nunique() < 2:
        print("ERRO: dataset possui menos de 2 classes.")
    elif invalid_values.sum() > 0:
        print("ERRO: existem NaNs ou infinitos nas features.")
    else:
        print("Dataset parece válido para treino.")

    print("\nPróximo passo:")
    print("python .\\train_model.py")


if __name__ == "__main__":
    main()