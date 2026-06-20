from pathlib import Path
import shutil
import hashlib


BASE_DIR = Path(__file__).resolve().parent

# Pasta onde estão os vídeos bagunçados que você baixou
SOURCE_DIR = BASE_DIR / "data" / "downloads" / "uta"

# Pastas finais usadas pelo feature_extractor
NORMAL_DIR = BASE_DIR / "data" / "raw" / "uta" / "normal"
DROWSY_DIR = BASE_DIR / "data" / "raw" / "uta" / "drowsiness"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def short_hash(path: Path):
    text = str(path).encode("utf-8")
    return hashlib.md5(text).hexdigest()[:8]


def detect_label(video_path: Path):
    """
    Detecta label pelo nome do arquivo ou pelas pastas.
    Ajustado para UTA:
    0  -> NORMAL
    10 -> SONOLENTO
    """

    path_text = str(video_path).lower()
    stem = video_path.stem.lower()

    # Detecta por pasta
    if "normal" in path_text or "alert" in path_text:
        return "NORMAL"

    if "drows" in path_text or "sonol" in path_text:
        return "SONOLENTO"

    # Detecta por nome do arquivo
    # exemplos: 0.mp4, 0 (1).mp4, pessoa_0.mp4
    if stem.startswith("0"):
        return "NORMAL"

    if stem.startswith("10"):
        return "SONOLENTO"

    return None


def make_unique_filename(video_path: Path, label: str, index: int):
    """
    Gera nomes únicos para evitar colisão.
    Usa label + contador + hash do caminho original.
    """

    label_prefix = "normal" if label == "NORMAL" else "drowsy"
    file_hash = short_hash(video_path)

    return f"{label_prefix}_{index:03d}_{file_hash}{video_path.suffix.lower()}"


def main():
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    DROWSY_DIR.mkdir(parents=True, exist_ok=True)

    videos = [
        file for file in SOURCE_DIR.rglob("*")
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
    ]

    print(f"Vídeos encontrados: {len(videos)}")

    normal_count = 0
    drowsy_count = 0
    skipped = []

    for video in videos:
        label = detect_label(video)

        if label is None:
            skipped.append(video)
            continue

        if label == "NORMAL":
            normal_count += 1
            new_name = make_unique_filename(video, label, normal_count)
            destination = NORMAL_DIR / new_name

        else:
            drowsy_count += 1
            new_name = make_unique_filename(video, label, drowsy_count)
            destination = DROWSY_DIR / new_name

        shutil.copy2(video, destination)

        print(f"Copiado: {video} -> {destination}")

    print("\nFinalizado!")
    print(f"NORMAL: {normal_count}")
    print(f"SONOLENTO: {drowsy_count}")

    if skipped:
        print("\nArquivos ignorados porque não consegui detectar label:")
        for file in skipped:
            print(f" - {file}")


if __name__ == "__main__":
    main()