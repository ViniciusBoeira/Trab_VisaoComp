import cv2
import time


def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera no índice {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap


def main():
    print("Iniciando câmera...")

    cap = open_camera(camera_index=0)

    previous_time = time.time()

    while True:
        success, frame = cap.read()

        if not success:
            print("Falha ao capturar frame da câmera.")
            break

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Espelha a imagem para ficar mais natural na visualização
        frame = cv2.flip(frame, 1)

        # Aqui futuramente entra:
        # 1. Pré-processamento com OpenCV
        # 2. MediaPipe
        # 3. Extração de features
        # 4. Modelo de Machine Learning
        # 5. Registro de eventos no banco/CSV

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SafeDrive Vision - Pressione Q para sair",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("SafeDrive Vision - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Encerrando captura...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()