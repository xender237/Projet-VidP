import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Impossible d'ouvrir la vidéo")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Vidéo chargée : {fps} FPS, {width}x{height}")

    return cap, fps, (width, height)


if __name__ == "__main__":
    video_path = "data/input_video.mp4"
    cap, fps, size = read_video(video_path)

    # Lecture test de quelques frames
    count = 0
    while cap.isOpened() and count < 5:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Frame", frame)
        cv2.waitKey(200)
        count += 1

    cap.release()
    cv2.destroyAllWindows()

