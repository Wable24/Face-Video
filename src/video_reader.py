import cv2
import os


def get_video_info(video_path):

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    duration = total_frames / fps if fps > 0 else 0

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": duration
    }


def read_video_frames(video_path, max_frames=None):

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_read = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            yield frame, fps

            frames_read += 1

            if (
                max_frames is not None
                and frames_read >= max_frames
            ):
                break

    finally:

        cap.release()