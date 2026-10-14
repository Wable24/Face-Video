import numpy as np


def extract_rgb_signal(frames, face_boxes):

    rgb_signal = []

    for frame, face in zip(frames, face_boxes):

        if face is None:
            continue

        x = face["x"]
        y = face["y"]
        w = face["w"]
        h = face["h"]

        # Make sure coordinates stay inside image
        height, width, _ = frame.shape

        x = max(0, x)
        y = max(0, y)

        x2 = min(width, x + w)
        y2 = min(height, y + h)

        face_crop = frame[y:y2, x:x2]

        if face_crop.size == 0:
            continue

        # --------------------------------
        # Forehead ROI
        # --------------------------------

        face_height = face_crop.shape[0]

        forehead = face_crop[
            int(face_height * 0.05):
            int(face_height * 0.30),
            int(face_crop.shape[1] * 0.20):
            int(face_crop.shape[1] * 0.80)
        ]

        if forehead.size == 0:
            continue

        # OpenCV = BGR
        b, g, r = np.mean(
            forehead,
            axis=(0, 1)
        )

        rgb_signal.append([r, g, b])

    return np.array(rgb_signal)