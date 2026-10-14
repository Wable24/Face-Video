import numpy as np


def extract_forehead_roi(frame, bbox):

    if bbox is None:
        return None

    height, width, _ = frame.shape

    x = int(bbox.xmin * width)
    y = int(bbox.ymin * height)

    w = int(bbox.width * width)
    h = int(bbox.height * height)

    # Keep coordinates inside image
    x = max(0, x)
    y = max(0, y)

    x2 = min(width, x + w)
    y2 = min(height, y + h)

    face = frame[y:y2, x:x2]

    if face.size == 0:
        return None

    face_height = face.shape[0]

    # Upper 25% of face
    forehead = face[
        0:int(face_height * 0.25),
        :
    ]

    return forehead