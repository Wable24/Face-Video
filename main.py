import os
import cv2
import matplotlib.pyplot as plt

from src.video_reader import read_video
from src.face_detection import detect_faces
from src.rgb_extraction import extract_rgb_signal
from src.rppg import extract_rppg
from src.signal_processing import (
    normalize_signal,
    bandpass_filter
)
from src.features import extract_features


# ============================================================
# VIDEO INPUT
# ============================================================

VIDEO_PATH = r"G:\Live-BP\data\videos\mov.mp4"


def main():

    print("\n===================================")
    print(" VIDEO BASED BP ESTIMATION")
    print("===================================\n")

    # --------------------------------
    # Check video
    # --------------------------------

    print("Checking video...")

    if not os.path.exists(VIDEO_PATH):

        print("\nERROR: Video file not found!")
        print("Expected location:")
        print(VIDEO_PATH)

        return

    print("Video found:")
    print(VIDEO_PATH)

    # --------------------------------
    # 1. Read video
    # --------------------------------

    print("\n1. Reading video...")

    frames, fps = read_video(VIDEO_PATH)

    print(f"FPS: {fps}")
    print(f"Frames: {len(frames)}")

    if len(frames) == 0:

        print("ERROR: No frames found.")

        return

    # --------------------------------
    # 2. Face detection
    # --------------------------------

    print("\n2. Detecting faces...")

    face_boxes = detect_faces(frames)

    detected = sum(
        face is not None
        for face in face_boxes
    )

    print(
        f"Faces detected in "
        f"{detected}/{len(frames)} frames"
    )

    if detected == 0:

        print("\nERROR: No face detected.")

        return

    # --------------------------------
    # 3. Show detected face
    # --------------------------------

    first_face_index = next(
        i
        for i, face in enumerate(face_boxes)
        if face is not None
    )

    display_frame = frames[
        first_face_index
    ].copy()

    face = face_boxes[
        first_face_index
    ]

    x = face["x"]
    y = face["y"]
    w = face["w"]
    h = face["h"]

    # --------------------------------
    # Draw face rectangle
    # --------------------------------

    cv2.rectangle(
        display_frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # --------------------------------
    # Forehead ROI
    # --------------------------------

    roi_x1 = x + int(w * 0.20)
    roi_x2 = x + int(w * 0.80)

    roi_y1 = y + int(h * 0.05)
    roi_y2 = y + int(h * 0.30)

    cv2.rectangle(
        display_frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255, 0, 0),
        2
    )

    # --------------------------------
    # Display
    # --------------------------------

    cv2.imshow(
        "Face Detection + Forehead ROI",
        display_frame
    )

    print("\nFace detected successfully.")
    print("Green = Face")
    print("Blue = Forehead ROI")

    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # --------------------------------
    # 4. RGB extraction
    # --------------------------------

    print("\n3. Extracting RGB signal...")

    rgb_signal = extract_rgb_signal(
        frames,
        face_boxes
    )

    print(
        "RGB signal shape:",
        rgb_signal.shape
    )

    if len(rgb_signal) < 100:

        print(
            "ERROR: Not enough RGB samples."
        )

        return

    # --------------------------------
    # 5. rPPG extraction
    # --------------------------------

    print("\n4. Extracting rPPG signal...")

    rppg_signal = extract_rppg(
        rgb_signal
    )

    # --------------------------------
    # 6. Normalize
    # --------------------------------

    print("\n5. Normalizing signal...")

    normalized = normalize_signal(
        rppg_signal
    )

    # --------------------------------
    # 7. Band-pass filtering
    # --------------------------------

    print("\n6. Filtering signal...")

    filtered = bandpass_filter(
        normalized,
        fps
    )

    # --------------------------------
    # 8. Feature extraction
    # --------------------------------

    print("\n7. Extracting features...")

    features = extract_features(
        filtered,
        fps
    )

    print("\n===================================")
    print(" FEATURES")
    print("===================================")

    for name, value in features.items():

        print(
            f"{name}: {value}"
        )

    # --------------------------------
    # 9. Plot rPPG
    # --------------------------------

    print("\nDisplaying rPPG signal...")

    plt.figure(figsize=(12, 5))

    plt.plot(filtered)

    plt.title(
        "Filtered rPPG Signal"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()