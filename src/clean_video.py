import cv2
import os

INPUT_VIDEO = r"G:\Live-BP\data\videos\mov.mp4"
OUTPUT_VIDEO = r"G:\Live-BP\data\videos\cleaned_video_60sec.mp4"

if not os.path.exists(INPUT_VIDEO):
    raise FileNotFoundError(
        f"Input video not found:\n{INPUT_VIDEO}"
    )

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise ValueError("Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Exactly 60 seconds
MAX_FRAMES = int(fps * 60)

print("Input video:")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Processing:", MAX_FRAMES, "frames")
print("Duration: 60 seconds")

# Output resolution
output_width = 640
output_height = int(height * output_width / width)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (output_width, output_height)
)

if not out.isOpened():
    cap.release()
    raise ValueError("Could not create output video.")

frame_count = 0

while frame_count < MAX_FRAMES:

    ret, frame = cap.read()

    if not ret:
        print("Video ended before 60 seconds.")
        break

    # Resize
    frame = cv2.resize(
        frame,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA
    )

    # Mild denoising
    frame = cv2.GaussianBlur(
        frame,
        (3, 3),
        0
    )

    # Mild contrast enhancement
    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    frame = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # Save frame
    out.write(frame)

    frame_count += 1

    if frame_count % 500 == 0:
        print(
            f"Processed {frame_count}/{MAX_FRAMES} frames"
        )

cap.release()
out.release()

duration = frame_count / fps

print("\nCleaning completed!")
print("Frames processed:", frame_count)
print("Output duration:", round(duration, 2), "seconds")
print("Output video:")
print(OUTPUT_VIDEO)