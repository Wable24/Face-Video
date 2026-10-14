import cv2
import os

VIDEO_PATH = r"G:\Live-BP\data\videos\mov.mp4"

print("Checking video...")
print("Path:", VIDEO_PATH)
print("Exists:", os.path.exists(VIDEO_PATH))

if not os.path.exists(VIDEO_PATH):
    print("ERROR: Video file does not exist.")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nVideo opened successfully!")
print("FPS:", fps)
print("Total frames:", total_frames)
print("Resolution:", width, "x", height)

count = 0

while count < 100:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame:", count)
        break

    count += 1

print("\nFrames successfully read:", count)

cap.release()

print("\nTest completed.")