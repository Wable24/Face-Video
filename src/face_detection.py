import cv2
import mediapipe as mp


# ============================================================
# MediaPipe Face Detection
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ------------------------------------------------------------
# Create detector
# ------------------------------------------------------------

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path="models/blaze_face_short_range.tflite"
    ),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5
)

detector = FaceDetector.create_from_options(options)


# ============================================================
# Detect faces
# ============================================================

def detect_faces(frames):

    face_boxes = []

    for frame in frames:

        height, width = frame.shape[:2]

        # OpenCV BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect
        result = detector.detect(mp_image)

        if result.detections:

            detection = result.detections[0]

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height

            # Keep coordinates inside frame
            x = max(0, x)
            y = max(0, y)

            w = min(w, width - x)
            h = min(h, height - y)

            face_boxes.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

        else:

            face_boxes.append(None)

    return face_boxes