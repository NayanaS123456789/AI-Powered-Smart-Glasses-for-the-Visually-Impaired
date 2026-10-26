
import cv2
import joblib
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

# Load the trained SVM model and scaler
svm_classifier = joblib.load(
    "/home/piaand/Downloads/svm_pose_classifier (4) (1).pkl"
)

scaler = joblib.load(
    "/home/piaand/Downloads/scaler (4) (1).pkl"
)

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

# Pose labels dictionary
pose_labels = {
    0: "Standing",
    1: "Sitting",
    2: "Walking"
}


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / denominator
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def extract_angles(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = results.pose_landmarks.landmark

        angles = [
            calculate_angle(
                keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value],
                keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value],
                keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                keypoints[mp_pose.PoseLandmark.NOSE.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
            ),

            calculate_angle(
                keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.NOSE.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            )
        ]

        return angles

    return None


def predict_pose(image):
    angles = extract_angles(image)

    if angles is not None:
        if len(angles) != 12:
            print(
                f"Error: Extracted {len(angles)} features, expected 12."
            )
            return None

        angles_scaled = scaler.transform([angles])

        prediction = svm_classifier.predict(angles_scaled)[0]

        return pose_labels.get(prediction, "Unknown Pose")

    return None


# Initialize Picamera2
picam2 = Picamera2()

picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 480)}
    )
)

picam2.start()

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Convert RGB to BGR for OpenCV display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect pose landmarks
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Predict pose
            predicted_pose = predict_pose(frame)

            if predicted_pose:
                cv2.putText(
                    frame,
                    predicted_pose,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Pose Not Detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

        else:
            cv2.putText(
                frame,
                "No pose landmarks detected",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Display output
        cv2.imshow("Pose Classification", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    pose.close()
    cv2.destroyAllWindows()
