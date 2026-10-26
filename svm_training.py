
import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.9
)


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / denominator
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


# Extract 12 pose angles from an image
def extract_angles(image):
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    keypoints = [
        (lm.x, lm.y)
        for lm in results.pose_landmarks.landmark
    ]

    L = mp_pose.PoseLandmark

    angles = [
        calculate_angle(
            keypoints[L.LEFT_SHOULDER.value],
            keypoints[L.LEFT_ELBOW.value],
            keypoints[L.LEFT_WRIST.value]
        ),
        calculate_angle(
            keypoints[L.RIGHT_SHOULDER.value],
            keypoints[L.RIGHT_ELBOW.value],
            keypoints[L.RIGHT_WRIST.value]
        ),
        calculate_angle(
            keypoints[L.LEFT_HIP.value],
            keypoints[L.LEFT_KNEE.value],
            keypoints[L.LEFT_ANKLE.value]
        ),
        calculate_angle(
            keypoints[L.RIGHT_HIP.value],
            keypoints[L.RIGHT_KNEE.value],
            keypoints[L.RIGHT_ANKLE.value]
        ),
        calculate_angle(
            keypoints[L.LEFT_SHOULDER.value],
            keypoints[L.LEFT_HIP.value],
            keypoints[L.LEFT_KNEE.value]
        ),
        calculate_angle(
            keypoints[L.RIGHT_SHOULDER.value],
            keypoints[L.RIGHT_HIP.value],
            keypoints[L.RIGHT_KNEE.value]
        ),
        calculate_angle(
            keypoints[L.LEFT_ELBOW.value],
            keypoints[L.LEFT_SHOULDER.value],
            keypoints[L.LEFT_HIP.value]
        ),
        calculate_angle(
            keypoints[L.RIGHT_ELBOW.value],
            keypoints[L.RIGHT_SHOULDER.value],
            keypoints[L.RIGHT_HIP.value]
        ),
        calculate_angle(
            keypoints[L.LEFT_KNEE.value],
            keypoints[L.LEFT_HIP.value],
            keypoints[L.RIGHT_HIP.value]
        ),
        calculate_angle(
            keypoints[L.RIGHT_KNEE.value],
            keypoints[L.RIGHT_HIP.value],
            keypoints[L.LEFT_HIP.value]
        ),
        calculate_angle(
            keypoints[L.LEFT_HIP.value],
            keypoints[L.NOSE.value],
            keypoints[L.RIGHT_HIP.value]
        ),
        calculate_angle(
            keypoints[L.LEFT_SHOULDER.value],
            keypoints[L.NOSE.value],
            keypoints[L.RIGHT_SHOULDER.value]
        )
    ]

    return angles


# Parse VOC XML annotation
def parse_voc_annotation(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        labels = [
            obj.findtext("name")
            for obj in root.findall("object")
            if obj.findtext("name") is not None
        ]

        return labels[0].strip().lower() if labels else None

    except (ET.ParseError, OSError) as error:
        print(f"Could not read annotation {xml_file}: {error}")
        return None


# Dataset paths
dataset_path = "/content/images/all/train"

# Pose classes must match your XML annotation names
pose_labels = {
    "walking": 0,
    "seated": 1,
    "standing": 2
}

# Prepare dataset
dataset = []

for file in os.listdir(dataset_path):
    if not file.lower().endswith(".xml"):
        continue

    xml_path = os.path.join(dataset_path, file)
    label_name = parse_voc_annotation(xml_path)

    if label_name not in pose_labels:
        continue

    # Assumes each XML file has a matching JPG image
    image_stem = os.path.splitext(file)[0]
    image_path = os.path.join(dataset_path, image_stem + ".jpg")

    # Also check uppercase JPG extension
    if not os.path.exists(image_path):
        image_path = os.path.join(dataset_path, image_stem + ".JPG")

    if not os.path.exists(image_path):
        print(f"Image not found for annotation: {file}")
        continue

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
        continue

    angles = extract_angles(image)

    if angles is not None and len(angles) == 12:
        dataset.append(angles + [pose_labels[label_name]])


# Stop if no valid features were extracted
if not dataset:
    pose.close()
    raise ValueError(
        "No pose features extracted. Check dataset paths, XML labels, "
        "image filenames, and MediaPipe pose detection."
    )


# Convert to DataFrame
columns = [
    "elbow_left",
    "elbow_right",
    "knee_left",
    "knee_right",
    "hip_left",
    "hip_right",
    "elbow_shoulder_hip_left",
    "elbow_shoulder_hip_right",
    "knee_hip_left",
    "knee_hip_right",
    "hip_spine",
    "shoulder_spine",
    "label"
]

df = pd.DataFrame(dataset, columns=columns)

# Save extracted features
csv_filename = "pose_features.csv"
df.to_csv(csv_filename, index=False)

print(f"Feature extraction complete: {csv_filename}")
print(f"Total valid samples: {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts().sort_index())


# Prepare features and labels
X = df.iloc[:, :-1].values
y = df["label"].values

unique_classes, class_counts = np.unique(y, return_counts=True)

if len(unique_classes) < 2:
    pose.close()
    raise ValueError(
        "Only one class was extracted. Check your XML annotation labels "
        "and dataset."
    )

# A stratified split requires at least two samples per class
if np.min(class_counts) < 2:
    pose.close()
    raise ValueError(
        "At least two samples per class are required for a stratified "
        "train-test split. Collect more valid images for each class."
    )

print(f"\nFound {len(unique_classes)} classes. Training model...")


# Split before scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Normalize features using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train SVM classifier
svm_classifier = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

svm_classifier.fit(X_train_scaled, y_train)


# Evaluate model
y_pred = svm_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nSVM Pose Classification Accuracy: {accuracy:.2f}")
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(pose_labels.values()),
        target_names=list(pose_labels.keys()),
        zero_division=0
    )
)


# Save model and scaler for Raspberry Pi inference
joblib.dump(svm_classifier, "svm_pose_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved: svm_pose_classifier.pkl")
print("Scaler saved: scaler.pkl")


# Function to predict pose from a new image
def predict_pose(image, model, feature_scaler):
    angles = extract_angles(image)

    if angles is None:
        return None

    if len(angles) != 12:
        return None

    angles_scaled = feature_scaler.transform([angles])
    prediction = model.predict(angles_scaled)[0]

    return prediction


# Test with a new image
test_image_path = (
    "/content/images/all/test/"
    "humanwalkV259_jpg.rf.6c8c4d0d80c551a5b0f01261452acffa.jpg"
)

test_image = cv2.imread(test_image_path)

if test_image is None:
    print(f"\nTest image not found or unreadable: {test_image_path}")
else:
    predicted_pose = predict_pose(
        test_image,
        svm_classifier,
        scaler
    )

    if predicted_pose is not None:
        pose_name = next(
            (
                name
                for name, label in pose_labels.items()
                if label == predicted_pose
            ),
            "Unknown pose"
        )

        print(f"\nPredicted Pose: {pose_name}")
    else:
        print("\nNo pose detected in the test image.")


# Release MediaPipe resources
pose.close()
