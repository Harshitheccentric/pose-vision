import sys
import os
import cv2
import math
from ultralytics import YOLO
import yaml

# ---------------- Helper Functions ----------------
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def classify_posture(keypoints):
    try:
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        l_ankle = keypoints[15]
        r_ankle = keypoints[16]
        nose = keypoints[0]

        shoulder_mid = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        hip_mid = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
        ankle_mid = [(l_ankle[0] + r_ankle[0]) / 2, (l_ankle[1] + r_ankle[1]) / 2]

        torso_len = euclidean(shoulder_mid, hip_mid)
        full_body_height = euclidean(nose, ankle_mid)
        vertical_leg = (abs(l_hip[1] - l_ankle[1]) + abs(r_hip[1] - r_ankle[1])) / 2
        total_height = torso_len + vertical_leg
        # leg_ratio = vertical_leg / total_height
        # compression = torso_len / full_body_height

        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        torso_angle = abs(math.degrees(math.atan2(dy, dx)))

        if 60 <= torso_angle < 100:
            return "Standing"
        elif 100 <= torso_angle < 130:
            return "Sitting"
        elif torso_angle >= 130:
            return "Lying/Fallen"
        else:
            return "Unknown"
    except:
        return "Unknown"

# ---------------- Main Pipeline ----------------
if len(sys.argv) != 5:
    print("Usage: python fused_pipeline.py yolov8n-pose.pt best.pt ppe.yaml input.mp4")
    sys.exit(1)

pose_model_path = sys.argv[1]
ppe_model_path = sys.argv[2]
yaml_path = sys.argv[3]
video_path = sys.argv[4]

# Load models
pose_model = YOLO(pose_model_path)
ppe_model = YOLO(ppe_model_path)

# Load YAML class names for PPE model
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)
    label_map = {int(k): v for k, v in data["names"].items()} if isinstance(data["names"], dict) else {i: v for i, v in enumerate(data["names"])}
pass

# Setup video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

input_filename = os.path.basename(video_path)
output_path = os.path.join('assets\outputs', input_filename)
os.makedirs(os.path.join('assets', 'outputs'), exist_ok=True)
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pose_results = pose_model(frame, verbose=False)
    ppe_results = ppe_model(frame, verbose=False)
    annotated = frame.copy()

    # Draw PPE detections
    for box in ppe_results[0].boxes:
        cls = int(box.cls[0])
        label = label_map.get(cls, str(cls))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if "NO-" in label:   
            color = (0, 0, 255)     # Red
        else:
            color = (0, 255, 0)     # Green

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw keypoints and posture detection
    for kp in pose_results[0].keypoints:
        keypoints = kp.xy[0].tolist()
        for x, y in keypoints:
            cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)

        posture = classify_posture(keypoints)
        label = posture

        try:
            x_label, y_label = int(keypoints[11][0]), int(keypoints[11][1]) - 10
        except:
            x_label, y_label = 20, 30

        cv2.putText(annotated, label, (x_label, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(annotated)
    cv2.imshow("PPE + Pose Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
