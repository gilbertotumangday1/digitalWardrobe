import cv2
import mediapipe as mp
import json
import os

def generate_pose_keypoints(image_path, output_json_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    height, width, _ = image.shape
    keypoints = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            x = lm.x * width
            y = lm.y * height
            c = lm.visibility  # Confidence/visibility
            keypoints.extend([x, y, c])
    else:
        print("No keypoints detected.")
        return

    pose_data = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": keypoints,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
        ]
    }

    with open(output_json_path, 'w') as f:
        json.dump(pose_data, f, indent=4)
    print(f"Saved keypoints to {output_json_path}")


# ======== Example Usage ========
image_path = "fatperson.jpg"  # Replace this with your image
output_path = "fatperson_keypoints.json"
generate_pose_keypoints(image_path, output_path)
