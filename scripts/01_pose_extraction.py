import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Directories
video_dir = "../videos"       # Folder containing your MP4 videos
output_dir = "../pose_data"   # Folder to save extracted keypoints
os.makedirs(output_dir, exist_ok=True)

# Loop over videos
for vid_file in os.listdir(video_dir):
    if not vid_file.endswith(".mp4"):
        continue
    cap = cv2.VideoCapture(os.path.join(video_dir, vid_file))
    all_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            keypoints = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
            all_frames.append(keypoints)

    # Save keypoints as a NumPy file
    np.save(os.path.join(output_dir, vid_file.replace(".mp4", ".npy")), all_frames)
    cap.release()

pose.close()
print("Pose extraction completed!")
