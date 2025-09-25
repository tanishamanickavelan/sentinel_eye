import cv2
import mediapipe as mp
import numpy as np
import os

# ------------------- Initialize MediaPipe Pose -------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# ------------------- Directories -------------------
video_dir = "videos"        # Folder containing 'normal/' and 'shoplifting/' subfolders
output_dir = "pose_data"    # Folder to save extracted keypoints
os.makedirs(output_dir, exist_ok=True)

# ------------------- Walk through all subfolders -------------------
for root, dirs, files in os.walk(video_dir):
    for vid_file in files:
        if not vid_file.endswith(".mp4"):
            continue

        print(f"[INFO] Processing: {vid_file}")
        cap = cv2.VideoCapture(os.path.join(root, vid_file))
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

        # ------------------- Save keypoints -------------------
        rel_path = os.path.relpath(root, video_dir)  # 'normal' or 'shoplifting'
        save_folder = os.path.join(output_dir, rel_path)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, vid_file.replace(".mp4", ".npy"))
        np.save(save_path, all_frames)

        cap.release()
        print(f"[INFO] Saved: {save_path}")

pose.close()
print("✅ Pose extraction completed for all videos!")