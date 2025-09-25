from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import os

# ------------------- Directories -------------------
pose_dir = "pose_data"             # Folder with extracted keypoints (normal/ and shoplifting/)
tracked_dir = "pose_data/tracked"  # Folder for tracked sequences
os.makedirs(tracked_dir, exist_ok=True)

# ------------------- Initialize DeepSORT tracker -------------------
tracker = DeepSort(max_age=30)

# ------------------- Walk through all subfolders -------------------
for root, dirs, files in os.walk(pose_dir):
    # Skip the 'tracked' folder if already exists
    if "tracked" in root:
        continue

    for file in files:
        if not file.endswith(".npy"):
            continue

        keypoints = np.load(os.path.join(root, file), allow_pickle=True)
        
        # ------------------- Placeholder: DeepSORT tracking -------------------
        # TODO: Replace with real tracking logic
        tracked_keypoints = keypoints  # For now, just copy

        # ------------------- Save tracked keypoints -------------------
        rel_path = os.path.relpath(root, pose_dir)  # normal/ or shoplifting/
        save_folder = os.path.join(tracked_dir, rel_path)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, file)
        np.save(save_path, tracked_keypoints)
        print(f"[INFO] Tracked and saved: {save_path}")

print("✅ Tracking completed for all pose sequences!")
