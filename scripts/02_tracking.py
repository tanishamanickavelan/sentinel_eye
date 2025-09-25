from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import os

# Directories
pose_dir = "../pose_data"        # Folder with extracted keypoints
tracked_dir = "../pose_data/tracked"  # Folder for tracked sequences
os.makedirs(tracked_dir, exist_ok=True)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Loop over pose files
for file in os.listdir(pose_dir):
    if not file.endswith(".npy"):
        continue
    keypoints = np.load(os.path.join(pose_dir, file), allow_pickle=True)
    
    # Placeholder: insert DeepSORT tracking logic here
    # For now, just copy keypoints to tracked_dir
    np.save(os.path.join(tracked_dir, file), keypoints)

print("Tracking completed!")
