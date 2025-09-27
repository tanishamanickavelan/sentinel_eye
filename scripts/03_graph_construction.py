from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import os

# ------------------- Directories -------------------
<<<<<<< HEAD
tracked_dir = "pose_data/tracked"  # Folder with tracked keypoints (normal/ and shoplifting/)
graph_dir = "graphs"               # Folder to save ST-GCN graphs
os.makedirs(graph_dir, exist_ok=True)

# ------------------- Loop through all subfolders -------------------
for root, dirs, files in os.walk(tracked_dir):
    # Skip the root if empty
    for file in files:
        if not file.endswith(".npy"):
            continue

        keypoints_seq = np.load(os.path.join(root, file), allow_pickle=True)
        
        # ------------------- Example: create adjacency matrix -------------------
        num_joints = 33  # BlazePose
        adj_matrix = np.zeros((num_joints, num_joints))

        # Define edges (example: extend for full skeleton)
        edges = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6)]
        for i, j in edges:
            adj_matrix[i,j] = adj_matrix[j,i] = 1

        # ------------------- Save graph -------------------
        rel_path = os.path.relpath(root, tracked_dir)  # normal/ or shoplifting/
        save_folder = os.path.join(graph_dir, rel_path)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, file.replace(".npy","_graph.npy"))
        np.save(save_path, {"keypoints": keypoints_seq, "adj": adj_matrix})

        print(f"[INFO] Graph saved: {save_path}")

print("✅ Graph construction completed for all tracked sequences!")
=======
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
>>>>>>> 8f18dc003d78d752d07fb4075641556818283497
