from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import os

# ------------------- Directories -------------------
pose_dir = "../pose_data"             # Folder with extracted keypoints
tracked_dir = "../pose_data/tracked" # Folder for tracked sequences
os.makedirs(tracked_dir, exist_ok=True)

# ------------------- Initialize DeepSORT tracker -------------------
tracker = DeepSort(max_age=30)

# ------------------- Helper: Convert keypoints to bounding box -------------------
def keypoints_to_bbox(keypoints):
    """
    keypoints: list of [x, y] for each landmark
    Returns bbox: [x_min, y_min, width, height]
    """
    keypoints = np.array(keypoints)
    x_min = np.min(keypoints[:, 0])
    y_min = np.min(keypoints[:, 1])
    x_max = np.max(keypoints[:, 0])
    y_max = np.max(keypoints[:, 1])
    return [x_min, y_min, x_max - x_min, y_max - y_min]

# ------------------- Loop over pose files -------------------
for file in os.listdir(pose_dir):
    if not file.endswith(".npy"):
        continue
    
    keypoints_all_frames = np.load(os.path.join(pose_dir, file), allow_pickle=True)
    tracked_frames = []

    for frame_keypoints in keypoints_all_frames:
        bboxes = []
        for person_kp in frame_keypoints:
            bbox = keypoints_to_bbox(person_kp)
            bboxes.append(bbox)

        # Feed to DeepSORT
        tracks = tracker.update_tracks(bboxes, frame=None)  # frame=None since we only have keypoints
        frame_tracks = []
        for track in tracks:
            # track.track_id gives the ID
            # track.to_ltrb() returns [left, top, right, bottom] of the tracked bbox
            frame_tracks.append({
                "track_id": track.track_id,
                "bbox": track.to_ltrb()
            })
        tracked_frames.append(frame_tracks)

    # ------------------- Save tracked data -------------------
    save_path = os.path.join(tracked_dir, file)
    np.save(save_path, tracked_frames)
    print(f"[INFO] Tracked {file} -> {save_path}")

print("✅ Tracking completed!")
