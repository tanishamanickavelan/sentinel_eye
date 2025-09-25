import numpy as np
import os

# Directories
tracked_dir = "../pose_data/tracked"  # Folder with tracked keypoints
graph_dir = "../graphs"               # Folder to save ST-GCN graphs
os.makedirs(graph_dir, exist_ok=True)

# Loop over tracked pose files
for file in os.listdir(tracked_dir):
    if not file.endswith(".npy"):
        continue
    keypoints_seq = np.load(os.path.join(tracked_dir, file), allow_pickle=True)
    
    # Example: create adjacency matrix for 33 joints (BlazePose)
    num_joints = 33
    adj_matrix = np.zeros((num_joints, num_joints))
    
    # Define edges (example: extend for full skeleton)
    edges = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6)]  
    for i,j in edges:
        adj_matrix[i,j] = adj_matrix[j,i] = 1

    # Save graph as NumPy file
    np.save(os.path.join(graph_dir, file.replace(".npy","_graph.npy")), 
            {"keypoints": keypoints_seq, "adj": adj_matrix})

print("Graph construction completed!")
