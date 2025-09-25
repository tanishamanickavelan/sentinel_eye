# Sentinel Eye - Shoplifting Detection using ST-GCN

## Overview
This project detects shoplifting actions from surveillance videos using **Spatial-Temporal Graph Convolutional Networks (ST-GCN)**.  
Videos are preprocessed into skeleton keypoints, tracked, converted into graph sequences, and then used to train the ST-GCN model.

## Folder Structure
- `videos/` → Raw video clips (shoplifting + normal)
- `pose_data/` → Skeleton keypoints extracted per frame
- `graphs/` → Graph sequences for ST-GCN input
- `scripts/` → Python scripts for preprocessing and graph generation
- `labels.csv` → Video labels for supervised learning

## Steps to Run
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
