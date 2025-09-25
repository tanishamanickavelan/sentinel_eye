import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# --------- Custom Dataset ---------
class GraphDataset(Dataset):
    def __init__(self, graph_dir, labels_csv):
        self.graph_files = []
        for root, dirs, files in os.walk(graph_dir):
            for f in files:
                if f.endswith("_graph.npy"):
                    self.graph_files.append(os.path.join(root, f))
        self.labels = {}
        with open(labels_csv, "r") as f:
            next(f)  # skip header
            for line in f:
                name, label = line.strip().split(",")
                self.labels[name] = 1 if label=="shoplifting" else 0

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        file_path = self.graph_files[idx]
        data = np.load(file_path, allow_pickle=True).item()
        keypoints = np.array(data['keypoints'])  # T x J x 2
        keypoints = torch.tensor(keypoints, dtype=torch.float32).permute(2,0,1)  # 2 x T x J

        # Extract base filename for label
        base_name = os.path.basename(file_path).replace("_graph.npy",".mp4")
        label = torch.tensor(self.labels[base_name], dtype=torch.long)
        return keypoints, label

# --------- Simple ST-GCN Layer ---------
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), padding=(1,0))
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.temporal_conv(x))

class STGCN(nn.Module):
    def __init__(self, num_joints, num_classes=2):
        super().__init__()
        self.block1 = STGCNBlock(2,32,num_joints)
        self.block2 = STGCNBlock(32,64,num_joints)
        self.fc = nn.Linear(64*num_joints, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.fc(x)

# --------- Training ---------
graph_dir = "graphs"
labels_csv = "labels.csv"
dataset = GraphDataset(graph_dir, labels_csv)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = STGCN(num_joints=33).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # small example
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

print("✅ Training completed!")
