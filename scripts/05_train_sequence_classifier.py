import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# NOTE: The Attention import was removed to fix the ValueError
# The Attention layer requires a list of inputs ([query, value]) and is difficult 
# to use correctly within a simple Sequential model for self-attention.

# --- Configuration ---
TRACKED_DIR = "pose_data/tracked"  # Contains 'normal/' and 'shoplifting/' .npy files
MODEL_SAVE_PATH = "sentinel_model.h5"
SEQUENCE_LENGTH = 30  # How many frames (time steps) to consider in one sequence (1 second = 30 frames)
BATCH_SIZE = 32
EPOCHS = 50

# --- 1. Data Loading and Structuring ---
def load_data(data_dir):
    """Loads keypoints and labels from the tracked directory."""
    sequences = []
    labels = []
    
    # 0 = Normal, 1 = Shoplifting
    class_map = {'normal': 0, 'shoplifting': 1}

    for class_name, label in class_map.items():
        class_path = os.path.join(data_dir, class_name, "*.npy")
        
        for file_path in glob.glob(class_path):
            keypoints_data = np.load(file_path, allow_pickle=True)
            
            # The structure is (Frames, Joints, Coords). Reshape to (Frames, Joints * Coords)
            # Example: (N, 33, 2) -> (N, 66)
            num_frames, num_joints, num_coords = keypoints_data.shape
            flat_keypoints = keypoints_data.reshape(num_frames, num_joints * num_coords)
            
            # Create overlapping sequences (e.g., 30 frames long)
            for i in range(0, num_frames - SEQUENCE_LENGTH, 1): # Step of 1 for more data
                seq = flat_keypoints[i:i + SEQUENCE_LENGTH]
                sequences.append(seq)
                labels.append(label)

    return np.array(sequences), to_categorical(np.array(labels), num_classes=len(class_map))

print("[INFO] Loading and structuring data...")
X, y = load_data(TRACKED_DIR)
print(f"[INFO] Total sequences: {X.shape[0]}, Sequence shape: {X.shape[1:]}")

# --- 2. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[INFO] Train sequences: {X_train.shape[0]}, Test sequences: {X_test.shape[0]}")

# --- 3. Model Definition (LSTM with Attention - Simplified) ---
def build_model(input_shape, num_classes):
    """Defines a sequence-to-sequence classification model."""
    model = Sequential([
        # LSTM for sequence feature extraction (returns sequence of 30 frames)
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        
        # Second LSTM for final sequence summary (return_sequences=False by default)
        LSTM(64),
        Dropout(0.3),
        
        # Output layer
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Get input shape: (SEQUENCE_LENGTH, Joints * Coords)
input_shape = (SEQUENCE_LENGTH, X.shape[2])
num_classes = y.shape[1]
model = build_model(input_shape, num_classes)
model.summary()

# --- 4. Training ---
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1)
]

print("[INFO] Starting model training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print(f"✅ Model training completed. Best model saved to: {MODEL_SAVE_PATH}")
# NOTE: After running this, a file 'sentinel_model.h5' will be created.
