import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
import sqlite3
import hashlib
from collections import deque

# --- CONFIGURATION CONSTANTS ---
MODEL_PATH = "sentinel_model.h5"  # Path to your trained model
VIDEO_SOURCE = 0                   # 0 for webcam, or path to a video file
ALERT_CONFIDENCE_THRESHOLD = 0.85 # Minimum probability to trigger an alert (0.0 to 1.0)
SEQUENCE_LENGTH = 30               # Must match training script (05_train_sequence_classifier.py)
PRE_EVENT_FRAMES = 150             # 5 seconds * 30 FPS = 150 frames
POST_EVENT_FRAMES = 150            # 5 seconds * 30 FPS = 150 frames
TOTAL_CLIP_FRAMES = PRE_EVENT_FRAMES + POST_EVENT_FRAMES # Total 10-second clip
CLIPS_DIR = "alerts/clips"
DB_PATH = "sentinel_alerts.db"

# --- INITIALIZATION ---
os.makedirs(CLIPS_DIR, exist_ok=True)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 1. Initialize SQLite Database
def init_db():
    """Initializes the SQLite database and creates the alerts table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            confidence REAL,
            clip_path TEXT,
            clip_hash_sha256 TEXT,
            status TEXT DEFAULT 'PENDING',
            staff_notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 2. Load Model and Initialize Pose Estimator
def load_assets():
    """Loads the trained Keras model and MediaPipe pose module."""
    try:
        model = load_model(MODEL_PATH)
        print(f"[SUCCESS] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        print("    -> Make sure you have run 05_train_sequence_classifier.py successfully.")
        return None, None

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1 # Use 1 for speed on CPU
    )
    print("[SUCCESS] MediaPipe Pose Initialized.")
    return model, pose

# 3. Hashing and DB Logging
def hash_and_log(clip_path, confidence):
    """Calculates SHA-256 hash of the clip and logs the alert to the database."""
    try:
        # Calculate SHA-256 Hash for tamper-proof evidence
        hasher = hashlib.sha256()
        with open(clip_path, 'rb') as clip_file:
            buf = clip_file.read()
            hasher.update(buf)
        file_hash = hasher.hexdigest()
        
        # Log to Database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, confidence, clip_path, clip_hash_sha256, staff_notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, confidence, clip_path, file_hash, "Suspicious movement detected."))
        
        conn.commit()
        conn.close()
        
        print(f"\n[ALERT LOGGED] Clip: {os.path.basename(clip_path)}")
        print(f"             Confidence: {confidence:.4f}")
        print(f"             SHA-256 Hash: {file_hash[:10]}...")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to hash or log clip: {e}")
        return False

# 4. Save Video Clip
def save_clip(frame_buffer, clip_frames, confidence):
    """Saves the buffered frames (pre- and post-event) to a video file."""
    
    # Generate unique filename based on current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clip_filename = f"alert_{timestamp}.mp4"
    clip_path = os.path.join(CLIPS_DIR, clip_filename)

    # Get video properties from the buffer (assuming they are constant)
    if not clip_frames:
        print("[ERROR] Frame buffer is empty.")
        return False
        
    H, W, _ = clip_frames[0].shape
    
    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v for wider compatibility
    out = cv2.VideoWriter(clip_path, fourcc, 30.0, (W, H))

    # Write frames from the buffer
    for frame in clip_frames:
        out.write(frame)
        
    out.release()
    print(f"[INFO] Video clip saved to: {clip_path}")
    
    # Log the event and hash the file
    hash_and_log(clip_path, confidence)

# 5. Main Detection Loop
def sentinel_core():
    """Runs the main camera processing and anomaly detection loop."""
    model, pose = load_assets()
    if model is None:
        return

    init_db()
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Frame ring buffer (stores the last 150 frames + current)
    # This is used to capture 5 seconds *before* the detection event
    frame_buffer = deque(maxlen=PRE_EVENT_FRAMES)
    
    # Sequence buffer (stores the last 30 normalized keypoint sequences for model input)
    keypoint_sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # State control variables
    alert_triggered = False
    post_event_counter = 0
    clip_frames_to_save = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a selfie-view if using a webcam
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        
        # Add current frame to the ring buffer
        frame_buffer.append(frame.copy()) 

        # --- Pose Detection ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # --- Drawing and Keypoint Extraction ---
        keypoints_normalized = []
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract keypoints (33 joints * 2 coords (x, y))
            keypoints = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
            keypoints_normalized = np.array(keypoints).flatten() # Shape (66,)
            
            # Add to sequence buffer if enough data is available
            keypoint_sequence_buffer.append(keypoints_normalized)

        # --- Anomaly Detection and State Management ---
        if len(keypoint_sequence_buffer) == SEQUENCE_LENGTH and not alert_triggered:
            
            # Reshape buffer for Keras model: (1, 30, 66)
            input_seq = np.expand_dims(keypoint_sequence_buffer, axis=0)
            
            # Make prediction
            prediction = model.predict(input_seq, verbose=0)[0]
            # prediction is a list: [Prob_Normal, Prob_Shoplifting]
            shoplifting_confidence = prediction[1]
            
            # Display Prediction
            status_text = f"Status: Normal ({prediction[0]:.2f}) | Anomaly ({shoplifting_confidence:.2f})"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # --- Trigger Alert ---
            if shoplifting_confidence >= ALERT_CONFIDENCE_THRESHOLD:
                alert_triggered = True
                
                # Retrieve PRE-EVENT frames from the ring buffer
                pre_event_frames = list(frame_buffer) 
                clip_frames_to_save.extend(pre_event_frames)
                
                print(f"\n=============================================")
                print(f"🚨 ANOMALY DETECTED! Confidence: {shoplifting_confidence:.4f}")
                print(f"    Capturing 10-second evidence clip...")
                print(f"=============================================")

        # --- Clip Recording State ---
        if alert_triggered:
            # Change indicator color to RED
            cv2.putText(frame, "RECORDING EVIDENCE", (W - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save POST-EVENT frames
            if post_event_counter < POST_EVENT_FRAMES:
                clip_frames_to_save.append(frame.copy())
                post_event_counter += 1
            else:
                # Clip recording complete!
                print(f"[INFO] Post-event capture complete. Total frames: {len(clip_frames_to_save)}")
                save_clip(frame_buffer, clip_frames_to_save, shoplifting_confidence)
                
                # Reset state variables
                alert_triggered = False
                post_event_counter = 0
                clip_frames_to_save = []
                # Clear keypoint buffer to prevent immediate re-trigger
                keypoint_sequence_buffer.clear()

        cv2.imshow('Sentinel Eye Monitoring', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\nSentinel Eye shut down.")

if __name__ == "__main__":
    sentinel_core()
