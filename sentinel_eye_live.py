import cv2
import mediapipe as mp
import numpy as np
import os
import sqlite3
import hashlib
from datetime import datetime
import time
import pywhatkit
from tensorflow.keras.models import load_model

# --- Configuration ---
DATABASE = 'sentinel_alerts.db'
MODEL_PATH = 'sentinel_model.h5'
ALERT_CONFIDENCE_THRESHOLD = 0.85
CLIP_DURATION_SECONDS = 10
POST_EVENT_RECORDING_SECONDS = 5
VIDEO_CLIP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alerts', 'clips')

# --- WhatsApp Configuration ---
TARGET_WHATSAPP_NUMBERS = ['+918122205620', '+916374878931']

# --- Initialize MediaPipe Pose ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- Database Functions ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def setup_database():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            confidence TEXT,
            clip_path TEXT,
            clip_hash_sha256 TEXT,
            status TEXT DEFAULT 'PENDING',
            staff_notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def send_realtime_alert(alert_id, confidence, clip_path, file_hash):
    alert_time = datetime.now()
    message = f"🚨 SENTINEL EYE ALERT 🚨\nTime: {alert_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    message += f"Confidence: {confidence}\nStatus: PENDING REVIEW\nEvidence Hash: {file_hash[:12]}..."
    
    for number in TARGET_WHATSAPP_NUMBERS:
        try:
            pywhatkit.sendwhatmsg_instantly(phone_no=number, message=message, wait_time=15, tab_close=True)
            print(f"[INFO] WhatsApp Alert sent to {number}")
        except Exception as e:
            print(f"[ERROR] Failed to send WhatsApp alert to {number}: {e}")

def save_clip(video_frames, confidence_score):
    if not video_frames:
        print("[WARNING] Empty clip. Skipping save.")
        return

    first_frame = video_frames[0]
    height, width, _ = first_frame.shape
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{timestamp_str}.avi"
    clip_full_path = os.path.join(VIDEO_CLIP_FOLDER, filename).replace('\\', '/')

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(clip_full_path, fourcc, 20.0, (width, height))
    for frame in video_frames:
        out.write(frame)
    out.release()

    file_hash = hash_file(clip_full_path)

    conn = get_db_connection()
    confidence_str = f"{confidence_score:.4f}"
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO alerts (timestamp, confidence, clip_path, clip_hash_sha256, staff_notes) VALUES (?, ?, ?, ?, ?)',
        (datetime.now(), confidence_str, clip_full_path, file_hash, "Suspicious movement detected.")
    )
    new_alert_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"[ALERT LOGGED] {filename} | Confidence: {confidence_str} | Hash: {file_hash[:12]}...")
    send_realtime_alert(new_alert_id, confidence_str, clip_full_path, file_hash)

# --- Main Detection Loop ---
def sentinel_core():
    try:
        model = load_model(MODEL_PATH)
        print(f"[SUCCESS] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[FATAL] Could not load model: {e}")
        return

    setup_database()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Could not open camera.")
        pose.close()
        return

    frame_buffer = []
    SEQUENCE_LENGTH = 30
    ALERT_ACTIVE = False
    ALERT_START_TIME = None
    RECORDING_ACTIVE = False
    RECORDING_FRAMES = []
    LAST_PREDICTION = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        display_frame = frame.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            keypoints = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
            frame_buffer.append(keypoints)
            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            if len(frame_buffer) == SEQUENCE_LENGTH:
                input_sequence = np.expand_dims(frame_buffer, axis=0)
                reshaped_sequence = input_sequence.reshape(1, SEQUENCE_LENGTH, -1)
                prediction = model.predict(reshaped_sequence, verbose=0)[0]
                LAST_PREDICTION = prediction[1]

                if LAST_PREDICTION > ALERT_CONFIDENCE_THRESHOLD:
                    if not RECORDING_ACTIVE:
                        ALERT_ACTIVE = True
                        RECORDING_ACTIVE = True
                        ALERT_START_TIME = time.time()
                        print(f"\n🚨 ANOMALY DETECTED! Confidence: {LAST_PREDICTION:.4f}")
                    RECORDING_FRAMES.append(frame)
                else:
                    ALERT_ACTIVE = False

        if RECORDING_ACTIVE:
            if not ALERT_ACTIVE:
                RECORDING_FRAMES.append(frame)

            elapsed_recording_time = time.time() - ALERT_START_TIME
            if elapsed_recording_time >= CLIP_DURATION_SECONDS or len(RECORDING_FRAMES) >= CLIP_DURATION_SECONDS*20:
                save_clip(RECORDING_FRAMES, LAST_PREDICTION)
                RECORDING_ACTIVE = False
                RECORDING_FRAMES = []
                ALERT_START_TIME = None

        cv2.putText(display_frame, f"Status: {'ANOMALY' if ALERT_ACTIVE else 'NORMAL'}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if ALERT_ACTIVE else (0,255,0), 2)
        cv2.putText(display_frame, f"Conf: {LAST_PREDICTION:.4f}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if RECORDING_ACTIVE:
            cv2.putText(display_frame, "RECORDING...", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow('Sentinel Eye Monitor', display_frame)
        time.sleep(0.001)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# --- Entrypoint ---
if __name__ == '__main__':
    os.makedirs(VIDEO_CLIP_FOLDER, exist_ok=True)
    sentinel_core()
