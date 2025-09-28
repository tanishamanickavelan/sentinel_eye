import cv2
import mediapipe as mp
import numpy as np
import os
import sqlite3
import hashlib
from datetime import datetime
import time
import pywhatkit 

# --- Configuration ---
DATABASE = 'sentinel_alerts.db'
MODEL_PATH = 'sentinel_model.h5' 
ALERT_CONFIDENCE_THRESHOLD = 0.60 # The trigger and minimum save threshold
CLIP_DURATION_SECONDS = 10 
POST_EVENT_RECORDING_SECONDS = 5
VIDEO_CLIP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(_file_)), 'alerts', 'clips')
IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(_file_)), 'alerts', 'screenshots') # NEW FOLDER
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# --- WhatsApp Configuration ---
TARGET_WHATSAPP_NUMBERS = ['+918122205620', '+916374878931'] 

# --- Initialize MediaPipe Pose ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- Database Functions ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def setup_database():
    """Initializes the database and creates the alerts table."""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            confidence TEXT, 
            clip_path TEXT,
            clip_hash_sha256 TEXT,
            image_path TEXT,  -- NEW: Field for screenshot path
            status TEXT DEFAULT 'PENDING',
            staff_notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_file(filepath):
    """Calculates SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def send_realtime_alert(alert_id, confidence, clip_path, file_hash):
    """
    Sends real-time alerts via WhatsApp automation (using pywhatkit).
    """
    alert_time = datetime.now()
    
    message = f"🚨 SENTINEL EYE ALERT 🚨\n"
    message += f"Time: {alert_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    message += f"Confidence: {confidence}\n"
    message += f"Status: PENDING REVIEW\n"
    message += f"Evidence Hash: {file_hash[:12]}..."
    
    for number in TARGET_WHATSAPP_NUMBERS:
        try:
            pywhatkit.sendwhatmsg_instantly(
                phone_no=number, 
                message=message, 
                wait_time=15,
                tab_close=True 
            )
            print("---------------------------------------------")
            print(f"✅ WhatsApp Alert Sent successfully to {number}")
            print("---------------------------------------------")
        except Exception as e:
            print(f"[ERROR] WhatsApp send failed for {number}: {e}")

def save_clip(video_frames, confidence_score, screenshot_path): 
    """Saves the recorded frames to an AVI file and logs to DB."""
    
    # *** CRITICAL FIX: DO NOT SAVE OR ALERT IF CONFIDENCE DROPPED ***
    if confidence_score < ALERT_CONFIDENCE_THRESHOLD:
        print(f"[INFO] Skipping save: Final prediction {confidence_score:.4f} is below threshold {ALERT_CONFIDENCE_THRESHOLD}.")
        return
    # ***************************************************************

    if not video_frames:
        print("[WARNING] Attempted to save an empty clip. Skipping.")
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
    print(f"[INFO] Video clip saved to: {clip_full_path}")

    file_hash = hash_file(clip_full_path)

    # Log to Database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    confidence_str = f"{confidence_score:.4f}" 
    
    # CRITICAL CHANGE: Insert image_path into the database
    cursor.execute(
        'INSERT INTO alerts (timestamp, confidence, clip_path, clip_hash_sha256, image_path, staff_notes) VALUES (?, ?, ?, ?, ?, ?)',
        (datetime.now(), confidence_str, clip_full_path, file_hash, screenshot_path, "Suspicious movement detected.")
    )
    new_alert_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"[ALERT LOGGED] Clip: {filename}\n \
            Confidence: {confidence_str}\n \
            SHA-256 Hash: {file_hash[:12]}...")
            
    send_realtime_alert(new_alert_id, confidence_str, clip_full_path, file_hash)


# --- Main Detection Loop ---

def sentinel_core():
    # Load the trained model
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        print(f"[SUCCESS] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[FATAL] Could not load model: {e}")
        return

    # Initialize variables
    setup_database()
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    os.makedirs(VIDEO_CLIP_FOLDER, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Could not open camera.")
        pose.close()
        return

    # Video buffer setup
    frame_buffer = []
    SEQUENCE_LENGTH = 30
    ALERT_ACTIVE = False
    ALERT_START_TIME = None
    RECORDING_ACTIVE = False
    RECORDING_FRAMES = []
    LAST_PREDICTION = 0.0
    
    # Screenshot variables
    SCREENSHOT_TAKEN = False
    SAVED_SCREENSHOT_PATH = ""

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
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
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
                
                # 4. Anomaly Detection and Recording Control
                if LAST_PREDICTION > ALERT_CONFIDENCE_THRESHOLD:
                    
                    # CAPTURE SCREENSHOT ONLY ON FIRST DETECTION
                    if not SCREENSHOT_TAKEN:
                        # 1. Add Text Overlay to frame before saving
                        cv2.putText(display_frame, f"CONF: {LAST_PREDICTION:.4f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        # 2. Save the Screenshot
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_filename = f"screenshot_{timestamp_str}.jpg"
                        img_full_path = os.path.join(IMAGE_FOLDER, img_filename).replace('\\', '/')
                        cv2.imwrite(img_full_path, display_frame) # Save the image with overlay
                        SAVED_SCREENSHOT_PATH = img_full_path
                        SCREENSHOT_TAKEN = True
                        print(f"[INFO] Screenshot saved to: {img_full_path}")
                    
                    # START RECORDING (Debounce Logic Fix)
                    if not RECORDING_ACTIVE:
                        ALERT_ACTIVE = True
                        RECORDING_ACTIVE = True
                        ALERT_START_TIME = time.time()
                        print(f"\n=============================================")
                        print(f"🚨 ANOMALY DETECTED! Confidence: {LAST_PREDICTION:.4f}")
                        print(f"    Capturing {CLIP_DURATION_SECONDS}-second evidence clip...")
                    
                    RECORDING_FRAMES.append(frame)

                else:
                    ALERT_ACTIVE = False
                    
        # 5. Continuous Recording while Alert is Active
        if RECORDING_ACTIVE:
            if not ALERT_ACTIVE:
                RECORDING_FRAMES.append(frame)
            
            elapsed_recording_time = time.time() - ALERT_START_TIME

            if elapsed_recording_time >= CLIP_DURATION_SECONDS or len(RECORDING_FRAMES) >= CLIP_DURATION_SECONDS * 20:
                
                # Save the final clip and reset state
                save_clip(RECORDING_FRAMES, LAST_PREDICTION, SAVED_SCREENSHOT_PATH) # Pass screenshot path
                
                # Reset recording state
                RECORDING_ACTIVE = False
                RECORDING_FRAMES = []
                ALERT_START_TIME = None
                SCREENSHOT_TAKEN = False # Reset screenshot flag
                SAVED_SCREENSHOT_PATH = ""
                
        # 6. Display Output (Visual Confirmation)
        cv2.putText(display_frame, f"Status: {'ANOMALY' if ALERT_ACTIVE else 'NORMAL'}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if ALERT_ACTIVE else (0, 255, 0), 2)
        cv2.putText(display_frame, f"Conf: {LAST_PREDICTION:.4f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if RECORDING_ACTIVE:
            cv2.putText(display_frame, "RECORDING...", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Sentinel Eye Monitor', display_frame)
        time.sleep(0.001) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    pose.close()

if _name_ == '_main_':
    os.makedirs(VIDEO_CLIP_FOLDER, exist_ok=True)
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    sentinel_core()