import sqlite3
from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime
import base64 

# --- Configuration ---
DATABASE = 'sentinel_alerts.db'
VIDEO_CLIP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(_file_)), 'alerts', 'clips')
IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(_file_)), 'alerts', 'screenshots') # NEW FOLDER
DISPLAY_THRESHOLD = 0.60 # CRITICAL: NEW DISPLAY THRESHOLD SET HERE

app = Flask(_name_)

# --- Database Functions ---

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_alerts():
    """Retrieves all alerts from the database, ensuring data types are correct and sorting them."""
    conn = get_db_connection()
    alerts_raw = conn.execute('SELECT * FROM alerts ORDER BY confidence DESC, timestamp DESC').fetchall()
    conn.close()

    alerts = []
    for row in alerts_raw:
        alert = dict(row)
        
        # NEW FIX: Format timestamp to remove microseconds for cleaner display
        try:
            dt_object = datetime.strptime(alert['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            alert['timestamp'] = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Handle cases where timestamp might be stored in a simpler format
            pass

        # Aggressively convert confidence from stored string to float
        try:
            alert['confidence'] = float(str(alert['confidence']))
        except Exception:
            alert['confidence'] = 0.0
        
        # NEW FILTER: Only show alerts that meet the minimum display threshold
        if alert['confidence'] >= DISPLAY_THRESHOLD:
            
            # --- NEW: Base64 Encoding for Screenshot Playback ---
            img_full_path = alert.get('image_path') # Use the new image_path field
            alert['image_data'] = None
            
            if img_full_path and os.path.exists(img_full_path):
                try:
                    with open(img_full_path, 'rb') as image_file:
                        image_bytes = image_file.read()
                        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                        # CRITICAL CHANGE: Use data:image/jpeg mime type
                        alert['image_data'] = f"data:image/jpeg;base64,{encoded_image}"
                except Exception as e:
                    print(f"[ERROR] Failed to read or encode image {img_full_path}: {e}")
            
            alerts.append(alert)

    return alerts

# --- Flask Routes ---
@app.route('/')
def index():
    """Displays the main dashboard with all alerts."""
    alerts = get_alerts()
    return render_template('index.html', alerts=alerts)

@app.route('/update', methods=['POST'])
def update_status():
    """Handles the form submission to update an alert's status and add staff notes based on button click."""
    alert_id = request.form['id']
    notes = request.form['notes'].strip()
    
    # LOGIC: Determine status based on button click (action field)
    action = request.form['action']
    if action == 'confirm':
        new_status = 'BAD'
    elif action == 'dismiss':
        new_status = 'GOOD'
    else:
        new_status = 'PENDING'

    conn = get_db_connection()
    conn.execute(
        'UPDATE alerts SET status = ?, staff_notes = ? WHERE id = ?',
        (new_status, notes, alert_id)
    )
    conn.commit()
    conn.close()
    
    return redirect(url_for('index'))

@app.route('/reset_db')
def reset_db():
    """Utility route to clear all data and video clips (for testing/hackathon use only)."""
    conn = get_db_connection()
    
    # Drop and recreate the alerts table
    conn.execute('DROP TABLE IF EXISTS alerts')
    conn.execute("""
        CREATE TABLE alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            confidence TEXT, 
            clip_path TEXT,
            clip_hash_sha256 TEXT,
            image_path TEXT, 
            status TEXT DEFAULT 'PENDING',
            staff_notes TEXT
        )
    """)
    conn.commit()
    conn.close()
    
    # Attempt to clear the saved video clips and screenshots
    for folder in [VIDEO_CLIP_FOLDER, IMAGE_FOLDER]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                        print(f"Deleted file: {filename}")
                    except Exception as e:
                        print(f"Could not delete {filename}: {e}. Manual deletion required.")

    return redirect(url_for('index'))

if _name_ == '_main_':
    # Ensure both folders exist on startup
    os.makedirs(VIDEO_CLIP_FOLDER, exist_ok=True)
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    app.run(debug=True)