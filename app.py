import sqlite3
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from datetime import datetime

# --- Configuration ---
DATABASE = 'sentinel_alerts.db'
VIDEO_CLIP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alerts', 'clips')

app = Flask(__name__)

# --- Ensure clip folder exists ---
os.makedirs(VIDEO_CLIP_FOLDER, exist_ok=True)

# --- Database Functions ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_alerts():
    conn = get_db_connection()
    alerts_raw = conn.execute('SELECT * FROM alerts ORDER BY confidence DESC, timestamp DESC').fetchall()
    conn.close()

    alerts = []
    for row in alerts_raw:
        alert = dict(row)
        try:
            alert['confidence'] = float(str(alert['confidence']))
        except Exception:
            alert['confidence'] = 0.0

        filename = os.path.basename(alert['clip_path'])
        alert['clip_url'] = url_for('serve_clip', filename=filename).replace('\\', '/')
        alerts.append(alert)
    return alerts

# --- Flask Routes ---
@app.route('/')
def index():
    alerts = get_alerts()
    return render_template('index.html', alerts=alerts)

@app.route('/update', methods=['POST'])
def update_status():
    alert_id = request.form['id']
    notes = request.form['notes'].strip()
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
    conn = get_db_connection()
    conn.execute('DROP TABLE IF EXISTS alerts')
    conn.execute("""
        CREATE TABLE alerts (
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

    # Clear clips folder
    if os.path.exists(VIDEO_CLIP_FOLDER):
        for filename in os.listdir(VIDEO_CLIP_FOLDER):
            file_path = os.path.join(VIDEO_CLIP_FOLDER, filename)
            if os.path.isfile(file_path) and filename.endswith('.mp4'):
                try:
                    os.unlink(file_path)
                    print(f"Deleted clip: {filename}")
                except Exception as e:
                    print(f"Could not delete {filename}: {e}")

    return redirect(url_for('index'))

@app.route('/clips/<filename>')
def serve_clip(filename):
    return send_from_directory(VIDEO_CLIP_FOLDER, filename)

# --- Main Entry ---
if __name__ == '__main__':
    app.run(debug=True)
