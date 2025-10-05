# Sentinel Eye Project

A privacy-preserving theft detection system.

Sentinel Eye is an AI-powered surveillance system designed to detect shoplifting or suspicious human behavior in real time using pose estimation and deep learning.
The project processes video footage, extracts skeletal movement data using MediaPipe BlazePose, tracks poses over time, models the human motion using ST-GCN (Spatial Temporal Graph Convolutional Network) logic, and classifies sequences using an LSTM-based deep learning model.
When suspicious activity is detected, the system records evidence, saves it in a database, and sends instant WhatsApp alerts to security staff. A Flask web dashboard allows staff to review, confirm, or dismiss alerts with visual evidence.

Sentinel Eye integrates computer vision, deep learning, and automation to build a complete AI surveillance system.
It bridges human motion analysis (MediaPipe) with temporal understanding (LSTM) and automated incident response (alerts + dashboard).
The project proves how AI can assist in crime prevention and real-time monitoring, making it suitable for retail, malls, and public safety environments.

WORKFLOW OVERVIEW:
    The entire project pipeline is divided into five major stages:

    1. Pose Extraction (Pose Estimation using MediaPipe)

        Code file: Scripts/01_pose_extraction.py
        The system uses MediaPipe Pose from Google to detect 33 body landmarks per frame.
        It reads videos from two folders:
            normal/ → Normal customer behavior
            shoplifting/ → Suspicious behavior
        Each frame is processed to extract (x, y) coordinates for all joints.
        For every video, a .npy file containing sequential pose data is saved in pose_data/.

    2. Object Tracking (DeepSORT Integration)

        Code file: Scripts/02_tracking.py
        Uses DeepSORT (Deep Simple Online and Realtime Tracking) for object consistency — ensuring one person’s pose sequence is tracked throughout the video.
        Currently uses placeholder logic (since only one subject is in each frame), but ready for extension to multi-person tracking.
        Output stored in pose_data/tracked/.

    3. Graph Construction (ST-GCN Graph Generation)

        Code file: Scripts/03_graph_construction.py
        Converts each tracked pose sequence into a graph representation suitable for Spatial-Temporal Graph Convolutional Networks (ST-GCN).
        Each human body joint acts as a node; physical connections between joints (limbs) act as edges.
        Creates an adjacency matrix (33×33) showing which joints are connected.
        Saves graph data (keypoints + adj_matrix) in graphs/.
        Purpose: Modeling motion as a spatial-temporal graph enables learning both spatial relationships (body structure) and temporal patterns (motion over time).  

    4. Behavior Classification (LSTM Deep Learning Model)

        Code file: Scripts/04_train_stgcn.py and Scripts/05_train_sequence_classifier.py
        The tracked pose sequences are used to train an LSTM (Long Short-Term Memory) model.
        Each training sample is a 30-frame (1-second) sequence of keypoints.
        LSTM learns motion patterns — how the human pose changes over time.
        Optimizer: Adam
        Loss: Categorical Crossentropy
        Metrics: Accuracy
        EarlyStopping and ModelCheckpoint used for best model saving.
        Output: Trained model → sentinel_model.h5

    5. Real-Time Surveillance and Alert System

        Code file: sentinel_eye_live.py
        Continuously captures live video from the webcam.(Can be integrated to cctv cameras of the shop for future implementation.)
        Performs real-time pose extraction using MediaPipe.
        Maintains a rolling buffer of 30 frames (1-second motion window).
        Passes the sequence to the trained LSTM model for prediction.
        Alert Trigger Logic:
            If shoplifting confidence > 0.6, an alert is triggered:
            Saves a 10-second video clip of the event.
            Captures a screenshot of the suspect’s pose.
            Calculates a SHA-256 hash for clip integrity.
            Logs everything in SQLite database (sentinel_alerts.db).
            Sends a real-time WhatsApp alert using pywhatkit.

    6. Staff Review Dashboard (Flask Web App)

        Code file: app.py
        Displays all alerts from the database in a clean, sorted dashboard.
        Each alert card includes:
            Timestamp
            Confidence score
            Hash code
            Screenshot preview
            Status (PENDING, GOOD, BAD)
            Staff notes
        Staff can confirm or dismiss alerts directly from the UI.
        Database auto-updates the status.
        Includes a /reset_db route for clearing alerts and media during testing.


DEMO VIDEO DRIVE LINK (Input is feed by playing an anomaly theft video in phone and keeping it near the laptop's camera): https://drive.google.com/file/d/1-upStSB6YTz32JGmad6ZZRpTs2g29IeD/view?usp=sharing 

DEMO VIDEO DRIVE LINK (Real time demo where one of our teammate acts like stealing snacks in a supermarket for evaluating our sentinel eye project in real time): https://drive.google.com/file/d/1IaOZhS0DTTO5bWD_TvfpP-8oKJjdCH2r/view?usp=sharing


# HOW TO RUN OUR CODE: 
This project requires python environment with versions less than python 3.11.0 (here we used python 3.10.0) 

Type these in terminal:

py -3.10 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

To run the program: 

python sentinel_eye_live.py 

Use this video input for checking the code: https://drive.google.com/file/d/18lciIneOIZlSROykKM1N0qVJtnWDhi6C/view?usp=sharing 

(NOTE: You can change the whatsapp number in the code file to get allert messages. Make sure to keep whatsapp web open in the browser to send messages.Press ctrl+C in the terminal to terminate after the alert is sent and the video clip is stored.)

To view the dashboard:

python app.py

(NOTE: Follow the link to view the dashboard.)
