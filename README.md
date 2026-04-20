# AI Security System

A high-scale, real-time unauthorized person detection system built with **YOLOv8**, **InsightFace (ArcFace)**, and **FastAPI** — designed to run on both Windows (development) and Raspberry Pi (deployment).

---

## 📌 Features

- 👥 **Face ID** To add the authorised people into the system to capture their faces with 360 Degree
- 🎯 **Real-time person detection** using YOLOv8-nano
- 🧠 **Face recognition** using InsightFace / ArcFace (~99.8% accuracy)
- 👤 **Authorized persons database** with face enrollment from image or webcam
- 🚨 **Unauthorized person alerts** with snapshot capture
- 🧵 Thread-safe camera capture for smooth multi-module access

---

## 🗂️ Project Structure

```
security-system/
├── main.py
├── FaceID.py
├── FaceMesh.py 
├── config.py                   # All system settings (paths, thresholds, FPS...)
├── camera.py                   # Thread-safe webcam/Pi camera capture module
├── detector.py                 # YOLOv8-nano person detection module
├── recognizer.py               # InsightFace ArcFace recognition module
├── enroll.py                   # CLI tool to enroll authorized persons
├── test_camera.py              # Test: verify camera feed
├── test_detector.py            # Test: verify YOLO person detection
├── test_recognizer.py          # Test: full detection + recognition pipeline
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignored files (venv, models, captures...)
│
├── captures/                   # Snapshots of detected unauthorized persons
├── database/
│   ├── authorized_faces/       # Enrolled face images per person
│   └── embeddings.pkl          # Saved ArcFace face embeddings
├── models/                     # YOLO model weights
├── logs/                       # System logs
├── alerts/                     # Alert audio/files
├── static/                     # Dashboard CSS / JS (Step 6)
└── templates/                  # Dashboard HTML templates (Step 6)
```

---

## ⚙️ Requirements

- Python **3.10+**
- Windows 10/11 (development) or Raspberry Pi 4 (deployment)
- Webcam or Pi Camera Module
- 4GB+ RAM recommended

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/security-system.git
cd security-system
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Raspberry Pi
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


### 4. Download InsightFace models (first run only)

```bash
python IF_model.py
```

This downloads ~300MB of ArcFace models to `~/.insightface/` — only needed once.

---

## 👤 Enrolling Authorized Persons

### From webcam (recommended)

```bash
python FaceID.py enroll "lenda deif" 
```

Position your face clearly, then press **SPACE** to capture.


### List all enrolled persons

```bash
python FaceID.py list
```

### Remove a person

```bash
python FaceID.py  remove "lenda deif"
```

---

## ▶️ Running the System
```bash
python main.py
```



---

## 🎨 Visual Guide

| Bounding Box Color | Meaning |
|---|---|
| 🟠 Orange | Person detected — face not yet identified |
| 🟢 Green | ✅ Authorized person — recognized |
| 🔴 Red | 🚨 **UNAUTHORIZED** — unknown person |

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Person Detection | YOLOv8-nano (Ultralytics) |
| Face Recognition | InsightFace / ArcFace |
| Inference Engine | ONNX Runtime (CPU) |
| Camera Capture | OpenCV + threading |
| Database | SQLite + embeddings.pkl |
| Logging | Loguru |


---

## 🗺️ Roadmap

| Step | Description | Status |
|---|---|---|
| Step 1 | Project setup, environment & folder structure | ✅ Done |
| Step 2 | Camera capture module | ✅ Done |
| Step 3 | Person detection with YOLOv8-nano | ✅ Done |
| Step 4 | Face recognition with InsightFace/ArcFace | ✅ Done |
| Step 5 | Alert system (sound + snapshot + log) | 🔄 In Progress |
| Step 6 | Live web dashboard | ⏳ Upcoming |
| Step 7 | Main entry point + system integration | ⏳ Upcoming |
| Step 8 | 🍓 Raspberry Pi deployment & optimization | ⏳ Upcoming |

---

## ⚙️ Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `0` | `0` = default webcam |
| `CAMERA_FPS` | `30` | Target FPS |
| `YOLO_CONFIDENCE` | `0.5` | Min confidence for person detection |
| `DETECTION_INTERVAL` | `2` | Run YOLO every N frames |
| `FACE_TOLERANCE` | `0.5` | Face match threshold (lower = stricter) |
| `FACE_RECOGNITION_INTERVAL` | `5` | Run recognition every N frames |
| `ALERT_COOLDOWN` | `30` | Seconds between repeated alerts |
| `SAVE_CAPTURES` | `True` | Save snapshots of unauthorized persons |

---

## 🔒 Privacy & Legal

- Face embeddings (not raw images) are stored for authorized persons
- All captured snapshots are stored **locally only**
- Ensure compliance with local privacy laws before deploying in public spaces
- Post visible signage if deploying in monitored areas

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [InsightFace](https://github.com/deepinsight/insightface)
- [OpenCV](https://opencv.org/)
