from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "database"
FACES_DIR = DB_DIR / "authorized_faces"
CAPTURES = BASE_DIR / "captures"
MODELS = BASE_DIR / "models"
LOGS = BASE_DIR / "logs"
DB_PATH = DB_DIR / "security.db"

CAMERA_SOURCE   = 0       
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 480
CAMERA_FPS      = 30

YOLO_MODEL = "yolov8n.pt"  
YOLO_CONFIDENCE = 0.5      
DETECTION_INTERVAL   = 2 

FACE_TOLERANCE = 0.5       
FACE_RECOGNITION_INTERVAL = 10

ALERT_COOLDOWN = 30        
SAVE_CAPTURES = True 

DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8000


LOG_LEVEL = "INFO"