"""
Централизованная конфигурация проекта Car Brand Recognition.

Все пути вычисляются относительно корня проекта.
Переменные окружения загружаются из .env (см. .env.example).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Корень проекта ─────────────────────────────────────────
# Определяем автоматически: поднимаемся из src/config.py → корень
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Roboflow ────────────────────────────────────────────────
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = "carbrandrecognition"
ROBOFLOW_DETECTION_PROJECT = "car-logo-detection-pcfc6-ir2g4"
ROBOFLOW_DETECTION_VERSION = 1

# ─── Пути к моделям ─────────────────────────────────────────
DETECTOR_PATH = os.getenv(
    "DETECTOR_PATH",
    str(PROJECT_ROOT / "models" / "car_logo_detector.pt"),
)
CLASSIFIER_PATH = os.getenv(
    "CLASSIFIER_PATH",
    str(PROJECT_ROOT / "models" / "car_logo_classifier.pth"),
)

# ─── Пороги уверенности ─────────────────────────────────────
DETECTION_CONF_THRESHOLD = float(os.getenv("DETECTION_CONF_THRESHOLD", "0.4"))
CLASSIFIER_CONF_THRESHOLD = float(os.getenv("CLASSIFIER_CONF_THRESHOLD", "0.3"))

# ─── Список классов (production) ────────────────────────────
CLASS_LIST = [
    "hyundai", "lexus", "mazda", "mercedes",
    "opel", "skoda", "toyota", "volkswagen",
]

CLASS_LIST_RU = [
    "Hyundai", "Lexus", "Mazda", "Mercedes",
    "Opel", "Skoda", "Toyota", "Volkswagen",
]

NUM_CLASSES = len(CLASS_LIST)

# ─── Пути к датасетам ───────────────────────────────────────
DATASETS_DIR = PROJECT_ROOT / "datasets"
DETECTION_DATASET_DIR = DATASETS_DIR / "car_logos_detection"
CLASSIFICATION_DATASET_DIR = DATASETS_DIR / "car_brands_classification"

# ─── Параметры обучения детектора ───────────────────────────
DETECTOR_MODEL_SIZE = os.getenv("DETECTOR_MODEL_SIZE", "yolov8n.pt")
DETECTOR_EPOCHS = int(os.getenv("DETECTOR_EPOCHS", "50"))
DETECTOR_BATCH = int(os.getenv("DETECTOR_BATCH", "16"))
DETECTOR_IMG_SIZE = int(os.getenv("DETECTOR_IMG_SIZE", "640"))

# ─── Параметры обучения классификатора ──────────────────────
CLASSIFIER_IMG_SIZE = (224, 224)
CLASSIFIER_BATCH_SIZE = 16
CLASSIFIER_LEARNING_RATE = 1e-3
CLASSIFIER_EPOCHS = 25
CLASSIFIER_UNFREEZE_EPOCH = 10
