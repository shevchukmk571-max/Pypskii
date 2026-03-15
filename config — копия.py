"""🚀 Pypsiki in Space — Конфигурация"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "pypsiki.db"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Данные
DATA_URL = "https://disk.yandex.ru/d/WT1ZCvyIbtrgnA"
DATA_FILE = DATA_DIR / "Data.npz"

# Модель
MODEL_PATH = MODELS_DIR / "alien_classifier.h5"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# Обучение
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# Интерфейс
APP_NAME = "🚀 Pypsiki in Space"
APP_VERSION = "3.0"

# Цвета тёмной темы
COLORS = {
    "bg_primary": "#1a1a2e", "bg_secondary": "#16213e", "bg_tertiary": "#0f0f1a",
    "accent": "#00d9ff", "accent_hover": "#00b8d9",
    "success": "#00c853", "warning": "#ff9800", "error": "#f44336",
    "text_primary": "#eeeeee", "text_secondary": "#888888",
}
