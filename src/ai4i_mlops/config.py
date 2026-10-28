
from pathlib import Path

# Rutas
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_PATH = DATA_DIR / "ai4i2020.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "ai4i_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Datos
FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

# Entrenamiento
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hiperparámetros
MODELS_PARAMS = {
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "learning_rate": 0.1,
    "l2_regularization": 0.0,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE
}

# Umbral de clasificación
THRESHOLD = 0.5

# Calidad mínima aceptable
MIN_RECALL = 0.70
MIN_F1 = 0.65
