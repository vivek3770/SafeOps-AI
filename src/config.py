import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # API configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Storage paths
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
    SQLITE_DB_PATH = str(DATA_DIR / "safeops.db")
    LSTM_MODEL_PATH = str(DATA_DIR / "lstm_weights.pth")
    RAW_DOCS_DIR = str(DATA_DIR / "raw_docs")
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DOCS_DIR, exist_ok=True)

    # LSTM Hyperparameters
    LSTM_WINDOW_SIZE = 12       # 12 steps of 5 seconds = 60 seconds context
    LSTM_FEATURES = 3          # Gas (LEL%), Temperature (C), Pressure (bar)
    LSTM_HIDDEN_DIM = 16       # LSTM hidden state dimension
    LSTM_EPOCHS = 100          # Fast training for prototype
    LSTM_BATCH_SIZE = 32
    LSTM_LR = 0.001

    # Risk Fusion Engine parameters
    RISK_WEIGHTS = {
        'sensor': 0.35,
        'cv': 0.28,
        'permit': 0.22,
        'shift': 0.15
    }
    
    SEVERITY_THRESHOLDS = {
        'LOW': 0.30,
        'MED': 0.50,
        'HIGH': 0.75,
        'CRITICAL': 0.90
    }

    @classmethod
    def validate(cls):
        """Prints warnings for configuration issues."""
        if not cls.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY environment variable is not set. RAG capabilities will run in fallback mock mode.")
