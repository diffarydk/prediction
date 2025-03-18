"""
Configuration module for Baccarat Prediction System.
Contains all constants, file paths, and configuration parameters.
"""

# File paths
MODEL_FILE = "models/pkl/baccarat_model.pkl"
BACARAT_DATA_FILE = "data/csv/bacarat_data.csv"
REALTIME_FILE = "data/csv/realtime_bacarat.csv"
LOG_FILE = "data/csv/log/prediction_log.csv"
BALANCE_FILE = "data/json/balance.json"
BET_HISTORY_FILE = "data/json/bet_history.json"

PLAYER_PAYOUT = 1.0   # 1:1 for Player

REALTIME_DATA_FILE = REALTIME_FILE  # Beberapa file menggunakan nama ini

# Algorithm parameters
MONTE_CARLO_SAMPLES = 300  # Jumlah sampel untuk simulasi Monte Carlo
MARKOV_MEMORY = 50  # Jumlah game terakhir untuk model Markov

# Betting constants
INITIAL_BALANCE = 0  # Rp 25k (dari kode Anda terlihat menggunakan 0, tapi komentar menunjukkan 25k)
MIN_BET = 2000  # Minimum bet (Rp 2k)
MAX_BET = 50000000  # Maximum bet (Rp 50,000k)
BANKER_PAYOUT = 0.95  # 0.95:1 for Banker (5% commission)
PLAYER_PAYOUT = 1.0   # 1:1 for Player

# Model Registry parameters
REGISTRY_PATH = "models/registry"
MODEL_REGISTRY_PATH = REGISTRY_PATH  # Add this line to fix the import error
COMPETITION_INTERVAL = 10  # Run model competition every N predictions
MAX_MODELS = 20  # Maximum number of active models to maintain

# XGBoost parameters
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05