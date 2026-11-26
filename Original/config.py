"""
config.py
---------
Central configuration – all default values.
"""

from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR.parent / "Project_Final_Cleaned" / "DATA"
if not DATA_ROOT.exists():
    DATA_ROOT = SCRIPT_DIR / "DATA"

PLOTS_DIR = DATA_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Target
TARGET = "PM_MET_IN1"
UPPER_LIMIT = 100
DEFAULT_FREQ = "10min"

# Features
DEFAULT_FEATURE_LEVEL = "medium"   # ultra_low, low, medium, high

# Model defaults
DEFAULT_N_ESTIMATORS = 200
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_MAX_DEPTH = 6
DEFAULT_NUM_LEAVES = 15
RANDOM_STATE = 42

# Split
DEFAULT_SPLIT = "household"
DEFAULT_TEST_FRACTION = 0.2

# Cross-validation & tuning
N_CV_SPLITS = 5
N_TUNING_TRIALS = 40

# Lags
LAG_STEPS = [1, 3, 6, 12, 18, 36, 72, 144]
LAG_COLUMNS = ["PM_MET_IN1", "MetOne_PM", "StaW_PM", "QCFM_Total", "ACH"]