from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "sam.xlsx"
TRAINING_DIR = PROJECT_DIR / "training"
RESULTS_DIR = PROJECT_DIR / "training_results"
FIGURE_DIR = PROJECT_DIR / "figure"

DATA_DIR = RESULTS_DIR / "data"
MODEL_DIR = RESULTS_DIR / "models"
EMBEDDING_DIR = RESULTS_DIR / "embeddings"
METRIC_DIR = RESULTS_DIR / "metrics"
PREDICTION_DIR = RESULTS_DIR / "predictions"
LOG_DIR = RESULTS_DIR / "logs"

LLM_MODEL_ID = "Qwen/Qwen3-14B"
LLM_MODEL_NAME = "qwen3-14b"
LLM_MODEL_DIR = MODEL_DIR / LLM_MODEL_NAME

LLM_MLX_MODEL_NAME = "qwen3-14b-mlx-4bit"
LLM_MLX_MODEL_DIR = MODEL_DIR / LLM_MLX_MODEL_NAME

TARGET_COLUMNS = ["PCE（%）", "VOC（V）", "JSC(mA·cm⁻²)", "FF"]
IDENTITY_COLUMNS = ["name", "SMILES", "doi"]
RANDOM_SEED = 42

TEXT_COLUMNS = [
    "name",
    "SMILES",
    "Substrate Type",
    "Substrate\u00a0Work\u00a0Function",
    "SAM Solution Concentration (mM)",
    "SAM Solvent",
    "Soaking",
    "Spin Coating",
    "Soak Time",
    "Spin Coating Time (s)",
    "Spin Coating Speed (rpm)",
    "Annealing",
    "Initial Annealing Temperature for SAM (°C)",
    "Post-washing Annealing Temperature (°C)",
    "：Initial Annealing Time for SAM (min)",
    "Post-washing Annealing Time (min)",
    "UV-Ozone Treatment Time (min)",
    "Carrier_Role",
    "Active Layer",
    "Device Type",
]

def ensure_directories():
    for path in [
        DATA_DIR,
        MODEL_DIR,
        EMBEDDING_DIR,
        METRIC_DIR,
        PREDICTION_DIR,
        LOG_DIR,
        FIGURE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

PROTECTED_PROCESS_COLUMNS = [
    "SAM Solution Concentration (mM)",
    "Soak Time",
    "Spin Coating Time (s)",
    "Spin Coating Speed (rpm)",
    "Initial Annealing Temperature for SAM (°C)",
    "Post-washing Annealing Temperature (°C)",
    "：Initial Annealing Time for SAM (min)",
    "Post-washing Annealing Time (min)",
    "UV-Ozone Treatment Time (min)",
]