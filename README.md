# LLM-Enhanced SAM/OPV Device Performance Prediction

This repository integrates Large Language Model (LLM) text representations with tabular machine learning and probabilistic regression models[cite: 5, 7, 8] to predict four key photovoltaic performance metrics for Self-Assembled Monolayers (SAM) and Organic/Perovskite Photovoltaic (OPV/Perovskite) devices[cite: 1, 2]:
- **PCE (%)**: Power Conversion Efficiency[cite: 1]
- **VOC (V)**: Open-Circuit Voltage[cite: 1]
- **JSC (mA·cm⁻²)**: Short-Circuit Current Density[cite: 1]
- **FF**: Fill Factor[cite: 1]

---

## 📁 Repository Structure

```text
├── sam.xlsx                                      # Raw experimental dataset containing device/material parameters
├── requirements.txt                              # Pip dependency specification
├── environment.yml                               # Conda environment configuration
├── figure/                                       # Generated figures (heatmaps, parity plots, distributions)
├── training/                                     # Source code for training, embedding extraction, and evaluation
│   ├── config.py                                 # Path configurations, target definitions, and model parameters
│   ├── data_utils.py                             # Data preprocessing, prompt construction, and PCA feature enhancement
│   ├── download_model.py                         # Model downloader from Hugging Face Hub
│   ├── extract_embeddings.py                     # Single-model hidden state representation extraction
│   ├── extract_5_models.py                       # Batch embedding extraction across multiple open LLMs
│   ├── train.py                                  # Model training, PCA feature fusion, and permutation importance
│   ├── prediction.py                             # Multi-target test set evaluation and prediction exporter
│   └── excel_reader.py                           # Low-level XLSX/XML data stream parser
└── training_results/                             # Experimental artifacts, representations, and evaluation metrics
    ├── data/                                     # Standardized datasets, validation splits, and screening outputs
    │   ├── clean_dataset.csv                     # Processed tabular dataset
    │   ├── split_assignments.csv                 # Train/test index mapping and group labels
    │   ├── pred_vs_true_llm_enhanced_qwen3-14b.csv # Test set predictions (mean and uncertainty) vs. ground truth
    │   ├── repeated_splits_r2_comparison.csv     # Multi-seed repeated split stability benchmark
    │   ├── sam_group_extrapolation_qwen3_14b_ngboost.csv # Out-of-distribution (OOD) SAM scaffold group split results
    │   ├── top_500_virtual_candidates_real_micro.csv      # Top 500 candidate materials from high-throughput virtual screening
    │   ├── academic_table_data.csv               # Baseline and model comparison summary tables
    │   └── robustness_raw_data/                  # Raw experimental data for robustness and perturbation tests
    ├── embeddings/                               # Extracted LLM hidden-state embeddings (.npy) and prompt caches
    ├── models/                                   # Serialized regression pipelines and best estimators (.joblib)
    ├── metrics/                                  # Hyperparameter settings (cs.json), metrics (metrics.json), and feature importance
    └── predictions/                              # Prediction outputs across scenarios (test_predictions.csv)
```[cite: 1, 2, 3, 4, 5, 6, 7, 8]

---

## ⚙️ Environment Setup & Installation

### Option 1: Using Conda (Recommended)
Create and activate the environment directly using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate opv_ml
```