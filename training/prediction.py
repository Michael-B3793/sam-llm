import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from ngboost import NGBRegressor
except ImportError:
    pass

try:
    from deepforest import CascadeForestRegressor
except ImportError:
    pass

try:
    from gplearn.genetic import SymbolicRegressor
except ImportError:
    pass

from config import TARGET_COLUMNS, MODEL_DIR, EMBEDDING_DIR, RANDOM_SEED, DATA_DIR
from data_utils import (
    add_embeddings,
    build_group_labels,
    coerce_feature_frame,
    feature_columns,
    load_dataset,
    numeric_columns,
)


def split_data(frame, groups, test_size, seed):
    train_index, test_index = train_test_split(
        np.arange(len(frame)), test_size=test_size, random_state=seed
    )
    return train_index, test_index


def main():
    frame = load_dataset()
    groups = build_group_labels(frame)
    train_index, test_index = split_data(frame, groups, 0.1, RANDOM_SEED)

    y = frame[TARGET_COLUMNS].to_numpy(dtype=float)
    y_train = y[train_index]
    y_test = y[test_index]

    v_min = max(0.4, float(np.percentile(y_train[:, 1], 1)))
    v_max = float(np.percentile(y_train[:, 1], 99))

    clean_base_features = frame[feature_columns(frame)].copy()
    best_models = list(MODEL_DIR.glob("best_*_model.joblib"))

    if not best_models:
        print("⚠️ No best_*_model.joblib found in MODEL_DIR. Please run train.py first.")
        return

    output_dir = DATA_DIR

    for model_path in best_models:
        scenario_name = model_path.name.replace("best_", "").replace("_model.joblib", "")

        try:
            pipeline = joblib.load(model_path)
        except Exception as e:
            print(f"❌ Failed to load {model_path.name}: {e}")
            continue

        if scenario_name == "baseline":
            features = clean_base_features.copy()
        else:
            raw_emb_name = scenario_name.replace("llm_enhanced_", "")
            emb_file = Path(EMBEDDING_DIR) / f"{raw_emb_name}_embeddings.npy"

            if not emb_file.exists():
                continue

            embeddings = np.load(emb_file)
            features = add_embeddings(
                clean_base_features, embeddings,
                train_index=train_index, test_index=test_index
            )

        columns = list(features.columns)
        numeric = numeric_columns(features, columns)
        features_coerced = coerce_feature_frame(features, columns, numeric)
        X_test = features_coerced.iloc[test_index]
        raw_pred = pipeline.predict(X_test)

        if raw_pred.shape[1] > 1:
            raw_pred[:, 1] = np.clip(raw_pred[:, 1], v_min, v_max)
        y_pred = np.clip(raw_pred, 0.0, None)

        if scenario_name == "llm_enhanced_qwen3-14b":
            output_dir.mkdir(parents=True, exist_ok=True)
            df_out = pd.DataFrame()
            for i, target in enumerate(TARGET_COLUMNS):
                df_out[f"{target}_True"] = y_test[:, i]
                df_out[f"{target}_Pred_Mu"] = y_pred[:, i]
                if hasattr(pipeline, "estimators_"):
                    try:
                        dist = pipeline.estimators_[i].pred_dist(X_test.to_numpy(dtype=float))
                        df_out[f"{target}_Pred_Sigma"] = dist.params["scale"]
                    except Exception:
                        pass
            df_out.to_csv(output_dir / "pred_vs_true_llm_enhanced_qwen3-14b.csv", index=False)

        r2_scores = []
        for i, target in enumerate(TARGET_COLUMNS):
            r2 = float(r2_score(y_test[:, i], y_pred[:, i]))
            r2_scores.append(r2)

        mean_r2 = float(np.mean(r2_scores))

        print(f" {scenario_name.ljust(20)}")
        print(f"   ➤ Mean R2: {mean_r2:.4f}  |  (PCE R2: {r2_scores[0]:.4f}, Voc R2: {r2_scores[1]:.4f}, Jsc R2: {r2_scores[2]:.4f}, FF R2: {r2_scores[3]:.4f})")
        print("-" * 80)


if __name__ == "__main__":
    main()