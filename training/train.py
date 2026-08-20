import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from ngboost import NGBRegressor

from config import (
    DATA_DIR,
    EMBEDDING_DIR,
    LLM_MODEL_NAME,
    METRIC_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    TARGET_COLUMNS,
    RANDOM_SEED,
    ensure_directories,
)

from data_utils import (
    add_embeddings,
    build_group_labels,
    categorical_columns,
    coerce_feature_frame,
    compact_json,
    feature_columns,
    load_dataset,
    numeric_columns,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def build_preprocessor(numeric, categorical):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], sparse_threshold=0.0, verbose_feature_names_out=False)


def create_models(scenario_name, config, seed):
    scenario_config = config.get(scenario_name)
    if scenario_config is None:
        clean_name = scenario_name.replace("llm_enhanced_", "")
        scenario_config = config.get(clean_name, config.get("baseline", {}))

    models = {}
    for model_name, params in scenario_config.items():
        p = params.copy()
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])

        if model_name == "extra_trees":
            models[model_name] = ExtraTreesRegressor(**p, random_state=seed, n_jobs=-1)
        elif model_name == "random_forest":
            models[model_name] = RandomForestRegressor(**p, random_state=seed, n_jobs=-1)
        elif model_name == "xgboost":
            models[model_name] = XGBRegressor(**p, random_state=seed, n_jobs=-1)
        elif model_name == "svr":
            models[model_name] = MultiOutputRegressor(SVR(**p))
        elif model_name == "k_neighbors":
            models[model_name] = KNeighborsRegressor(**p, n_jobs=-1)
        elif model_name in ["deep_mlp", "wide_mlp"]:
            models[model_name] = MultiOutputRegressor(MLPRegressor(**p, random_state=seed))
        elif model_name == "ngboost":
            models[model_name] = MultiOutputRegressor(NGBRegressor(**p, random_state=seed, verbose=False))
        elif model_name == "vector_xgboost":
            models[model_name] = XGBRegressor(**p, random_state=seed, n_jobs=-1)
        elif model_name == "modern_mlp":
            models[model_name] = MLPRegressor(**p, random_state=seed)

    return models


def target_metrics(y_true, y_pred):
    records = []
    for index, target in enumerate(TARGET_COLUMNS):
        true = y_true[:, index]
        pred = y_pred[:, index]
        rmse = mean_squared_error(true, pred) ** 0.5
        scale = float(np.std(true)) or 1.0
        records.append({
            "target": target,
            "mae": float(mean_absolute_error(true, pred)),
            "rmse": float(rmse),
            "nrmse": float(rmse / scale),
            "r2": float(r2_score(true, pred)),
        })

    summary = {
        "mean_mae": float(np.mean([record["mae"] for record in records])),
        "mean_rmse": float(np.mean([record["rmse"] for record in records])),
        "mean_nrmse": float(np.mean([record["nrmse"] for record in records])),
        "mean_r2": float(np.mean([record["r2"] for record in records])),
    }
    return records, summary


def split_data(frame, groups, test_size, seed):
    train_index, test_index = train_test_split(
        np.arange(len(frame)), test_size=test_size, random_state=seed
    )
    return train_index, test_index


def save_split(frame, train_index, test_index, groups):
    split = pd.DataFrame({
        "row_index": np.arange(len(frame)),
        "group": groups,
        "split": "train",
        "device_type": frame.get("Device Type", pd.Series(["unknown"] * len(frame))).astype(str),
    })
    split.loc[test_index, "split"] = "test"
    split.to_csv(DATA_DIR / "split_assignments.csv", index=False)


def feature_importance(pipeline, output_path, X_test, y_test):
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=2, random_state=42, n_jobs=1
    )
    names = X_test.columns.tolist()
    values = np.clip(result.importances_mean, 0.0, None)
    frame = pd.DataFrame({"feature": names, "importance": values})
    frame_sorted = frame.sort_values("importance", ascending=False).head(120)
    frame_sorted[["feature", "importance"]].to_csv(output_path, index=False)


def fit_scenario(name, features, y, train_index, test_index, seed, config, display_name):
    print(f"\n🚀 Fitting scenario: [{display_name}]")
    columns = list(features.columns)
    numeric = numeric_columns(features, columns)
    categorical = categorical_columns(features, columns, numeric)
    features_coerced = coerce_feature_frame(features, columns, numeric)

    X_train, y_train = features_coerced.iloc[train_index], y[train_index]
    X_test, y_test = features_coerced.iloc[test_index], y[test_index]

    results, predictions, fitted_models = [], [], {}
    models_dict = create_models(name, config, seed)

    for model_name, estimator in models_dict.items():
        print(f"  -> Training {model_name:<16} ...", end="", flush=True)
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        raw_test_pred = pipeline.predict(X_test)

        if raw_test_pred.shape[1] > 1:
            v_min = max(0.4, float(np.percentile(y_train[:, 1], 1)))
            v_max = float(np.percentile(y_train[:, 1], 99))
            raw_test_pred[:, 1] = np.clip(raw_test_pred[:, 1], v_min, v_max)

        final_test_pred = np.clip(raw_test_pred, 0.0, None)
        metric_records, summary = target_metrics(y_test, final_test_pred)

        results.append({
            "scenario": display_name,
            "model": model_name,
            "metrics": metric_records,
            "summary": summary,
        })
        fitted_models[model_name] = pipeline

        y_pred_all = np.clip(pipeline.predict(features_coerced), 0.0, None)
        if y_pred_all.shape[1] > 1:
            y_pred_all[:, 1] = np.clip(y_pred_all[:, 1], v_min, v_max)

        split_labels = np.array(["train"] * len(features_coerced), dtype=object)
        split_labels[test_index] = "test"

        prediction_frame = pd.DataFrame({
            "row_index": np.arange(len(features_coerced)),
            "scenario": display_name,
            "model": model_name,
            "split": split_labels,
        })
        for target_index, target in enumerate(TARGET_COLUMNS):
            prediction_frame[f"true_{target}"] = y[:, target_index]
            prediction_frame[f"pred_{target}"] = y_pred_all[:, target_index]
        predictions.append(prediction_frame)

        r2_details = ", ".join([
            f"{rec['target'].split()[-1] if ' ' in rec['target'] else rec['target']}: {rec['r2']:.4f}"
            for rec in metric_records
        ])
        print(f" Done! (Mean R2: {summary['mean_r2']:.4f} | {r2_details})")

    best = max(results, key=lambda record: record["summary"]["mean_r2"])
    print(f"🏆 [{display_name}] Training finished. Best model: {best['model']} (R2: {best['summary']['mean_r2']:.4f})")
    return results, pd.concat(predictions, ignore_index=True), fitted_models, best


def main():
    ensure_directories()
    args = parse_args()

    cs_path = METRIC_DIR / "cs.json"
    if not cs_path.exists():
        raise FileNotFoundError("cs.json not found.")
    config = json.load(open(cs_path, "r"))

    frame = load_dataset()
    frame.to_csv(DATA_DIR / "clean_dataset.csv", index=False)

    groups = build_group_labels(frame)
    train_index, test_index = split_data(frame, groups, args.test_size, args.seed)
    save_split(frame, train_index, test_index, groups)

    y = frame[TARGET_COLUMNS].to_numpy(dtype=float)
    clean_base_features = frame[feature_columns(frame)].copy()

    all_results, all_predictions, best_models = [], [], {}

    results, predictions, all_models, best = fit_scenario(
        "baseline", clean_base_features, y, train_index, test_index, args.seed, config, "baseline"
    )
    all_results.extend(results)
    all_predictions.append(predictions)

    for model_name, pipeline in all_models.items():
        joblib.dump(pipeline, MODEL_DIR / f"baseline_{model_name}_model.joblib")
    joblib.dump(all_models[best["model"]], MODEL_DIR / "best_baseline_model.joblib")

    best_models["baseline"] = {"pipeline": all_models[best["model"]], "record": best}

    if not args.baseline_only:
        for path in Path(EMBEDDING_DIR).glob("*_embeddings.npy"):
            if path.name.startswith("._"):
                continue

            name = path.stem.replace("_embeddings", "")
            scenario_name = name if "llm" in name else f"llm_enhanced_{name}"

            embeddings = np.load(path)
            if embeddings.shape[0] == len(frame):
                enhanced_features = add_embeddings(
                    clean_base_features, embeddings, train_index=train_index, test_index=test_index
                )

                results, predictions, all_models, best = fit_scenario(
                    name, enhanced_features, y, train_index, test_index, args.seed, config, scenario_name
                )

                all_results.extend(results)
                all_predictions.append(predictions)

                for model_name, pipeline in all_models.items():
                    joblib.dump(pipeline, MODEL_DIR / f"{scenario_name}_{model_name}_model.joblib")
                joblib.dump(all_models[best["model"]], MODEL_DIR / f"best_{scenario_name}_model.joblib")

                best_models[scenario_name] = {"pipeline": all_models[best["model"]], "record": best}

    if all_results:
        metric_payload = {
            "targets": TARGET_COLUMNS,
            "records": all_results,
            "best_by_scenario": {name: value["record"] for name, value in best_models.items()},
        }
        compact_json(metric_payload, METRIC_DIR / "metrics.json")

        prediction_frame = pd.concat(all_predictions, ignore_index=True)
        device = frame.get("Device Type", pd.Series(["unknown"] * len(frame))).astype(str)
        prediction_frame["device_type"] = prediction_frame["row_index"].map(device.to_dict())
        prediction_frame.to_csv(PREDICTION_DIR / "test_predictions.csv", index=False)

        enhanced_keys = [k for k in best_models.keys() if k != "baseline"]
        if enhanced_keys:
            best_scenario = max(enhanced_keys, key=lambda k: best_models[k]["record"]["summary"]["mean_r2"])
            selected = best_models[best_scenario]
        else:
            best_scenario = "baseline"
            selected = best_models["baseline"]

        columns = list(clean_base_features.columns)
        if best_scenario != "baseline":
            raw_embedding_name = best_scenario.replace("llm_enhanced_", "")
            best_path = Path(EMBEDDING_DIR) / f"{raw_embedding_name}_embeddings.npy"
            if best_path.exists():
                embeddings = np.load(best_path)
                features_coerced = add_embeddings(
                    clean_base_features, embeddings, train_index=train_index, test_index=test_index
                )
            else:
                features_coerced = clean_base_features
        else:
            features_coerced = clean_base_features

        numeric = numeric_columns(features_coerced, list(features_coerced.columns))
        features_coerced_clean = coerce_feature_frame(features_coerced, list(features_coerced.columns), numeric)

        X_test_best = features_coerced_clean.iloc[test_index]
        y_test_best = y[test_index]

        print("\n⏳ Calculating feature importance (Permutation Importance)...")
        feature_importance(selected["pipeline"], METRIC_DIR / "feature_importance.csv", X_test_best, y_test_best)
        print("✅ Feature importance calculation complete!")
        print("\n🎉 Evaluation complete! Report saved to metrics.json")


if __name__ == "__main__":
    main()