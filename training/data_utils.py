import json
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from config import RESULTS_DIR, DATASET_PATH, IDENTITY_COLUMNS, TARGET_COLUMNS, TEXT_COLUMNS, PROTECTED_PROCESS_COLUMNS

WORK_FUNCTION_COLUMN = "Substrate" + "\u00a0" + "Work" + "\u00a0" + "Function"

def clean_column_name(value):
    return str(value).replace("\n", " ").strip()

def normalize_text(value):
    if pd.isna(value):
        return "not reported"
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return "not reported"
    return re.sub(r"\s+", " ", text)

def load_dataset():
    frame = pd.read_excel(DATASET_PATH)
    return frame

def build_group_labels(frame):
    names = frame["name"].map(normalize_text) if "name" in frame.columns else pd.Series(["unknown"] * len(frame))
    smiles = frame["SMILES"].map(normalize_text) if "SMILES" in frame.columns else pd.Series(["unknown"] * len(frame))
    return (names.str.lower() + "::" + smiles.str.lower()).to_numpy()

def build_sample_text(row):
    values = {column: normalize_text(row[column]) for column in TEXT_COLUMNS if column in row.index}
    parts = [
        f"SAM name: {values.get('name', 'not reported')}.",
        f"SMILES: {values.get('SMILES', 'not reported')}.",
        f"Substrate: {values.get('Substrate Type', 'not reported')}; work function: {values.get(WORK_FUNCTION_COLUMN, 'not reported')}.",
        f"SAM process: concentration {values.get('SAM Solution Concentration (mM)', 'not reported')} mM in {values.get('SAM Solvent', 'not reported')}; soaking {values.get('Soaking', 'not reported')} for {values.get('Soak Time', 'not reported')} min; spin coating {values.get('Spin Coating', 'not reported')} at {values.get('Spin Coating Speed (rpm)', 'not reported')} rpm for {values.get('Spin Coating Time (s)', 'not reported')} s.",
        f"Thermal treatment: annealing {values.get('Annealing', 'not reported')}; initial temperature {values.get('Initial Annealing Temperature for SAM (°C)', 'not reported')} C for {values.get('：Initial Annealing Time for SAM (min)', 'not reported')} min; post-washing temperature {values.get('Post-washing Annealing Temperature (°C)', 'not reported')} C for {values.get('Post-washing Annealing Time (min)', 'not reported')} min; UV-ozone {values.get('UV-Ozone Treatment Time (min)', 'not reported')} min.",
        f"Device context: carrier role {values.get('Carrier_Role', 'not reported')}; active layer {values.get('Active Layer', 'not reported')}; device type {values.get('Device Type', 'not reported')}.",
    ]
    return " ".join(parts)

def write_prompts(frame, path):
    records = []
    for index, row in frame.iterrows():
        record = {"index": int(index), "text": build_sample_text(row)}
        records.append(record)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records

def feature_columns(frame):
    excluded = set(TARGET_COLUMNS + IDENTITY_COLUMNS)
    return [column for column in frame.columns if column not in excluded]

def numeric_columns(frame, columns):
    result = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        ratio = float(values.notna().mean())
        if column.startswith("Bit_") or ratio >= 0.75:
            result.append(column)
    return result

def categorical_columns(frame, columns, numeric):
    numeric_set = set(numeric)
    return [column for column in columns if column not in numeric_set]

def coerce_feature_frame(frame, columns, numeric):
    output = frame[columns].copy()
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in output.columns:
        if column not in numeric:
            output[column] = output[column].map(normalize_text)
    return output


def add_embeddings(features, embeddings, train_index=None, test_index=None, n_components=32):
    from sklearn.decomposition import PCA
    import pandas as pd

    if train_index is not None and test_index is not None:
        emb_train = embeddings[train_index]
        actual_components = min(n_components, emb_train.shape[0], emb_train.shape[1])

        pca_raw = PCA(n_components=actual_components, random_state=42)
        pca_raw.fit(emb_train)

        reduced_embeddings = pca_raw.transform(embeddings)
    else:
        actual_components = min(n_components, embeddings.shape[0], embeddings.shape[1])

        pca_raw = PCA(n_components=actual_components, random_state=42)
        reduced_embeddings = pca_raw.fit_transform(embeddings)

    columns = [f"llm_pca_{index:04d}" for index in range(actual_components)]
    embedding_frame = pd.DataFrame(reduced_embeddings, columns=columns, index=features.index)

    return pd.concat([features.copy(), embedding_frame], axis=1)

def compact_json(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

def finite_array(values):
    return np.asarray(values, dtype=float)


