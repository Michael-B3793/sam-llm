import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA_DIR, EMBEDDING_DIR, LLM_MODEL_DIR, LLM_MODEL_ID, LLM_MODEL_NAME, ensure_directories
from data_utils import load_dataset, write_prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(LLM_MODEL_DIR))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def model_ready(model_dir):
    return os.path.exists(os.path.join(model_dir, "config.json"))


def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )
    model.eval()
    return tokenizer, model


def pool_hidden_state(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    masked = hidden_state * mask
    counts = mask.sum(dim=1).clamp(min=1)
    return masked.sum(dim=1) / counts


def encode_texts(texts, tokenizer, model, batch_size, max_length):
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            pooled = pool_hidden_state(output.hidden_states[-1], encoded["attention_mask"])
            embeddings.append(pooled.float().cpu().numpy().astype("float32"))
            print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(embeddings)


def main():
    ensure_directories()
    args = parse_args()
    if not model_ready(args.model_dir):
        print(f"{LLM_MODEL_ID} is not downloaded.")
        print("Run: /opt/anaconda3/bin/conda run -n OPVLLM python training/download_model.py")
        sys.exit(2)
    frame = load_dataset()
    if args.limit:
        frame = frame.head(args.limit).copy()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    prompt_path = EMBEDDING_DIR / f"{LLM_MODEL_NAME}_prompts.jsonl"
    records = write_prompts(frame, prompt_path)
    tokenizer, model = load_model(args.model_dir)
    texts = [record["text"] for record in records]
    embeddings = encode_texts(texts, tokenizer, model, args.batch_size, args.max_length)
    np.save(EMBEDDING_DIR / f"{LLM_MODEL_NAME}_embeddings.npy", embeddings)
    metadata = {
        "model_dir": args.model_dir,
        "samples": int(len(frame)),
        "embedding_dim": int(embeddings.shape[1]),
        "max_length": int(args.max_length),
    }
    with open(EMBEDDING_DIR / f"{LLM_MODEL_NAME}_embedding_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print("Embedding extraction complete.")


if __name__ == "__main__":
    main()
