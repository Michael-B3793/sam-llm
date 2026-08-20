import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from config import EMBEDDING_DIR, ensure_directories
from data_utils import load_dataset, write_prompts

TARGET_DIR = "new model"
os.makedirs(TARGET_DIR, exist_ok=True)
ensure_directories()


model_ids = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen1.5-0.5B-Chat",
    "facebook/opt-350m",
    "HuggingFaceTB/SmolLM-360M-Instruct",
    "EleutherAI/pythia-410m"
]

print("📂 Loading dataset and generating prompts...")
frame = load_dataset()
prompt_path = EMBEDDING_DIR / "temp_5models_prompts.jsonl"
records = write_prompts(frame, prompt_path)
texts = [record["text"] for record in records]
print(f"✅ Created {len(texts)} prompts for extraction.")


def pool_hidden_state(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    masked = hidden_state * mask
    counts = mask.sum(dim=1).clamp(min=1)
    return masked.sum(dim=1) / counts


for model_id in model_ids:
    model_name_short = model_id.split("/")[-1]
    save_path = os.path.join(TARGET_DIR, f"{model_name_short}_embeddings.npy")

    if os.path.exists(save_path):
        print(f"\n⏭️ Skipping {model_name_short}: Already exists.")
        continue

    print(f"\n=============================================")
    print(f"🚀 Processing: {model_name_short}")
    print(f"=============================================")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=TARGET_DIR, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=TARGET_DIR,
            torch_dtype=torch.float16,
            device_map="auto",

            max_memory={0: "3.5GiB", "cpu": "8GiB"},
            trust_remote_code=True
        )
        model.eval()

        all_embeddings = []

        max_length = 512

        with torch.inference_mode():
            for text in tqdm(texts, desc="Extracting"):
                encoded = tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(model.device)

                output = model(**encoded, output_hidden_states=True, use_cache=False)
                pooled = pool_hidden_state(output.hidden_states[-1], encoded["attention_mask"])
                all_embeddings.append(pooled.float().cpu().numpy().astype("float32"))

                del encoded, output, pooled
                torch.cuda.empty_cache()

        final_matrix = np.vstack(all_embeddings)
        np.save(save_path, final_matrix)
        print(f"💾 Saved {save_path} (Shape: {final_matrix.shape})")

    except Exception as e:
        print(f"❌ Error on {model_name_short}: {e}")

    finally:
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

print("\n🎉 All extractions completed successfully!")