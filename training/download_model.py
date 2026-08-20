import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from config import LLM_MODEL_DIR, LLM_MODEL_ID, LLM_MODEL_NAME, ensure_directories


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=LLM_MODEL_ID)
    parser.add_argument("--output-dir", default=str(LLM_MODEL_DIR))
    parser.add_argument("--confirm-vpn-closed", action="store_true")
    return parser.parse_args()


def confirm_download(args):
    if args.confirm_vpn_closed:
        return
    print(f"About to download {args.model_id} to {LLM_MODEL_NAME}.")
    print("Type 'DOWNLOAD_MODEL' to proceed; any other input will cancel.")
    answer = input("> ").strip()
    if answer != "DOWNLOAD_MODEL":
        print("Model download cancelled.")
        sys.exit(2)


def check_access(model_id):
    api = HfApi()
    try:
        api.model_info(model_id)
    except GatedRepoError:
        print("Model access is gated. Log in to Hugging Face and accept the model license first.")
        sys.exit(1)
    except RepositoryNotFoundError:
        print(f"Model repository not found or not accessible: {model_id}")
        sys.exit(1)
    except HfHubHTTPError as error:
        print(f"Hugging Face access check failed: {error}")
        sys.exit(1)


def main():
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    ensure_directories()
    args = parse_args()
    confirm_download(args)
    check_access(args.model_id)
    output_dir = args.output_dir
    print(f"Downloading {args.model_id} to {output_dir}")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=output_dir,
    )
    print("Model download complete.")


if __name__ == "__main__":
    main()