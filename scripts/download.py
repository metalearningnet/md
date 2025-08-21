import os
import sys
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

_root_dir = Path(__file__).parent.parent
_model_dir = _root_dir / 'model'
_conf_dir = _root_dir / 'conf'
sys.path.append(str(_conf_dir))
from settings import MODEL

def download_model():
    if not _model_dir.exists():
        os.mkdir(_model_dir)

    model_path = MODEL['lm']['path']
    model_name = os.path.basename(model_path)

    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.save_pretrained(_model_dir)
    tokenizer.save_pretrained(_model_dir)

def main():
    parser = argparse.ArgumentParser(description="A downloader for MD")
    parser.add_argument("--model", action="store_true", default=False, help="Download model")
    args = parser.parse_args()
    if args.model:
        download_model()

if __name__ == "__main__":
    main()
