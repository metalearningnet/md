import sys
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

_root_dir = Path(__file__).parent.parent
_lm_dir = _root_dir / 'models' / 'lm'
_conf_dir = _root_dir / 'conf'
sys.path.append(str(_conf_dir))

from settings import MODEL

def download_model(name):
    if name == 'lm':
        model_path = MODEL['lm']['path']
        save_dir = _lm_dir
    else:
        raise ValueError(f"Invalid model name: {name}")
    
    print(f"Downloading {model_path}...")
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print(f"Saving {name} model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

def main():
    parser = argparse.ArgumentParser(description="A downloader for MD")
    parser.add_argument("--lm", action="store_true", default=False, help="Download the language model")
    
    args = parser.parse_args()
    
    if not args.lm:
        parser.print_help()
        print("No model selected.")
        sys.exit(1)
    
    download_model('lm')

if __name__ == "__main__":
    main()
