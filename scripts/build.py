import sys
import shutil
import argparse
from pathlib import Path
from huggingface_hub import model_info

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD
from utils import Cfg, get_ckpt_path, get_md_dir, info, cfg

def validate_and_copy_model():
    if cfg.lm_path == cfg.md_path:
        raise ValueError(f"Hugging Face model path must differ from model directory: '{cfg.md_path}'")
    
    ckpt_path = get_ckpt_path()
    if not ckpt_path or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: '{ckpt_path}'")

    md_cfg = Cfg.load()
    md = MD.from_pretrained(ckpt_path=ckpt_path)
    
    model_type = model_info(md_cfg.lm_path).config['model_type']
    if md.lm_config.model_type != model_type:
        raise ValueError(f"Model type mismatch: expected '{model_type}', got '{md.lm_config.model_type}'")

    dest = get_ckpt_path(parent=get_md_dir())
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt_path, dest)
    info(f"Checkpoint copied to: '{dest}'")

def main():
    parser = argparse.ArgumentParser(
        description="Build model components."
    )
    parser.add_argument(
        'command',
        nargs='?',
        choices=['lm'],
        help="Command to run: 'lm' to convert MD model to language model"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'lm':
            validate_and_copy_model()
        else:
            raise ValueError(f"unknown command {args.command}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
