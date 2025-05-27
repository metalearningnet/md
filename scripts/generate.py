import sys
import json
import torch
import argparse
import pandas as pd
import lightning as L
from tqdm import tqdm 
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download

_src_dir = Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD
from utils import cfg

DEFAULT_DATASET = {
    'path': 'tatsu-lab/alpaca_eval',
    'name': 'alpaca_eval.json'
}

def get_eval_set(path, name):
    if name.endswith('.json'):
        try:
            file_path = hf_hub_download(
                repo_id=path,
                filename=name,
                repo_type='dataset'
            )
            return pd.read_json(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load JSON dataset from {name}: {str(e)}")
    else:
        try:
            return load_dataset(path, name, trust_remote_code=True)['eval']
        except Exception as e:
            raise ValueError(f"Failed to load dataset {name}: {str(e)}")

def generate_response(model, prompt):
    try:
        device = next(model.parameters()).device
        inputs = model.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs)
        return model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    except RuntimeError as e:
        if "MPS device" in str(e):
            inputs = model.tokenizer(prompt, return_tensors='pt').to('cpu')
            with torch.no_grad():
                outputs = model.generate(**inputs)
            return model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raise RuntimeError(f"Failed to generate response: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {str(e)}")
    
def generate(config: dict):
    """
    config:
        - path: Dataset path.
        - name: Dataset name.
        - ckpt: Checkpoint path.
        - fabric_config: Configuration options for the Lightning Fabric setup.
        - generator: LLM name.
        - output_file: Output file path.
    """
    try:
        dataset_path = config['path']
        dataset_name = config.get('name')
        ckpt_path = config.get('ckpt')
        fabric_config = config['fabric_config']
        generator = config['generator']
        output_file = config['output_file']
        
        fabric = L.Fabric(**fabric_config)
        fabric.launch()
        
        if ckpt_path:
            model = MD.from_pretrained(checkpoint_path=ckpt_path)
        else:
            model = MD.from_pretrained()

        model = fabric.setup(model)
        model.eval()

        results = []
        eval_set = get_eval_set(dataset_path, dataset_name)

        progress_bar = tqdm(
            eval_set.iterrows(),
            total=len(eval_set),
            desc="Generating responses",
            unit="example"
        )

        for _, example in progress_bar:
            try:
                response = generate_response(model, example['instruction'])
                results.append({
                    **example.to_dict(),
                    'output': response,
                    'generator': generator
                })
            except Exception as e:
                print(f"Error processing example: {str(e)}")
                continue
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    
def main():
    parser = argparse.ArgumentParser(description="Generate outputs")

    parser.add_argument("--path", type=str,
                        help="Dataset path")
    parser.add_argument("--name", type=str, default=None,
                        help="Dataset name")
    parser.add_argument("--ckpt", type=str, default=cfg.ckpt_path,
                        help="Checkpoint path")
    parser.add_argument("--out", type=str,
                        help="Output file path")
    parser.add_argument("--generator", type=str, default=cfg.lm_name,
                        help="LLM name")

    args = parser.parse_args()
    
    if not args.path:
        args.path = DEFAULT_DATASET['path']
        args.name = DEFAULT_DATASET['name']
    
    if not args.out:
        cfg.eval_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(cfg.eval_dir / f'{Path(args.name).stem}.json')
    
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint path not found: {args.ckpt}")
    
    fabric_config = {
        'accelerator': cfg.accelerator,
        'precision': cfg.precision
    }
    
    config = {
        'path': args.path,
        'name': args.name,
        'ckpt': args.ckpt,
        'output_file': args.out, 
        'generator': args.generator,
        'fabric_config': fabric_config
    }

    generate(config)

if __name__ == '__main__':
    main()