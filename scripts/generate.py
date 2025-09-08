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
from utils import cfg, get_device, get_initial_prompt

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

def apply_chat_template(messages, tokenizer):
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            return text[len(tokenizer.bos_token):]
        return text
    except Exception as e:
        print(f"Failed to format messages: {messages}")
        raise e

def generate_response(model, prompt, skip_special_tokens=False):
    device = get_device()
    messages = [{'role': 'user', 'content': prompt}]
    initial_prompt = get_initial_prompt()
    if initial_prompt:
        messages = initial_prompt + messages
    prompt = apply_chat_template(messages, model.tokenizer)
    try:
        inputs = model.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'])
        return prompt, model.tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)
    except RuntimeError as e:
        if "MPS device" in str(e):
            inputs = model.tokenizer(prompt, return_tensors='pt').to('cpu')
            with torch.no_grad():
                outputs = model.generate(**inputs)
            return prompt, model.tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)
        raise RuntimeError(f"Failed to generate response: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {str(e)}")
    
def generate(config: dict):
    """
    config:
        - path: Dataset path.
        - name: Config name.
        - ckpt: Checkpoint path.
        - examples: Number of examples.
        - fabric_config: Configuration options for the Lightning Fabric setup.
        - lm: LLM name.
        - output_file: Output file path.
        - skip_special_tokens: Hides special tokens in the output.
    """
    try:
        dataset_path = config['path']
        dataset_name = config.get('name')
        ckpt_path = config.get('ckpt')
        examples = config.get('examples', -1)
        fabric_config = config['fabric_config']
        lm = config['lm']
        output_file = config['output_file']
        skip_special_tokens = config['skip_special_tokens']
        
        fabric = L.Fabric(**fabric_config)
        fabric.launch()

        if ckpt_path:
            model = MD.from_pretrained(checkpoint_path=ckpt_path)
        else:
            model = MD.from_pretrained()

        model = fabric.setup(model)
        model.eval()

        if model.has_anno:
            model.mark_forward_method('generate')

        results = []
        eval_set = get_eval_set(dataset_path, dataset_name)

        total_examples = len(eval_set)
        if examples != -1:
            total_examples = min(examples, total_examples)

        progress_bar = tqdm(
            eval_set.head(total_examples).iterrows(),
            total=total_examples,
            desc="Generating responses"
        )

        for _, example in progress_bar:
            try:
                prompt, response = generate_response(model, example['instruction'], skip_special_tokens)
                results.append({
                    'dataset': example['dataset'],
                    'instruction': example['instruction'],
                    'output': response[len(prompt):],
                    'lm': lm
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
                        help="Config name")
    parser.add_argument("--examples", type=int, default=-1,
                        help="Number of examples to generate responses for")
    parser.add_argument("--ckpt", type=str, default=cfg.ckpt_path,
                        help="Checkpoint path")
    parser.add_argument("--out", type=str,
                        help="Output file path")
    parser.add_argument("--lm", type=str, default=cfg.lm_name,
                        help="LLM name")
    parser.add_argument("--skip-special-tokens", action="store_true", default=False,
                        help="Hides special tokens in the output")

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
        'precision': cfg.precision
    }
    
    config = {
        'path': args.path,
        'name': args.name,
        'ckpt': args.ckpt,
        'examples': args.examples,
        'output_file': args.out, 
        'lm': args.lm,
        'fabric_config': fabric_config,
        'skip_special_tokens': args.skip_special_tokens
    }

    generate(config)

if __name__ == '__main__':
    main()
