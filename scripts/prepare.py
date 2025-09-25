import json
import yaml
import tqdm
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SHOW_SCORES = False

track_output_format = set()

def show_analysis(data_idx, candidate_idx, output, config):
    if SHOW_SCORES:
        if data_idx < 3 and candidate_idx == 0:
            print(f"\n=== Output analysis for {config['reward_model']['name']} ===")
            print(f"Output type: {type(output)}")
            print(f"Output attributes: {[attr for attr in dir(output) if not attr.startswith('_')]}")

            if hasattr(output, 'logits'):
                print(f"Logits shape: {output.logits.shape}")
                print(f"Logits dtype: {output.logits.dtype}")
                print(f"Logits sample: {output.logits}")
                if output.logits.numel() == 1:
                    print(f"Single logit value: {output.logits.item()}")

            if hasattr(output, 'score'):
                print(f"Score: {output.score}")
                print(f"Score type: {type(output.score)}")

            for attr in ['last_hidden_state', 'hidden_states', 'attentions', 'loss']:
                if hasattr(output, attr):
                    val = getattr(output, attr)
                    if torch.is_tensor(val):
                        print(f"{attr} shape: {val.shape}")
                    else:
                        print(f"{attr}: {val}")

            print("=" * 58)

            output_format = []
            if hasattr(output, 'logits'):
                output_format.append('logits')
            if hasattr(output, 'score'):
                output_format.append('score')
            track_output_format.add(tuple(output_format))

def show_candidates(data_idx, candidates, prompt, scores):
    if SHOW_SCORES:
        if data_idx < 3:
            print(f"\nSample scores for prompt {data_idx}:")
            print(f"Prompt: {prompt[:100]}...")
            for i, (candidate, score) in enumerate(zip(candidates[:3], scores[:3])):
                print(f"  Candidate {i}: Score={score:.4f}, Text: {candidate[:50]}...")
            print("\n")

def show_summary(output_data, config):
    if SHOW_SCORES:
        print(f"\n=== Reward Model Output Summary ===")
        print(f"Model: {config['reward_model']['name']}")
        print(f"Output formats encountered: {track_output_format}")
        print(f"Number of examples processed: {len(output_data)}")
        print(f"Score range: {min(min(data['all_rm_scores']) for data in output_data):.4f} to {max(max(data['all_rm_scores']) for data in output_data):.4f}")
        print("=" * 58)
    
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_responses(config: Dict[str, Any], output_dir: Path) -> List[List[Dict[str, Any]]]:
    print("Loading dataset...")
    dataset = load_dataset(
        config['generation']['prompts_dataset'],
        split=config['generation']['split']
    )
    
    prompts_field = config['generation']['prompts_field']

    if isinstance(prompts_field, str):
        prompts = sorted(list(set(dataset[prompts_field])))
    elif isinstance(prompts_field, list):
        prompts = []
        seen_prompts = set()

        for example in dataset:
            prompt_parts = []
            for field in prompts_field:
                if field in example and example[field]:
                    if field == 'options':
                        # Special handling for options field
                        options_content = example[field]
                        if isinstance(options_content, list):
                            options_text = '\n'.join(options_content)
                        else:
                            options_text = str(options_content)
                        prompt_parts.append(f"Options: {options_text}")
                    else:
                        content = example[field]
                        if isinstance(content, list):
                            content = '\n'.join(content)
                        prompt_parts.append(str(content))
            
            combined_prompt = '\n'.join(prompt_parts)
            uniqueness_key = tuple(str(example.get(field, '')) for field in prompts_field)
            if uniqueness_key not in seen_prompts:
                seen_prompts.add(uniqueness_key)
                prompts.append(combined_prompt)
        
        prompts = sorted(prompts)
    else:
        raise ValueError(f"prompts_field must be string or list, got {type(prompts_field)}")
    
    print(f"Found {len(prompts)} unique prompts")

    max_examples = config['generation'].get('max_examples')
    if max_examples and max_examples > 0:
        prompts = prompts[:max_examples]
        print(f"Limiting to {max_examples} prompts")
    
    print(f"Initializing LLM: {config['generation']['model_name']}")
    llm = LLM(
        model=config['generation']['model_name'],
        gpu_memory_utilization=config['generation']['vllm']['gpu_memory_utilization'],
        max_model_len=config['generation']['vllm']['max_model_len']
    )
    
    tokenizer = llm.get_tokenizer()
    
    conversations = [
        tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': 'You are a helpful assistant. Provide accurate and detailed responses. Use clear line breaks between sentences to improve readability and organization.'},
                {'role': 'user', 'content': prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        ) for prompt in prompts
    ]
    
    all_output_data = []
    generation_file_prefix = config['processing']['generation_file_prefix']
    
    for seed in config['generation']['sampling']['seeds']:
        print(f"Generating responses with seed {seed}...")
        
        sampling_params = SamplingParams(
            temperature=config['generation']['sampling']['temperature'],
            top_p=config['generation']['sampling']['top_p'],
            max_tokens=config['generation']['sampling']['max_tokens'],
            seed=seed,
        )
        
        outputs = llm.generate(conversations, sampling_params)
        
        output_data = []
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            output_data.append({
                'seed': seed,
                'prompt': prompts[i],
                'format_prompt': prompt,
                'generated_text': generated_text
            })
        
        output_file = output_dir / f"{generation_file_prefix}_{seed}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Outputs for seed {seed} saved to {output_file}")
        
        all_output_data.append(output_data)
    
    return all_output_data

def filter_and_combine_responses(
    all_output_data: List[List[Dict[str, Any]]],
    config: Dict[str, Any],
    output_dir: Path
) -> List[Dict[str, Any]]:
    print("Filtering and combining responses...")
    
    all_res = []
    num_identical = 0
    num_examples = len(all_output_data[0])
    
    for i in range(num_examples):
        prompt = all_output_data[0][i]['prompt']
        gen_text = []
        for data in all_output_data:
            gen_text.append(data[i]['generated_text'])

        if config['processing']['filter_identical_responses'] and len(set(gen_text)) == 1:
            # Filter out examples where all generated responses are identical
            num_identical += 1
            continue

        all_res.append({
            'prompt': prompt,
            'all_generated_responses': gen_text,
        })
    
    print(f"Filtered out {num_identical} examples with identical generated responses")
    print(f"Remaining examples: {len(all_res)}")
    
    output_file = output_dir / "all_outputs.json"
    with open(output_file, 'w') as f:
        json.dump(all_res, f, indent=4)
    print(f"Combined outputs saved to {output_file}")
    
    return all_res

def score_with_reward_model(
    output_data: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path
) -> List[Dict[str, Any]]:
    print(f"Loading reward model: {config['reward_model']['name']}")
    
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config['reward_model']['name'],
        device_map=device,
        trust_remote_code=config['reward_model']['trust_remote_code'],
        torch_dtype=getattr(torch, config['reward_model']['dtype'])
    )
    tokenizer = AutoTokenizer.from_pretrained(config['reward_model']['name'], use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = config['generation']['sampling']['max_tokens']
    
    print("Scoring responses with reward model...")
    for data_idx, data in enumerate(tqdm.tqdm(output_data)):
        prompt = data['prompt']
        candidates = data['all_generated_responses']
        scores = []
        
        for candidate_idx, candidate in enumerate(candidates):
            messages = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': candidate}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            
            with torch.no_grad():
                output = model(**inputs)
                
                show_analysis(data_idx, candidate_idx, output, config)
                
                # Handle different output formats from different reward models
                if hasattr(output, 'score'):
                    score = output.score.float().item()
                elif hasattr(output, 'logits'):
                    if output.logits.numel() == 1:
                        # Single value logits
                        score = output.logits.item()
                    elif output.logits.dim() == 2 and output.logits.shape[1] == 1:
                        # [batch_size, 1] shape - single score per example
                        score = output.logits[0, 0].item()
                    else:
                        # Multi-dimensional logits
                        score = output.logits[0, -1].float().item()  # Last token approach
                else:
                    # Fallback: try to find the score in the output
                    score = 0.0
                    if hasattr(output, 'last_hidden_state'):
                        score = output.last_hidden_state.mean().item()
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        first_element = output[0]
                        if torch.is_tensor(first_element) and first_element.numel() == 1:
                            score = first_element.item()
                    
                    print(f"WARNING: Using fallback scoring. Output keys: {list(output.keys())}")
                
                scores.append(score)
        
        data['all_rm_scores'] = scores
        show_candidates(data_idx, candidates, prompt, scores)
    
    show_summary(output_data, config)
    scored_file = output_dir / config['output']['scored_file']
    with open(scored_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Scored outputs saved to {scored_file}")
    
    return output_data

def binarize_data(output_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Binarize data by selecting highest and lowest scoring responses."""
    print("Binarizing data...")
    
    for data in output_data:
        chosen_idx = np.argmax(data['all_rm_scores'])
        rejected_idx = np.argmin(data['all_rm_scores'])
        
        chosen = [
            {'role': 'user', 'content': data['prompt']},
            {'role': 'assistant', 'content': data['all_generated_responses'][chosen_idx]}
        ]
        
        rejected = [
            {'role': 'user', 'content': data['prompt']},
            {'role': 'assistant', 'content': data['all_generated_responses'][rejected_idx]}
        ]
        
        data.update({
            'chosen': chosen,
            'rejected': rejected,
        })
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description="Generate training data via reward LLM")
    parser.add_argument("--config", type=str, default=str(Path("conf") / "reward.yaml"),
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default=str(Path("output") / "dataset"),
                        help="Output directory")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    all_output_data = generate_responses(config, output_dir)
    combined_data = filter_and_combine_responses(all_output_data, config, output_dir)
    scored_data = score_with_reward_model(combined_data, config, output_dir)
    binarized_data = binarize_data(scored_data)
    binarized_file = output_dir / config['output']['binarized_file']
    with open(binarized_file, 'w') as f:
        json.dump(binarized_data, f, indent=4)
    print(f"Binarized outputs saved to {binarized_file}")
    
    if config['output']['save_hf_dataset']:
        dataset = Dataset.from_list(binarized_data)
        dataset.save_to_disk(output_dir)
        print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    main()
