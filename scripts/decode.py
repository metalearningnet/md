# ==============================================================================
# This file originates from the SimPO repository:
# Repository: https://github.com/princeton-nlp/SimPO
#
# SimPO (Simple Preference Optimization) is developed by Princeton NLP.
# The original work and its associated documentation can be found in the repository.
#
# For comprehensive details about the implementation, licensing, and updates,
# please refer to the original repository and its LICENSE file.
# ==============================================================================

from vllm import LLM, SamplingParams
from datasets import load_dataset
from pathlib import Path
import argparse
import json
import os

_root_dir = Path(__file__).parent.parent
_output_dir = _root_dir / 'datasets' / 'gemma2-ultrafeedback-armorm'

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--data_dir', type=str, default="princeton-nlp/gemma2-ultrafeedback-armorm",
                    help='Directory containing the data')
parser.add_argument('--model', type=str, default='google/gemma-2-9b-it',
                    help='Path to the LLM model')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--output_dir', type=str, default=_output_dir,
                    help='output_dir')
args = parser.parse_args()

print(args)

data_dir = args.data_dir
llm = LLM(model=args.model)
tokenizer = llm.get_tokenizer()

train_dataset= load_dataset(data_dir, split='train')

prompts = sorted(list(set(train_dataset['prompt'])))

conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

sampling_params = SamplingParams(temperature=args.temperature, 
                                 top_p=args.top_p, 
                                 max_tokens=args.max_tokens, 
                                 seed=args.seed,)
outputs = llm.generate(conversations, sampling_params)

# Save the outputs as a JSON file.
output_data = []
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    output_data.append({
        'prompt': prompts[i],
        "format_prompt": prompt,
        'generated_text': generated_text,
    })

output_file = f'output_{args.seed}.json'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
