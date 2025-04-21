
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

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import argparse
import tqdm
import numpy as np
import datasets
from pathlib import Path

_root_dir = Path(__file__).parent.parent
_output_dir = _root_dir / 'datasets' / 'gemma2-ultrafeedback-armorm'
_generation_file = _output_dir / 'all_outputs.json'

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file", type=str, default=_generation_file, help="Path to the output generation file")
parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1", help="Path to reward model")
parser.add_argument("--output_dir", type=str, default=_output_dir, help="Path to output directory")
args = parser.parse_args()

print(args)

device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

generation_file = args.generation_file
with open(generation_file, 'r') as f:
    output_data = json.load(f)

inputs = [data["prompt"] for data in output_data]
candidates_texts = [data["all_generated_responses"] for data in output_data]

model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, 
                                                           device_map=device, 
                                                           trust_remote_code=True, 
                                                           torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)

for data in tqdm.tqdm(output_data):
    prompt = data["prompt"]
    candidates = data["all_generated_responses"]
    scores = []
    for candidate in candidates:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": candidate}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(input_ids)
            score = output.score.float().item()
            scores.append(score)
    data["all_rm_scores"] = scores

file_name = os.path.basename(args.generation_file).split('.json')[0] + "_rm.json"
with open(os.path.join(args.output_dir, file_name), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Annotated outputs saved to {os.path.join(args.output_dir, file_name)}")

# Binarize data: win = highest scoring reponse; lose = lowest scoring response
for data in output_data:
    chosen_idx = np.argmax(data["all_rm_scores"])
    rejected_idx = np.argmin(data["all_rm_scores"])
    chosen = []
    chosen.append({
        "role": "user",
        "content": data["prompt"]
    })
    chosen.append({
        "role": "assistant",
        "content": data["all_generated_responses"][chosen_idx]
    })
    rejected = []
    rejected.append({
        "role": "user",
        "content": data["prompt"]
    })
    rejected.append({
        "role": "assistant",
        "content": data["all_generated_responses"][rejected_idx]
    })
    data.update({
        "chosen": chosen,
        "rejected": rejected,
    })

output_file = os.path.basename(args.generation_file).split('.json')[0] + "_bin.json"
with open(os.path.join(args.output_dir, file_name), 'w') as f:
    json.dump(output_data, f, indent=4)
print(f"Binarized outputs saved to {output_file}")

# Convert the data to Hugging Face datasets format
dataset = datasets.Dataset.from_list(output_data)
dataset.save_to_disk(os.path.join(args.output_dir))
print(f"Binarized dataset saved to {os.path.join(args.output_dir)}")
