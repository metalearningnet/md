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

import os
import json
import argparse
from pathlib import Path

_root_dir = Path(__file__).parent.parent
_output_dir = _root_dir / 'datasets' / 'gemma2-ultrafeedback-armorm'

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file_dir", type=str, help="Diretory containing the generation files", default=_output_dir)
args = parser.parse_args()

print(args)

all_data = []
for file_name in os.listdir(args.generation_file_dir):
    if file_name.startswith("output") and file_name.endswith(".json"):
        generation_file = os.path.join(args.generation_file_dir, file_name)
        with open(generation_file, 'r') as f:
            output_data = json.load(f)
            all_data.append(output_data)

num_samples = len(all_data[0])
all_res = []
num_identical = 0
for i in range(num_samples):
    prompt = all_data[0][i]["prompt"]
    gen_text = []
    for data in all_data:
        gen_text.append(data[i]["generated_text"])

    if len(set(gen_text)) == 1:
        # filter out samples where all generated responses are identical
        num_identical += 1
        continue

    all_res.append(
        {
            "prompt": prompt,
            "all_generated_responses": gen_text,
        }
    )

print(f"Filtered out {num_identical} samples with identical generated responses")

with open(os.path.join(args.generation_file_dir, 'all_outputs.json'), 'w') as f:
    json.dump(all_res, f, indent=4)

print(f"Processed outputs saved to {os.path.join(args.generation_file_dir, 'all_outputs.json')}")
