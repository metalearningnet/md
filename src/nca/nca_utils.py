# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def nca_collate(
    batch,
    tokenizer,
    max_length,
    max_prompt_length,
    label_pad_token_id,
    truncation_mode,
    is_encoder_decoder
) -> Dict[str, torch.Tensor]:
    collator = NCADataCollatorWithPadding(
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        label_pad_token_id=label_pad_token_id,
        padding_value=tokenizer.pad_token_id,
        truncation_mode=truncation_mode,
        is_encoder_decoder=is_encoder_decoder,
    )
    return collator(batch)

@dataclass
class NCADataCollatorWithPadding:
    r"""
    Contrast DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None
    optimize_prompt: Optional[bool] = False

    def tokenize_batch_element(
        self,
        item,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            A0_tokens = self.tokenizer(item["A0"], add_special_tokens=False)#{"input_ids": "attention_mask":}
            A1_tokens = self.tokenizer(item["A1"], add_special_tokens=False)#{"input_ids": "attention_mask":}
            A2_tokens = self.tokenizer(item["A2"], add_special_tokens=False)#{"input_ids": "attention_mask":}
            A3_tokens = self.tokenizer(item["A3"], add_special_tokens=False)#{"input_ids": "attention_mask":}
            prompt_tokens = self.tokenizer(item["prompt"], add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for A 0-3
            eos_indices_A0 = [i for i, x in enumerate(A0_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_A0 else p for i, p in enumerate(A0_tokens["attention_mask"])
            ]
            A0_tokens["attention_mask"] = new_attention_mask_c
            A0_tokens["input_ids"].append(self.tokenizer.eos_token_id) # prompt is not added with eos finish
            A0_tokens["attention_mask"].append(1)

            eos_indices_A1 = [i for i, x in enumerate(A1_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_A1 else p for i, p in enumerate(A1_tokens["attention_mask"])
            ]
            A1_tokens["attention_mask"] = new_attention_mask_c
            A1_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            A1_tokens["attention_mask"].append(1)
            
            eos_indices_A2 = [i for i, x in enumerate(A2_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_A2 else p for i, p in enumerate(A2_tokens["attention_mask"])
            ]
            A2_tokens["attention_mask"] = new_attention_mask_c
            A2_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            A2_tokens["attention_mask"].append(1)
            
            eos_indices_A3 = [i for i, x in enumerate(A3_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_A3 else p for i, p in enumerate(A3_tokens["attention_mask"])
            ]
            A3_tokens["attention_mask"] = new_attention_mask_c
            A3_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            A3_tokens["attention_mask"].append(1)

            longer_response_length = max(len(A0_tokens["input_ids"]), len(A1_tokens["input_ids"]), len(A2_tokens["input_ids"]), len(A3_tokens["input_ids"]))

            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:#
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                A0_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in A0_tokens.items()}
                A1_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in A1_tokens.items()}
                A2_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in A2_tokens.items()}
                A3_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in A3_tokens.items()}

            # Create labels
            A0_sequence_tokens = {k: prompt_tokens[k] + A0_tokens[k] for k in A0_tokens}
            A1_sequence_tokens = {k: prompt_tokens[k] + A1_tokens[k] for k in A1_tokens}
            A2_sequence_tokens = {k: prompt_tokens[k] + A2_tokens[k] for k in A2_tokens}
            A3_sequence_tokens = {k: prompt_tokens[k] + A3_tokens[k] for k in A3_tokens}        
            
            A0_sequence_tokens["labels"] = A0_sequence_tokens["input_ids"][:]
            A1_sequence_tokens["labels"] = A1_sequence_tokens["input_ids"][:]
            A2_sequence_tokens["labels"] = A2_sequence_tokens["input_ids"][:]
            A3_sequence_tokens["labels"] = A3_sequence_tokens["input_ids"][:]
 
            if not self.optimize_prompt:
                A0_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                        prompt_tokens["input_ids"]
                    )
                A1_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                        prompt_tokens["input_ids"]
                    )
                A2_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                        prompt_tokens["input_ids"]
                    )
                A3_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                        prompt_tokens["input_ids"]
                    )
            for k, toks in {
                "A0": A0_sequence_tokens,
                "A1": A1_sequence_tokens,
                "A2": A2_sequence_tokens,
                "A3": A3_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens

        else:
            raise NotImplementedError

        batch["prompt"] = item["prompt"]
        batch["A0"] = item["prompt"] + item["A0"]
        batch["A0_response_only"] = item["A0"]
        batch["A1"] = item["prompt"] + item["A1"]
        batch["A1_response_only"] = item["A1"]
        batch["A2"] = item["prompt"] + item["A2"]
        batch["A2_response_only"] = item["A2"]
        batch["A3"] = item["prompt"] + item["A3"]
        batch["A3_response_only"] = item["A3"]
        
        batch["A0_score"] = item["score_A0"]
        batch["A1_score"] = item["score_A1"]
        batch["A2_score"] = item["score_A2"]
        batch["A3_score"] = item["score_A3"]
        
        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    raise NotImplementedError
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_score"):
                padded_batch[k] = torch.FloatTensor([ex[k] for ex in batch])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            batch_element = self.tokenize_batch_element(feature)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)

def apply_chat_template(
    example, 
    tokenizer, 
    task: Literal["sft", "generation", "rm", "dpo", "reward"] = "reward", 
    assistant_prefix: str = "<|assistant|>\n"
):
    def _strip_prefix(s: str, pattern: str) -> str:
        """Strip the given pattern from the start of the string if present."""
        return re.sub(f"^{re.escape(pattern)}", "", s)

    def _validate_role_alternation(messages: list) -> None:
        """Validate that messages strictly alternate between user/assistant roles."""
        # Skip system message if present
        start_idx = 1 if messages and messages[0]["role"] == "system" else 0
        
        for i in range(start_idx, len(messages)):
            current_role = messages[i]["role"]
            if i == start_idx:
                # First message should be user (after system)
                if current_role != "user":
                    raise ValueError(
                        f"First message after system must be 'user', got '{current_role}' at position {i}. "
                        f"Full messages: {messages}"
                    )
            else:
                prev_role = messages[i-1]["role"]
                if current_role == prev_role:
                    raise ValueError(
                        f"Consecutive messages with same role '{current_role}' at positions {i-1} and {i}. "
                        f"Messages must alternate between user and assistant. Full messages: {messages}"
                    )
                if current_role not in ["user", "assistant"]:
                    raise ValueError(
                        f"Invalid role '{current_role}' at position {i}. "
                        f"Only 'user' and 'assistant' roles are allowed after system message."
                    )

    def _prepare_messages(messages: list) -> list:
        """Prepare messages by ensuring system message exists and validating roles."""
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Ensure system message exists
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        
        _validate_role_alternation(messages)
        return messages

    try:
        if task in ["sft", "generation"]:
            messages = _prepare_messages(example["messages"])
            example["text"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(task == "generation")
            )

        elif task == "rm":
            if not all(k in example for k in ("chosen", "rejected")):
                raise ValueError(
                    f"Require 'chosen' and 'rejected' keys for 'rm' task, found: {list(example.keys())}"
                )
            
            chosen_messages = _prepare_messages(example["chosen"])
            rejected_messages = _prepare_messages(example["rejected"])
            
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

        elif task == "dpo":
            if not all(k in example for k in ("chosen", "rejected")):
                raise ValueError(
                    f"Require 'chosen' and 'rejected' keys for 'dpo' task, found: {list(example.keys())}"
                )
            
            # Prepare prompt messages (user query + system message)
            prompt_messages = [msg for msg in example["chosen"] if msg["role"] == "user"][:1]
            if not prompt_messages:
                raise ValueError("No user message found in 'chosen' messages")
            
            if example["chosen"][0]["role"] == "system":
                prompt_messages.insert(0, example["chosen"][0])
            else:
                prompt_messages.insert(0, {"role": "system", "content": ""})
            
            # Prepare response messages
            chosen_messages = _prepare_messages(example["chosen"])
            rejected_messages = _prepare_messages(example["rejected"])
            
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Strip prefixes
            example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
            example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)

        elif task == "reward":
            if not all(k in example for k in ("A0", "A1", "A2", "A3")):
                raise ValueError(
                    f"Require 'A0'-'A3' keys for 'reward' task, found: {list(example.keys())}"
                )
            
            # Prepare prompt messages (first user query + system message)
            prompt_messages = [msg for msg in example["A0"] if msg["role"] == "user"][:1]
            if not prompt_messages:
                raise ValueError("No user message found in 'A0' messages")
            
            if example["A0"][0]["role"] == "system":
                prompt_messages.insert(0, example["A0"][0])
            else:
                prompt_messages.insert(0, {"role": "system", "content": ""})
            
            # Process each response
            for key in ["A0", "A1", "A2", "A3"]:
                messages = _prepare_messages(example[key])
                example[f"text_{key}"] = tokenizer.apply_chat_template(messages, tokenize=False)
                example[f"text_{key}"] = _strip_prefix(example[f"text_{key}"], assistant_prefix)
            
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

        else:
            raise ValueError(f"Unsupported task: {task}. Must be one of ['sft', 'generation', 'rm', 'dpo', 'reward']")

        return example

    except Exception as e:
        raise ValueError(f"Failed to process example for task '{task}': {str(e)}") from e

class DatasetMap(object):
    def __init__(self,
                 tokenizer,
                 truncation_mode,
                 max_prompt_length,
                 max_length,
                 max_target_length,
                 label_pad_token_id,
                 is_encoder_decoder=None):
        self.tokenizer = tokenizer
        self.truncation_mode = truncation_mode
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.label_pad_token_id = label_pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
    
    def generate(self, dataset):
        column_names = list(dataset.features)
        num_proc = os.cpu_count()
        dataset = dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer, "task": "reward"},
            num_proc=num_proc,
            remove_columns=[c for c in column_names if "score" not in c],
            desc="Formatting comparisons with prompt template",
            load_from_cache_file=False
        )
        return dataset.rename_columns(
            {"text_prompt": "prompt", "text_A0": "A0", "text_A1": "A1", "text_A2": "A2", "text_A3": "A3"}
        )
