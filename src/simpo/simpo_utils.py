import os
import torch
import numpy as np
import torch.nn as nn
from accelerate import PartialState
from typing import Dict, Optional, Union
from transformers import PreTrainedModel
from torch.nn.utils.rnn import pad_sequence
from alignment.data import maybe_insert_system_message, is_openai_format

def apply_chat_template(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True
):
    required_keys = {'chosen', 'rejected'}
    if not required_keys.issubset(example.keys()):
        raise ValueError(
            f"Could not format example as dialogue! Require either "
            f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
        )
    
    if not all(is_openai_format(msg) for msg in (example['chosen'], example['rejected'])):
        raise ValueError("Could not format example as dialogue! Require OpenAI format for all messages.")

    if 'prompt' in example and is_openai_format(example['prompt']):
        prompt_messages = example['prompt']
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
    else:
        prompt_messages = example['chosen'][:-1]
        chosen_messages = example['chosen'][-1:]
        rejected_messages = example['rejected'][-1:]

    if not (prompt_messages and chosen_messages and rejected_messages):
        raise ValueError("Prompt, chosen, and rejected must be non-empty.")
    
    if auto_insert_empty_system_msg:
        maybe_insert_system_message(prompt_messages, tokenizer)

    def apply_and_trim(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            return text[len(tokenizer.bos_token):]
        return text
    
    example['text_prompt'] = apply_and_trim(prompt_messages)
    example['text_chosen'] = apply_and_trim(chosen_messages)
    example['text_rejected'] = apply_and_trim(rejected_messages)
    
    return example

def simpo_collate(
    batch, 
    tokenizer, 
    max_length,
    max_prompt_length, 
    label_pad_token_id, 
    truncation_mode,
    is_encoder_decoder
) -> Dict[str, torch.Tensor]:
    if is_encoder_decoder:
        raise NotImplementedError
    
    if not batch:
        return {
            'chosen_input_ids': torch.tensor([[]], dtype=torch.long),
            'chosen_attention_mask': torch.tensor([[]], dtype=torch.long),
            'chosen_labels': torch.tensor([[]], dtype=torch.long),
            'rejected_input_ids': torch.tensor([[]], dtype=torch.long),
            'rejected_attention_mask': torch.tensor([[]], dtype=torch.long),
            'rejected_labels': torch.tensor([[]], dtype=torch.long),
            'prompt_input_ids': torch.tensor([[]], dtype=torch.long),
            'prompt_attention_mask': torch.tensor([[]], dtype=torch.long),
        }
    
    def clean_sequence(seq):
        """Remove None values from sequence and convert to integers"""
        return [int(x) if x is not None else tokenizer.pad_token_id for x in seq]

    # First filter out None items and invalid examples
    valid_items = []
    for item in batch:
        if item is None:
            continue
            
        required_fields = [
            'chosen_input_ids', 'rejected_input_ids',
            'chosen_attention_mask', 'rejected_attention_mask',
            'prompt_input_ids', 'prompt_attention_mask'
        ]
        
        # Check all required fields exist
        if not all(field in item for field in required_fields):
            continue
            
        # Clean sequences by replacing None with pad_token_id
        item['chosen_input_ids'] = clean_sequence(item['chosen_input_ids'])
        item['rejected_input_ids'] = clean_sequence(item['rejected_input_ids'])
        item['prompt_input_ids'] = clean_sequence(item['prompt_input_ids'])
        
        # Check sequences are not empty after cleaning
        if (len(item['chosen_input_ids']) > 0 and 
            len(item['rejected_input_ids']) > 0 and
            len(item['prompt_input_ids']) > 0):
            valid_items.append(item)

    if not valid_items:
        # Return empty tensors with proper shape
        return {
            'chosen_input_ids': torch.tensor([[]], dtype=torch.long),
            'chosen_attention_mask': torch.tensor([[]], dtype=torch.long),
            'chosen_labels': torch.tensor([[]], dtype=torch.long),
            'rejected_input_ids': torch.tensor([[]], dtype=torch.long),
            'rejected_attention_mask': torch.tensor([[]], dtype=torch.long),
            'rejected_labels': torch.tensor([[]], dtype=torch.long),
            'prompt_input_ids': torch.tensor([[]], dtype=torch.long),
            'prompt_attention_mask': torch.tensor([[]], dtype=torch.long),
        }

    # Process sequences
    def process_sequences(items, prefix=""):
        sequences = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for item in items:
            # Convert to tensors
            sequences['input_ids'].append(torch.tensor(item[f'{prefix}input_ids'], dtype=torch.long))
            sequences['attention_mask'].append(torch.tensor(item[f'{prefix}attention_mask'], dtype=torch.long))
            # Use input_ids as labels if specific labels not provided
            label_data = item.get(f'{prefix}labels', item[f'{prefix}input_ids'])
            sequences['labels'].append(torch.tensor(label_data, dtype=torch.long))
        
        return sequences

    # Process all sequence types
    chosen = process_sequences(valid_items, 'chosen_')
    rejected = process_sequences(valid_items, 'rejected_')
    prompt = {
        'input_ids': [torch.tensor(item['prompt_input_ids'], dtype=torch.long) for item in valid_items],
        'attention_mask': [torch.tensor(item['prompt_attention_mask'], dtype=torch.long) for item in valid_items]
    }

    # Pad sequences
    def pad_sequences(sequences, val):
        return pad_sequence(
            sequences,
            batch_first=True,
            padding_value=val
        )

    return {
        # Chosen sequences
        'chosen_input_ids': pad_sequences(chosen['input_ids'], tokenizer.pad_token_id),
        'chosen_attention_mask': pad_sequences(chosen['attention_mask'], 0),
        'chosen_labels': pad_sequences(chosen['labels'], label_pad_token_id),
        
        # Rejected sequences
        'rejected_input_ids': pad_sequences(rejected['input_ids'], tokenizer.pad_token_id),
        'rejected_attention_mask': pad_sequences(rejected['attention_mask'], 0),
        'rejected_labels': pad_sequences(rejected['labels'], label_pad_token_id),
        
        # Prompt sequences
        'prompt_input_ids': pad_sequences(prompt['input_ids'], tokenizer.pad_token_id),
        'prompt_attention_mask': pad_sequences(prompt['attention_mask'], 0),
    }

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
    
    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def generate(self, dataset):
        column_names = list(dataset.features)
        num_proc = os.cpu_count()
        dataset = dataset.map(
            apply_chat_template,
            fn_kwargs={'tokenizer': self.tokenizer},
            num_proc=num_proc,
            remove_columns=column_names,
            desc="Formatting comparisons with prompt template",
        )
        generated_columns = set(dataset.features.keys())
        expected_columns = {'text_prompt', 'text_chosen', 'text_rejected'}
        if expected_columns.issubset(generated_columns):
            dataset = dataset.rename_columns({
                'text_prompt': 'prompt',
                'text_chosen': 'chosen',
                'text_rejected': 'rejected'
            })
        with PartialState().local_main_process_first():
            dataset = dataset.map(self.tokenize_row, num_proc=num_proc)
        dataset = dataset.filter(
            lambda x: x is not None and all(
                key in x and x[key] is not None
                for key in ['prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids']
            ),
            num_proc=num_proc
        )
        return dataset
    
    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a SimPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt. Avoid adding if it's already there
            bos_token_id = self.tokenizer.bos_token_id
            if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer. Avoid adding if it's already there
            eos_token_id = self.tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )
        
        return batch