import os
import torch
import numpy as np
from accelerate import PartialState
from datasets.features import Value
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Union, Tuple, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from utils import cfg, info, apply_chat_template, get_device
from torch.utils.data import DataLoader, Dataset, random_split

class MDLoader(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        tokenizer_name: str = cfg.model_dir,
        max_length: int = cfg.max_length,
        min_length: int = cfg.min_length,
        max_prompt_length: int = cfg.max_prompt_length,
        label_pad_token_id: int = -100,
        split: Optional[str] = None,
        split_ratio: float = 0.0,
        truncation_mode: str = cfg.truncation_mode,
        seed: int = 42,
        dataset: Optional[Union[Dataset, DatasetDict]] = None
    ):
        """       
        Args:
            dataset_name: Dataset name/path
            dataset_config: Optional dataset configuration name
            tokenizer_name: Pretrained tokenizer name/path
            max_length: Maximum sequence length for truncation
            split: Predefined dataset split
            split_ratio: Train/validation split ratio
            seed: Random seed for reproducibility
            dataset: Preloaded dataset to use
        """
        super().__init__()

        self.is_encoder_decoder = None
        self.truncation_mode = truncation_mode
        self.label_pad_token_id = label_pad_token_id
        
        self.min_length = min_length
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        self._init_dataset(dataset_name, dataset_config, split, split_ratio, seed, dataset)
    
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
    
    def tokenize_row(self, feature) -> Dict:
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
        return batch
    
    def _init_dataset(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        split: Optional[str],
        split_ratio: float,
        seed: int,
        dataset: Optional[Union[Dataset, DatasetDict]]
    ):
        if dataset is not None:
            self.dataset = dataset
        else:
            self._load_dataset(dataset_name, dataset_config, split, split_ratio, seed)
        num_proc = os.cpu_count()
        string_fields = [
            field for field, feature in self.dataset.features.items()
            if isinstance(feature, Value) and feature.dtype == 'string'
        ]
        info(f'string fields: {', '.join(string_fields)}')
        if 'text' not in string_fields:
            column_names = list(self.dataset.features)
            self.dataset = self.dataset.map(
                apply_chat_template,
                fn_kwargs={'tokenizer': self.tokenizer},
                num_proc=num_proc,
                remove_columns=column_names,
                desc="Formatting comparisons with prompt template",
            )
            generated_columns = set(self.dataset.features.keys())
            expected_columns = {'text_prompt', 'text_chosen', 'text_rejected'}
            if expected_columns.issubset(generated_columns):
                self.dataset = self.dataset.rename_columns({
                    'text_prompt': 'prompt',
                    'text_chosen': 'chosen',
                    'text_rejected': 'rejected'
                })
            with PartialState().local_main_process_first():
                self.dataset = self.dataset.map(self.tokenize_row, num_proc=num_proc)
            self.dataset = self.dataset.filter(
                lambda x: x is not None and all(
                    key in x and x[key] is not None
                    for key in ['prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids']
                ),
                num_proc=num_proc
            )
        else:
            self.dataset = self.dataset.filter(
                lambda x: all(
                    len(self.tokenizer.encode(x[field])) >= self.min_length
                    for field in string_fields
                ),
                num_proc=num_proc
            )

    def _load_dataset(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        split: Optional[str],
        split_ratio: float,
        seed: int
    ):
        try:
            load_kwargs = {'name': dataset_config} if dataset_config else {}
            
            if split:
                self.dataset = load_dataset(dataset_name, **load_kwargs, split=split)
            else:
                full_ds = load_dataset(dataset_name, **load_kwargs)
                self.dataset = self._auto_select_split(full_ds)

            if split_ratio > 0:
                self._create_validation_split(split_ratio, seed)
        except Exception as e:
            raise RuntimeError(f"Dataset loading failed: {str(e)}")

    def _auto_select_split(self, dataset: Union[Dataset, DatasetDict]) -> Dataset:
        if isinstance(dataset, DatasetDict):
            for split_name in ['train', 'validation', 'test']:
                if split_name in dataset:
                    return dataset[split_name]
            return next(iter(dataset.values()))
        return dataset

    def _create_validation_split(self, split_ratio: float, seed: int):
        split = self.dataset.train_test_split(
            test_size=split_ratio,
            seed=seed,
            shuffle=True
        )
        self.dataset = DatasetDict({
            'train': split['train'],
            'validation': split['test']
        })
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            example = self.dataset[idx]
            if 'text' in example:
                tokens = self.tokenizer.encode(
                    example['text'],
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )[0]

                if len(tokens) <= 1:
                    raise ValueError(f"Sequence too short (length {len(tokens)}) at index {idx}")
                
                return {
                    'raw_text': example['text'],
                    'input_ids': tokens[:-1],
                    'labels': tokens[1:],
                }
            else:
                return example
        except Exception as e:
            raise RuntimeError(f"Failed processing example {idx}: {str(e)}")
    
    def _collate(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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
            return [int(x) if x is not None else self.tokenizer.pad_token_id for x in seq]

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
        def pad_sequences(sequences, padding_value):
            return pad_sequence(
                sequences,
                batch_first=True,
                padding_value=padding_value
            )

        return {
            # Chosen sequences
            'chosen_input_ids': pad_sequences(chosen['input_ids'], self.tokenizer.pad_token_id),
            'chosen_attention_mask': pad_sequences(chosen['attention_mask'], 0),
            'chosen_labels': pad_sequences(chosen['labels'], self.label_pad_token_id),
            
            # Rejected sequences
            'rejected_input_ids': pad_sequences(rejected['input_ids'], self.tokenizer.pad_token_id),
            'rejected_attention_mask': pad_sequences(rejected['attention_mask'], 0),
            'rejected_labels': pad_sequences(rejected['labels'], self.label_pad_token_id),
            
            # Prompt sequences
            'prompt_input_ids': pad_sequences(prompt['input_ids'], self.tokenizer.pad_token_id),
            'prompt_attention_mask': pad_sequences(prompt['attention_mask'], 0),
        }
    
    def collate_fn(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if all('input_ids' in item for item in batch):
            # Process text sequences
            input_ids = [item['input_ids'] for item in batch]
            padded_inputs = pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )

            # Create masks with correct dimensions
            attention_mask = (padded_inputs != self.tokenizer.pad_token_id).long()  # [batch, seq_len]

            # Ensure labels have same seq_len as inputs
            labels = pad_sequence(
                [item['labels'] for item in batch],
                batch_first=True,
                padding_value=-100
            )
            
            return {
                'input_ids': padded_inputs,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
           return self._collate(batch)

    def get_dataloader(
        self, 
        batch_size: int,
        shuffle: Optional[bool] = None,
        drop_last: bool = False
    ) -> DataLoader:
        num_workers = 0 if get_device().type == 'mps' else os.cpu_count()
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle if shuffle is not None else True,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            drop_last=drop_last
        )

    def get_dataloaders(
        self,
        batch_size: int,
        val_batch_size: Optional[int] = None,
        val_split: Optional[float] = None,
        seed: int = 42
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        if not val_split or val_split <= 0:
            return self.get_dataloader(batch_size), None

        # Create split
        val_size = int(len(self) * val_split)
        train_set, val_set = random_split(
            self,
            [len(self) - val_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        num_workers = 0 if get_device().type == 'mps' else os.cpu_count()
        return (
            DataLoader(
                train_set,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers
            ),
            DataLoader(
                val_set,
                batch_size=val_batch_size or batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers
            )
        )