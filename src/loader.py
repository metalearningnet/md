import os
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from typing import Optional, Union, Tuple, Dict, Any
from torch.utils.data import DataLoader, Dataset, random_split
from utils import DatasetMap, get_device, collate, info, warn, cfg

LOADER_CPU_MAX = 8

class MDLoader(Dataset):
    def __init__(self,
                 path: str,
                 name: Optional[str] = None,
                 split: str = 'train',
                 tokenizer_name: str = cfg.lm_dir,
                 max_length: int = cfg.max_length,
                 min_length: int = cfg.min_length,
                 max_prompt_length: int = cfg.max_prompt_length,
                 max_target_length: int = cfg.max_target_length,
                 label_pad_token_id: int = -100,
                 is_encoder_decoder: bool = False,
                 truncation_mode: str = cfg.truncation_mode,
                 dataset: Optional[Union[Dataset, DatasetDict]] = None):
        """       
        Args:
            path: Path of the dataset to load.
            name: Configuration name for the dataset.
            split: Predefined dataset split to load (e.g., 'train', 'test').
            tokenizer_name: Name or path of the pretrained tokenizer to use.
            max_length: Maximum total sequence length.
            min_length: Minimum sequence length to keep during filtering.
            max_prompt_length: Maximum allowed length for the prompt portion.
            max_target_length: Maximum allowed length for the target sequence.
            label_pad_token_id: Token ID used for padding labels.
            is_encoder_decoder: Whether the model uses separate encoder-decoder architecture
            truncation_mode: Mode for truncating prompts.
            dataset: Preloaded dataset to use instead of loading from dataset_name.
        """
        super().__init__()

        self.truncation_mode = truncation_mode
        self.label_pad_token_id = label_pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
        
        self.min_length = min_length
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            warn('add pad token')
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        self._init_dataset(path, name, split, dataset)
    
    def _init_dataset(
        self,
        path: str,
        name: Optional[str],
        split: Optional[str],
        dataset: Optional[Union[Dataset, DatasetDict]]
    ):
        if dataset is not None:
            self.dataset = dataset
        else:
            self._load_dataset(path, name, split)
        
        dataset_map = DatasetMap(
            self.tokenizer,
            self.truncation_mode,
            self.max_prompt_length,
            self.max_length,
            self.max_target_length,
            self.label_pad_token_id
        )
        self.dataset = dataset_map.generate(self.dataset)

    def _load_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
        fallback_split: str = 'train'
    ):
        try:
            if Path(path).exists():
                if Path(path).is_dir():
                    from datasets import load_from_disk
                    dataset = load_from_disk(path)
                    if split not in dataset:
                        warn(f"Split '{split}' not available. Using '{fallback_split}' instead.")
                        split = fallback_split
                    self.dataset = dataset[split]
                    info(f"Loaded local dataset (path: {path}, split: {split})")
                else:
                    raise RuntimeError(f"Local dataset path is not valid: {path} (not a directory)")
            else:
                from datasets import get_dataset_split_names, load_dataset
                split_names = get_dataset_split_names(path, config_name=name)
                
                if not split_names:
                    raise RuntimeError(f"No splits found for dataset: {path}")
                
                if split is None or split not in split_names:
                    original_split = split
                    split = fallback_split if fallback_split in split_names else split_names[0]
                    if original_split is not None:
                        warn(f"Split '{original_split}' not available. Using '{split}' instead.")
                
                self.dataset = load_dataset(path, name, split=split)
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {path} (config: {name}, split: {split}). "
                f"Original error: {str(e)}"
            ) from e
    
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

    def get_cpu_count(self):
        return min(LOADER_CPU_MAX, os.cpu_count())
    
    def get_workers(self):
        return 0 if get_device().type == 'mps' else self.get_cpu_count()

    def collate_fn(self, batch):
        return collate(
            batch=batch,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            label_pad_token_id=self.label_pad_token_id,
            truncation_mode=self.truncation_mode,
            is_encoder_decoder=self.is_encoder_decoder
        )
    
    def get_dataloader(
        self, 
        batch_size: int,
        shuffle: Optional[bool] = None,
        drop_last: bool = False
    ) -> DataLoader:
        num_workers = self.get_workers()
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

        val_size = int(len(self) * val_split)
        train_set, val_set = random_split(
            self,
            [len(self) - val_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        num_workers = self.get_workers()
        info(f'DataLooader (batch_size: {batch_size}, seed: {seed}, train_size: {len(self) - val_size}, val_size: {val_size})')
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
