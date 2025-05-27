import os
import torch
from datasets.features import Value
from transformers import AutoTokenizer
from utils import cfg, info, get_device
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Union, Tuple, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset, random_split

class MDLoader(Dataset):
    def __init__(self,
                 path: str,
                 name: Optional[str] = None,
                 tokenizer_name: str = cfg.model_dir,
                 max_length: int = cfg.max_length,
                 min_length: int = cfg.min_length,
                 max_prompt_length: int = cfg.max_prompt_length,
                 max_target_length: int = cfg.max_target_length,
                 label_pad_token_id: int = cfg.label_pad_token_id,
                 split: Optional[str] = None,
                 split_ratio: float = 0.0,
                 truncation_mode: str = cfg.truncation_mode,
                 seed: int = 42,
                 dataset: Optional[Union[Dataset, DatasetDict]] = None):
        """       
        Args:
            path: Path of the dataset to load.
            name: Configuration name for the dataset, if applicable.
            tokenizer_name: Name or path of the pretrained tokenizer to use.
            max_length: Maximum total sequence length.
            min_length: Minimum sequence length to keep during filtering.
            max_prompt_length: Maximum allowed length for the prompt portion.
            max_target_length: Maximum allowed length for the target sequence.
                               Required when using encoder-decoder architectures.
            label_pad_token_id: Token ID used for padding labels.
                                Required when using the default data collator.
            split: Predefined dataset split to load (e.g., 'train', 'test').
            split_ratio: Ratio for train/validation split (0.0 means no split).
            truncation_mode: Mode for truncating prompts. Options:
                             - 'keep_end': Preserve the end of the prompt
                             - 'keep_start': Preserve the start of the prompt
            seed: Random seed for reproducibility of operations.
            dataset: Preloaded dataset to use instead of loading from dataset_name.
        """
        super().__init__()

        self.is_encoder_decoder = None
        self.truncation_mode = truncation_mode
        self.label_pad_token_id = label_pad_token_id
        
        self.min_length = min_length
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        self._init_dataset(path, name, split, split_ratio, seed, dataset)
    
    def _init_dataset(
        self,
        path: str,
        name: Optional[str],
        split: Optional[str],
        split_ratio: float,
        seed: int,
        dataset: Optional[Union[Dataset, DatasetDict]]
    ):
        if dataset is not None:
            self.dataset = dataset
        else:
            self._load_dataset(path, name, split, split_ratio, seed)
        fields = [
            field for field, feature in self.dataset.features.items()
            if isinstance(feature, Value) and feature.dtype == 'string'
        ]
        info(f'fields: {', '.join(fields)}')
        if fields and 'text' not in fields:
            from utils import DatasetMap
            dataset_map = DatasetMap(
                self.tokenizer,
                self.truncation_mode,
                self.max_prompt_length,
                self.max_length,
                self.max_target_length,
                self.label_pad_token_id
            )
            self.dataset = dataset_map.generate(self.dataset)
        else:
            num_proc = os.cpu_count()
            self.dataset = self.dataset.filter(
                lambda x: all(
                    len(self.tokenizer.encode(x[field])) >= self.min_length
                    for field in fields
                ),
                num_proc=num_proc
            )

    def _load_dataset(
        self,
        path: str,
        name: Optional[str],
        split: Optional[str],
        split_ratio: float,
        seed: int
    ):
        try:
            load_kwargs = {'name': name} if name else {}
            if split:
                from datasets import get_dataset_split_names
                split_names = get_dataset_split_names(path)
                if split not in split_names:
                    new_split = split + '_prefs'
                    if new_split not in split_names:
                        raise RuntimeError(f"Invalid split: {split}")
                    split = new_split
                self.dataset = load_dataset(path, **load_kwargs, split=split)
            else:
                full_ds = load_dataset(path, **load_kwargs)
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
           from utils import collate
           return collate(
               batch=batch,
               tokenizer=self.tokenizer,
               max_length=self.max_length,
               max_prompt_length=self.max_prompt_length,
               label_pad_token_id=self.label_pad_token_id,
               truncation_mode=self.truncation_mode,
               is_encoder_decoder=self.is_encoder_decoder
            )

    def get_workers(self):
        return 0 if get_device().type == 'mps' else os.cpu_count()

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

        # Create split
        val_size = int(len(self) * val_split)
        train_set, val_set = random_split(
            self,
            [len(self) - val_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        num_workers = self.get_workers()
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