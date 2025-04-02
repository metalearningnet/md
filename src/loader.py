import os
import torch
from utils import cfg
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Union, Tuple, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

class MDLoader(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        tokenizer_name: str = cfg.model_dir,
        max_length: int = cfg.loader_max_length,
        state_window: int = cfg.loader_state_window,
        split: Optional[str] = None,
        split_ratio: float = 0.0,
        seed: int = 42,
        dataset: Optional[Union[Dataset, DatasetDict]] = None
    ):
        """
        Memory-efficient data loader for MD model training
        
        Args:
            dataset_name: HF dataset identifier
            dataset_config: Optional dataset configuration name
            tokenizer_name: Pretrained tokenizer name/path
            max_length: Maximum sequence length for truncation
            state_window: Context window for state averaging
            split: Predefined dataset split
            split_ratio: Train/validation split ratio
            seed: Random seed for reproducibility
            dataset: Preloaded dataset to use
        """
        super().__init__()
        
        # Sequence configuration
        self.max_length = max_length
        self.state_window = state_window
        self.min_sequence_length = state_window + 1
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        # Embedding model setup with device awareness
        self._init_embedding_model(tokenizer_name)
        
        # Dataset loading and processing
        self._init_dataset(dataset_name, dataset_config, split, split_ratio, seed, dataset)
        
        # Model configuration and projection layer
        self._init_projection_layer()

    def _init_embedding_model(self, tokenizer_name: str):
        """Initialize embedding model with proper device placement"""
        try:
            model = AutoModel.from_pretrained(tokenizer_name)
            self.embed_model = model.get_input_embeddings()
            self.embed_model.eval()
            self.device = self.embed_model.weight.device
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")

    def _init_projection_layer(self):
        """Initialize dimension projection layer using model config"""
        try:
            lm_config = AutoModelForCausalLM.from_pretrained(cfg.model_dir).config
            self.embed_proj = nn.Linear(
                self.embed_model.embedding_dim,
                lm_config.hidden_size
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Projection layer initialization failed: {str(e)}")

    def _init_dataset(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        split: Optional[str],
        split_ratio: float,
        seed: int,
        dataset: Optional[Union[Dataset, DatasetDict]]
    ):
        """Load and process dataset with error handling"""
        if dataset is not None:
            self.dataset = dataset
        else:
            self._load_hf_dataset(dataset_name, dataset_config, split, split_ratio, seed)
        
        # Filter short sequences
        self.dataset = self.dataset.filter(
            lambda x: len(self.tokenizer.encode(x["text"])) >= self.min_sequence_length,
            num_proc=os.cpu_count()
        )

    def _load_hf_dataset(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        split: Optional[str],
        split_ratio: float,
        seed: int
    ):
        """Load dataset from HuggingFace Hub"""
        try:
            load_kwargs = {"name": dataset_config} if dataset_config else {}
            
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
        """Automatically select the most appropriate split"""
        if isinstance(dataset, DatasetDict):
            for split_name in ['train', 'validation', 'test',
                        'training', 'val', 'testing',
                        'train-validation']:
                if split_name in dataset:
                    return dataset[split_name]
            return next(iter(dataset.values()))
        return dataset

    def _create_validation_split(self, split_ratio: float, seed: int):
        """Properly store both splits"""
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
        """Process single example with enhanced error handling"""
        try:
            example = self.dataset[idx]
            text = example.get("text", example.get("content", ""))
            
            # Tokenization with truncation
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )[0]

            if len(tokens) < self.min_sequence_length:
                raise ValueError(f"Sequence too short: {len(tokens)} tokens")

            # Embedding generation
            embeddings = self._generate_embeddings(tokens)
            
            # State computation
            states = self._compute_states(embeddings, tokens)
            
            return {
                "raw_text": text,
                "states": states.cpu(),
                "input_ids": tokens[:-1],
                "labels": tokens[1:],  # This will be used for both LM and action loss
            }
        except Exception as e:
            raise RuntimeError(f"Failed processing example {idx}: {str(e)}")

    def _generate_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate projected embeddings with device management"""
        with torch.no_grad():
            tokens = tokens.to(self.device)
            raw_embeddings = self.embed_model(tokens)
            return self.embed_proj(raw_embeddings)

    def _compute_states(self, embeddings: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Compute state representations with vectorized operations"""
        seq_len = len(tokens)
        valid_window = min(self.state_window, seq_len - 1)
        
        if seq_len <= 1:
            return torch.zeros(0, embeddings.size(-1), device=self.device)

        # Main pooling operation
        states = torch.nn.functional.avg_pool1d(
            embeddings.unsqueeze(0).transpose(1, 2),
            kernel_size=valid_window,
            stride=1,
            padding=0
        ).squeeze(0).transpose(0, 1)

        # Handle padding for remaining positions
        if len(states) < seq_len - 1:
            padding = embeddings[len(states):-1].mean(dim=0, keepdim=True)
            states = torch.cat([states, padding])

        return states

    def _process_example(self, example: dict) -> dict:
        """
        Process a single example for testing/validation purposes
        Maintains original non-vectorized state computation for reference
        
        Args:
            example: Raw example dictionary from the dataset
            
        Returns:
            Dictionary containing processed example data
        """
        try:
            text = example.get("text", example.get("content", ""))
            
            # Tokenize with truncation
            tokens = self.tokenizer.encode(
                text, 
                max_length=self.max_length, 
                truncation=True,
                return_tensors="pt"
            )[0]

            if len(tokens) < self.min_sequence_length:
                # Pad with [PAD] tokens to reach minimum length
                pad_length = self.min_sequence_length - len(tokens)
                tokens = torch.cat([
                    tokens,
                    torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])

            # Generate embeddings with device awareness
            with torch.no_grad():
                tokens = tokens.to(self.device)
                raw_embeddings = self.embed_model(tokens)
                embeddings = self.embed_proj(raw_embeddings)

            # Compute states using original loop-based method
            if len(tokens) > 1:
                states = []
                for i in range(len(tokens) - 1):
                    start = max(0, i - self.state_window + 1)
                    window = embeddings[start:i+1]
                    states.append(window.mean(dim=0))
                states = torch.stack(states)
            else:
                states = torch.zeros(0, embeddings.size(-1), device=self.device)

            return {
                "raw_text": text,
                "states": states.cpu(),
                "input_ids": tokens[:-1].cpu(),
                "labels": tokens[1:].cpu(),
            }
            
        except Exception as e:
            raise RuntimeError(f"Example processing failed: {str(e)}")
    
    def collate_fn(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Corrected collate function with proper mask dimensions"""
        # Process text sequences
        input_ids = [item["input_ids"] for item in batch]
        padded_inputs = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        # Process state representations
        states = [item["states"] for item in batch]
        padded_states = pad_sequence(
            states,
            batch_first=True,
            padding_value=0.0
        )

        # Create masks with correct dimensions
        attention_mask = (padded_inputs != self.tokenizer.pad_token_id).long()  # [batch, seq_len]
        state_mask = (padded_states.abs().sum(dim=-1) != 0).long()  # [batch, seq_len]

        # Ensure labels have same seq_len as inputs
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100
        )
        
        return {
            "states": padded_states,
            "state_mask": state_mask,
            "input_ids": padded_inputs,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def get_dataloader(
        self, 
        batch_size: int = cfg.batch_size,
        shuffle: Optional[bool] = None,
        drop_last: bool = False
    ) -> DataLoader:
        """Create configured DataLoader with optimized settings"""
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle if shuffle is not None else True,
            pin_memory=torch.cuda.is_available(),
            num_workers=min(4, os.cpu_count() // 2),
            drop_last=drop_last,
            persistent_workers=True
        )

    def get_dataloaders(
        self, 
        batch_size: int = cfg.batch_size,
        val_batch_size: Optional[int] = None,
        val_split: Optional[float] = None,
        seed: int = 42
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create train/validation dataloaders with automatic split"""
        if not val_split or val_split <= 0:
            return self.get_dataloader(batch_size), None

        # Create split
        val_size = int(len(self) * val_split)
        train_set, val_set = random_split(
            self,
            [len(self) - val_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        return (
            DataLoader(
                train_set,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=min(4, os.cpu_count() // 2)
            ),
            DataLoader(
                val_set,
                batch_size=val_batch_size or batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                num_workers=min(2, os.cpu_count() // 4)
            )
        )