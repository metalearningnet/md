import sys
import torch
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

_root_dir = Path(__file__).parent.parent
_conf_dir = _root_dir / 'conf'
sys.path.append(str(_conf_dir))

import settings
from settings import MODEL, SKILL_MEMORY, LOADER

WARN = getattr(settings, 'WARN', True)
VERBOSE = getattr(settings, 'VERBOSE', False)

USE_SDPA = False

MD_FILE = 'md.pt'
SAVE_DIR = Path('checkpoints')
MODEL_DIR = Path(__file__).parent.parent  / 'model'

ACCELERATOR = "auto"
PRECISION = "16-mixed"

# Starting position for action embeddings
ACTION_START = 1 << 32

def info(s):
    if VERBOSE:
        print(f"[INFO] {s}")

def warn(s):
    if WARN:
        print(f"[WARN] {s}")

@dataclass
class Cfg:
    model: dict
    loader: dict
    md_file: str
    use_sdpa: bool
    save_dir: Path
    model_dir: Path
    precision: str
    accelerator: str
    action_start: int
    skill_memory: dict
    
    
    @property
    def model_name(self):
        return self.model['name']
    
    @property
    def loader_max_length(self):
        return self.loader['max_length']
    
    @property
    def loader_state_window(self):
        return self.loader['state_window']

cfg = Cfg(
    model=MODEL,
    loader=LOADER,
    md_file=MD_FILE,
    save_dir=SAVE_DIR,
    use_sdpa=USE_SDPA,
    model_dir=MODEL_DIR,
    precision=PRECISION,
    accelerator=ACCELERATOR,
    action_start=ACTION_START,
    skill_memory=SKILL_MEMORY
)

def calculate_lm_loss(outputs, batch, loss_fn):
    # Get dimensions
    lm_logits = outputs['lm_logits']
    input_ids = batch['input_ids']
    _, logits_seq_len = lm_logits.size(0), lm_logits.size(1)
    input_len = input_ids.size(1)
    
    # Calculate memory context length
    M = logits_seq_len - input_len  # Memory tokens count
    
    # Slice logits to match text sequence
    logits = lm_logits[:, M:M+input_len-1, :]  # (batch, input_len-1, vocab)
    
    # Get labels (shifted input_ids)
    labels = input_ids[:, 1:]  # (batch, input_len-1)
    
    # Create valid mask
    mask = (batch['attention_mask'][:, 1:] != 0).flatten()
    
    assert logits.size(1) == input_len-1, \
        f"Logits seq {logits.size(1)} != labels seq {input_len-1}"
    assert mask.shape[0] == logits.size(0)*logits.size(1), \
        f"Mask {mask.shape} vs logits {logits.shape}"
    
    return loss_fn(
        logits.reshape(-1, logits.size(-1))[mask],
        labels.reshape(-1)[mask]
    )

def md_train(model, optimizer, loader, scheduler, fabric):
    model.train()
    metrics = {
        'total_loss': 0.0,
        'num_batches': 0
    }
    
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    pbar = tqdm(
        loader,
        desc="Training",
        leave=False,
        disable=not fabric.is_global_zero,
        dynamic_ncols=True
    )

    for batch in pbar:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            states=batch['states'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Loss calculations
        loss = calculate_lm_loss(outputs, batch, lm_loss_fn)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            warn("NaN/Inf loss detected, skipping batch")
            continue

        has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
        if has_nan or has_inf:
            warn("NaN/Inf gradients detected, skipping update")
            optimizer.zero_grad()
            continue
        
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0, error_if_nonfinite=False)
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        metrics['total_loss'] += loss.item()
        metrics['num_batches'] += 1
    
    # Normalize metrics
    for k in ['total_loss']:
        metrics[k] /= metrics['num_batches']
    
    return metrics

def md_validate(model, loader, fabric) -> dict:
    model.eval()
    metrics = {
        'total_loss': 0.0,
        'avg_loss': 0.0,
        'num_samples': 0,
        'num_tokens': 0
    }
    
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id, reduction='sum')

    with torch.no_grad():
        pbar = tqdm(
            loader, 
            desc="Validating", 
            disable=not fabric.is_global_zero,
            dynamic_ncols=True
        )
        for batch in pbar:
            # Device transfer
            states = batch['states']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Forward pass
            outputs = model(states=states, input_ids=input_ids, attention_mask=attention_mask)
            
            # --- LM Loss Calculation ---
            lm_logits = outputs['lm_logits']
            input_len = input_ids.size(1)
            M = lm_logits.size(1) - input_len  # Memory context length
            
            # Slice logits and labels
            sliced_lm_logits = lm_logits[:, M:M+input_len-1, :]
            lm_labels = input_ids[:, 1:]
            
            # LM Mask
            lm_mask = (attention_mask[:, 1:] != 0).flatten()
            assert sliced_lm_logits.size(1) == lm_labels.size(1), "LM dimension mismatch"

            # --- Loss Computations ---
            lm_loss = lm_loss_fn(
                sliced_lm_logits.reshape(-1, sliced_lm_logits.size(-1))[lm_mask],
                lm_labels.flatten()[lm_mask]
            ) if lm_mask.any() else 0.0
            
            # --- Metric Aggregation ---
            metrics['total_loss'] += lm_loss.item()
            metrics['num_samples'] += input_ids.size(0)
            metrics['num_tokens'] += lm_mask.sum().item()

    gathered_loss = fabric.all_gather(torch.tensor(metrics['total_loss'])).sum()
    gathered_samples = fabric.all_gather(torch.tensor(metrics['num_samples'])).sum()
    if fabric.is_global_zero and gathered_samples > 0:
        avg_loss = gathered_loss / gathered_samples
        metrics['avg_loss'] = avg_loss
        print(f"Average Loss: {avg_loss:.4f} | Samples: {gathered_samples}")

    # --- Metric Normalization ---
    if metrics['num_samples'] > 0:
        metrics['total_loss'] /= metrics['num_samples']

    return metrics
