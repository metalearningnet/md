import sys
import torch
from pathlib import Path
from dataclasses import dataclass

_root_dir = Path(__file__).parent.parent
_conf_dir = _root_dir / 'conf'
sys.path.append(str(_conf_dir))

from settings import VERBOSE, MODEL, SKILL_MEMORY, LOADER, BATCH_SIZE, EPOCHS

USE_SDPA = False
MD_FILE = 'md.pt'
SAVE_DIR = Path('checkpoints')
MODEL_DIR = Path(__file__).parent.parent  / 'model'

def info(s):
    if VERBOSE:
        print(f"[INFO] {s}")

def get_device():
    if torch.cuda.is_available():
        # NVIDIA GPU is available
        device_type = "cuda"
        info("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU is available (MPS backend)
        device_type = "mps"
        info("Apple Silicon GPU is available. Using MPS.")
    else:
        # Fallback to CPU
        device_type = "cpu"
        info("No GPU available. Using CPU.")
    
    return device_type, torch.device(device_type)

device_type, device = get_device()

@dataclass
class Cfg:
    model: dict
    epochs: int
    loader: dict
    md_file: str
    use_sdpa: bool
    save_dir: Path
    batch_size: int
    model_dir: Path
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
    epochs=EPOCHS,
    md_file=MD_FILE,
    save_dir=SAVE_DIR,
    use_sdpa=USE_SDPA,
    model_dir=MODEL_DIR,
    batch_size=BATCH_SIZE,
    skill_memory=SKILL_MEMORY
)

def calculate_lm_loss(outputs, batch, loss_fn, device):
    # Get dimensions
    lm_logits = outputs['lm_logits']
    input_ids = batch['input_ids'].to(device)
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

def calculate_action_loss(outputs, batch, loss_fn, device):
    # Get dimensions
    action_logits = outputs['action_logits']  # [batch, seq_len, action_dim]
    input_ids = batch['input_ids'].to(device)
    _, action_seq_len = action_logits.size(0), action_logits.size(1)
    input_len = input_ids.size(1)
    
    # Calculate memory context length
    M = action_seq_len - input_len  # Memory tokens count
    
    # Slice action logits to match text sequence
    sliced_logits = action_logits[:, M:M+input_len-1, :]  # [batch, input_len-1, action_dim]
    
    # Get labels (shifted input_ids)
    action_labels = input_ids[:, 1:]  # [batch, input_len-1]
    
    # Create valid mask
    mask = (action_labels != -100).flatten()
    
    assert sliced_logits.size(1) == input_len-1, \
        f"Action logits seq {sliced_logits.size(1)} != labels seq {input_len-1}"
    assert mask.shape[0] == sliced_logits.size(0) * sliced_logits.size(1), \
        f"Mask {mask.shape} vs logits {sliced_logits.shape}"
    
    return loss_fn(
        sliced_logits.reshape(-1, sliced_logits.size(-1))[mask],
        action_labels.reshape(-1)[mask]
    )

def md_train(model, optimizer, loader, device, scaler, scheduler, clip_grad_norm=1.0):
    """Enhanced training epoch with mixed precision and gradient clipping"""
    model.train()
    metrics = {
        'total_loss': 0.0,
        'lm_loss': 0.0,
        'action_loss': 0.0,
        'num_batches': 0
    }
    
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    action_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    for batch in loader:
        optimizer.zero_grad()
        
        with torch.autocast(
            device_type=device_type, 
            dtype=torch.float16 if device_type == "cuda" else None, 
            enabled=(device_type == "cuda")
        ):
            # Forward pass
            outputs = model(
                states=batch['states'].to(device),
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            
            # Loss calculations
            lm_loss = calculate_lm_loss(outputs, batch, lm_loss_fn, device)
            action_loss = calculate_action_loss(outputs, batch, action_loss_fn, device)
            total_loss = lm_loss + action_loss
        
        # Backprop with scaling
        scaler.scale(total_loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update metrics
        metrics['total_loss'] += total_loss.item()
        metrics['lm_loss'] += lm_loss.item()
        metrics['action_loss'] += action_loss.item()
        metrics['num_batches'] += 1
    
    # Normalize metrics
    for k in ['total_loss', 'lm_loss', 'action_loss']:
        metrics[k] /= metrics['num_batches']
    
    return metrics

def md_validate(model, loader, device) -> dict:
    model.eval()
    metrics = {
        'total_loss': 0.0,
        'lm_loss': 0.0,
        'action_loss': 0.0,
        'action_acc': 0.0,
        'action_top3_acc': 0.0,
        'perplexity': 0.0,
        'num_samples': 0,
        'num_tokens': 0,
        'num_actions': 0
    }
    
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id, reduction='sum')
    action_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
        for batch in loader:
            # Device transfer
            states = batch['states'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

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
            
            # --- Action Loss Calculation ---
            action_logits = outputs['action_logits']
            action_len = action_logits.size(1)
            action_M = action_len - input_len  # Should match M
            
            # Slice action logits and labels
            sliced_action_logits = action_logits[:, action_M:action_M+input_len-1, :]
            action_labels = input_ids[:, 1:]  # Single shift
            
            # Action Mask
            action_mask = (action_labels != -100).flatten()
            assert sliced_action_logits.size(1) == action_labels.size(1), "Action dimension mismatch"

            # --- Loss Computations ---
            # LM Loss
            lm_loss = lm_loss_fn(
                sliced_lm_logits.reshape(-1, sliced_lm_logits.size(-1))[lm_mask],
                lm_labels.flatten()[lm_mask]
            ) if lm_mask.any() else 0.0

            # Action Loss
            action_loss = action_loss_fn(
                sliced_action_logits.reshape(-1, sliced_action_logits.size(-1))[action_mask],
                action_labels.flatten()[action_mask]
            ) if action_mask.any() else 0.0

            # --- Accuracy Calculations ---
            if action_mask.any():
                # Top-1
                action_preds = torch.argmax(sliced_action_logits, dim=-1)
                correct = (action_preds.flatten()[action_mask] == action_labels.flatten()[action_mask])
                metrics['action_acc'] += correct.sum().item()

                # Top-3
                top3 = torch.topk(sliced_action_logits, 3, dim=-1).indices
                matches = (top3 == action_labels.unsqueeze(-1)).any(-1)
                metrics['action_top3_acc'] += matches.flatten()[action_mask].sum().item()
            
            # --- Metric Aggregation ---
            metrics['total_loss'] += (lm_loss + action_loss).item()
            metrics['lm_loss'] += lm_loss.item()
            metrics['action_loss'] += action_loss.item()
            metrics['num_samples'] += input_ids.size(0)
            metrics['num_tokens'] += lm_mask.sum().item()
            metrics['num_actions'] += action_mask.sum().item()

    # --- Metric Normalization ---
    if metrics['num_samples'] > 0:
        metrics['total_loss'] /= metrics['num_samples']

        if metrics['num_tokens'] > 0:
            metrics['lm_loss'] /= metrics['num_tokens']
            metrics['perplexity'] = torch.exp(torch.tensor(metrics['lm_loss'])).item()

        if metrics['num_actions'] > 0:
            metrics['action_loss'] /= metrics['num_actions']
            metrics['action_acc'] /= metrics['num_actions']
            metrics['action_top3_acc'] /= metrics['num_actions']

    del metrics['num_samples'], metrics['num_tokens'], metrics['num_actions']
    return metrics
