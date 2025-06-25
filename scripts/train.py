import sys
import torch
import argparse
import numpy as np
import lightning as L
from pathlib import Path

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD
from loader import MDLoader
from utils import MD_TAG, md_train, md_validate, cfg, add_dist_config, default_dataset_path, get_strategy, clear_directory

def train(config: dict):
    """
    config:
        - path: Dataset path.
        - name: Dataset name.
        - split: Dataset split name (e.g., "train").
        - split_ratio: Proportion of the dataset to be allocated for training or testing.
        - val_split: Proportion of the training set to be used for validation.
        - seed: Random seed.
        - epochs: Number of epochs.
        - samples: Number of samples.
        - batch_size: Training batch size.
        - gradient_accumulation_steps: Number of steps to accumulate gradients before updating model weights.
        - lr: Learning rate.
        - betas: Betas for AdamW optimizer
        - weight_decay: Weight decay.
        - ckpt_path: Checkpoint path.
        - save_interval: Checkpoint frequency.
        - dist: Whether to enable distributed training.
        - fabric_config: Configuration options for the Lightning Fabric setup.
        - restore: Whether to restore the model from the provided checkpoint.
        - log: Whether to enable logging.
        - log_dir: Directory where logs should be saved.
        - log_interval: Interval at which to update and write logs.
    """
    try:
        path = config['path']
        name = config.get('name')
        split = config.get('split', 'train')
        split_ratio = config.get('split_ratio', 0.0)
        val_split = config.get('val_split', 0.2)
        seed = config.get('seed', 42)
        num_epochs = config.get('epochs', 1)
        num_samples = config.get('samples', -1)
        batch_size = config.get('batch_size', 1)
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        lr = config.get('lr', 1e-4)
        betas = config.get('betas', (0.9, 0.98))
        weight_decay = config.get('weight_decay', 0.01)
        ckpt_path = Path(config.get('ckpt_path', ''))
        save_interval = config.get('save_interval', 1)
        dist = config.get('dist', False)
        fabric_config = config['fabric_config']
        restore = config.get('restore', False)
        has_log = config.get('log', False)
        log_dir = Path(config.get('log_dir', ''))
        log_interval = config.get('log_interval', 1)

        val_samples = max(int(num_samples * val_split), 1) if num_samples != -1 else -1
        
        if not dist:
            strategy = get_strategy()
            if strategy:
                fabric_config.update({'strategy': strategy})
        else:
            torch.backends.cudnn.enabled = False
        
        fabric = L.Fabric(**fabric_config)
        fabric.launch()
        if not restore:
            model = MD()
        else:
            if ckpt_path:
                model = MD.from_pretrained(checkpoint_path=ckpt_path)
            else:
                model = MD.from_pretrained()

        trainable_params = [p for p in model.get_trainable_parameters() if p.requires_grad]

        if not dist:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay
            )
            model, optimizer = fabric.setup(model, optimizer)
        else:
            model = fabric.setup(model)
            optimizer = None
        
        if model.has_anno:
            model.mark_forward_method('annotate')
        
        loader_args = {
            'path': path,
            'name': name,
            'split': split,
            'split_ratio': split_ratio,
            'seed': seed
        }
        
        loader = MDLoader(**loader_args)
        train_loader, val_loader = loader.get_dataloaders(
            batch_size=batch_size,
            val_split=val_split,
            seed=seed
        )
        train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=True)
        if val_loader:
            val_loader = fabric.setup_dataloaders(val_loader, use_distributed_sampler=True)
        
        best_val_loss = np.inf
        if ckpt_path:
            ckpt_dir = ckpt_path.parent
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        else:
            ckpt_dir = None
        
        if has_log:
            log_dir.mkdir(parents=True, exist_ok=True)
            clear_directory(log_dir)
        else:
            log_dir = None
            log_interval = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_metrics = md_train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                fabric=fabric,
                num_samples=num_samples,
                log_dir=log_dir,
                log_interval=log_interval,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            log_info = [
                f"Train Loss: {train_metrics['total_loss']:.4f}"
            ]
            
            val_metrics = md_validate(model, val_loader, fabric, num_samples=val_samples)

            if fabric.is_global_zero:
                log_info.append(f"Val Loss: {val_metrics.get('total_loss', 'N/A')}")
            
            print(" | ".join(log_info))
            
            if val_metrics.get('total_loss', np.inf) < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                if ckpt_path:
                    torch.save(model.state_dict(), ckpt_path)
                print(f"Saved best model with val loss: {best_val_loss:.4f}")
            
            if (epoch + 1) % save_interval == 0:
                if ckpt_dir:
                    torch.save(model.state_dict(), ckpt_dir / f"{MD_TAG}_epoch_{epoch+1}.pt")
                    print(f"Saved epoch {epoch+1} checkpoint")
    
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train the MD Model")
    
    # Dataset configuration
    parser.add_argument("--path", type=str, default=default_dataset_path,
                        help="Dataset path")
    parser.add_argument("--name", type=str, default=None,
                        help="Dataset name")
    parser.add_argument("--split", type=str, default="train",
                        help="Predefined dataset split")
    parser.add_argument("--split_ratio", type=float, default=0.0,
                        help="Train/test split ratio")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--samples", type=int, default=-1,
                        help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients before updating model weights")

    # System configuration
    parser.add_argument("--ckpt_path", type=str, default=cfg.ckpt_path,
                        help="Checkpoint path")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Checkpoint saving frequency")
    parser.add_argument("--restore", action="store_true", default=False,
                        help="Restore the model from the provided checkpoint")
    parser.add_argument("--log", action="store_true", default=cfg.log,
                        help="Whether to enable logging")
    parser.add_argument("--log_dir", type=str, default=cfg.train_log,
                        help="Directory where logs should be saved")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Interval at which to update and write logs")
    
    # Distributed training configuration
    parser.add_argument("--dist", action="store_true", default=False,
                        help="Enable distributed training")
    parser.add_argument("--addr", type=str, default=None,
                        help="Master address for distributed training")
    parser.add_argument("--port", type=int, default=None,
                        help="Master port for distributed training")
    parser.add_argument("--nodes", type=int, default=None,
                        help="The number of nodes for distributed training")

    args = parser.parse_args()
    
    config = {
        'path': args.path,
        'name': args.name,
        'split': args.split,
        'split_ratio': args.split_ratio,
        'val_split': args.val_split,
        'seed': args.seed,
        
        'epochs': args.epochs,
        'samples': args.samples,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,

        'lr': cfg.lr,
        'betas': cfg.betas,
        'weight_decay': cfg.weight_decay,

        'ckpt_path': args.ckpt_path,
        'save_interval': args.save_interval,
        'log': args.log,
        'log_dir': args.log_dir,
        'log_interval': args.log_interval,

        'dist': args.dist,
        'fabric_config': {
            'accelerator': 'auto',
            'precision': cfg.precision
        }
    }

    if args.dist:
        add_dist_config(
            config,
            main_addr=args.addr,
            main_port=args.port,
            num_nodes=args.nodes,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            lr=cfg.lr
        )

    train(config)

if __name__ == '__main__':
    main()
