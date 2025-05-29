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
from utils import md_train, md_validate, cfg, add_dist_config, default_dataset_path

def train(config: dict):
    """
    config:
        - path (str): Dataset path.
        - name (str): Dataset name.
        - split (str): Dataset split name (e.g., "train").
        - split_ratio (float): Proportion of the dataset to be allocated for training or testing.
        - epochs (int): Number of epochs.
        - batches (int): Number of batches.
        - batch_size (int): Training batch size.
        - seed (int): Random seed.
        - lr (float): Learning rate.
        - val_split (float): Proportion of the training set to be used for validation.
        - weight_decay (float): Weight decay.
        - gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating model weights.
        - ckpt (str): Checkpoint path.
        - save_interval (int): Checkpoint frequency.
        - fabric_config (dict): Configuration options for the Lightning Fabric setup.
        - dist (bool): Enable distributed training.
    """
    try:
        path = config['path']
        name = config.get('name')
        split = config.get('split', 'train')
        split_ratio = config.get('split_ratio', 0.0)
        num_epochs = config.get('epochs', 1)
        num_batches = config.get('batches', -1)
        batch_size = config.get('batch_size', 1)
        seed = config.get('seed', 42)
        lr = config.get('lr', 1e-4)
        val_split=config.get('val_split', 0.1)
        weight_decay = config.get('weight_decay', 0.01)
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        ckpt_path = Path(config.get('ckpt', cfg.ckpt_path))
        save_interval = config.get('save_interval', 1)
        fabric_config = config['fabric_config']
        dist = config.get('dist', False)
        
        fabric = L.Fabric(**fabric_config)
        fabric.launch()
        model = MD()

        # Configure optimizer
        trainable_params = []
        for param_group in model.get_trainable_parameters().values():
            trainable_params += [p for p in param_group if p.requires_grad]

        if not dist:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
            )
            model, optimizer = fabric.setup(model, optimizer)
        else:
            model = fabric.setup(model)
            optimizer = None

        # Configure dataset loader
        loader_args = {
            'path': path,
            'name': name,
            'split': split,
            'split_ratio': split_ratio,
            'seed': seed
        }
        
        loader = MDLoader(**loader_args)
        
        # Create dataloaders
        train_loader, val_loader = loader.get_dataloaders(
            batch_size=batch_size,
            val_split=val_split,
            seed=seed
        )
        train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=True)
        if val_loader:
            val_loader = fabric.setup_dataloaders(val_loader, use_distributed_sampler=True)
        
        # Training loop
        best_val_loss = np.inf
        ckpt_dir = ckpt_path.parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if cfg.log:
            cfg.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = cfg.train_log
            log_interval = cfg.log_interval
        else:
            log_path = None
            log_interval = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_metrics = md_train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                fabric=fabric,
                num_epochs=num_epochs,
                num_batches=num_batches,
                log_path=log_path,
                log_interval=log_interval,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            log_info = [
                f"Train Loss: {train_metrics['total_loss']:.4f}"
            ]
            
            val_metrics = md_validate(model, val_loader, fabric, num_batches=num_batches)
            if fabric.is_global_zero:
                log_info.append(f"Val Loss: {val_metrics.get('total_loss', 'N/A')}")
            
            print(" | ".join(log_info))
            
            # Save best checkpoint
            if val_metrics.get('total_loss', np.inf) < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved best model with val loss: {best_val_loss:.4f}")
            
            # Periodic checkpointing
            if (epoch + 1) % save_interval == 0:
                torch.save(model.state_dict(), ckpt_dir / f"train_epoch_{epoch+1}.pt")
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Training configuration
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")

    # System configuration
    parser.add_argument("--ckpt", type=str, default=cfg.ckpt_path,
                        help="Checkpoint path")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Checkpoint saving frequency")

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

        # Dataset parameters
        'seed': args.seed,
        'split': args.split,
        'split_ratio': args.split_ratio,
        
        # Training parameters
        'lr': args.lr,
        'val_split': args.val_split,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'batch_size': args.batch_size,

        # System parameters
        'dist': args.dist,
        'ckpt': args.ckpt,
        'save_interval': args.save_interval,
        'fabric_config': {
            'accelerator': cfg.accelerator,
            'precision': cfg.precision
        }
    }

    if args.dist:
        add_dist_config(
            config,
            main_addr=args.addr,
            main_port=args.port,
            num_nodes=args.nodes,
            weight_decay=args.weight_decay,
            lr=args.lr
        )
    
    train(config)

if __name__ == '__main__':
    main()