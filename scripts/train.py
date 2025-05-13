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
from utils import MD_TAG, md_train, md_validate, cfg, add_dist_config, default_dataset

def train(config: dict):
    """
    config:
        - name (str): Dataset name.
        - dataset_config (str): Dataset configuration name.
        - split (str): Dataset split name (e.g., "train").
        - split_ratio (float): Proportion of the dataset to be allocated for training or testing.
        - seed (int): Random seed.
        - lr (float): Learning rate.
        - epochs (int): Number of epochs.
        - batch_size (int): Training batch size.
        - val_split (float): Proportion of the training set to be used for validation.
        - weight_decay (float): Weight decay.
        - ckpt_dir (str): Checkpoint directory.
        - save_interval (int): Checkpoint frequency.
        - fabric_config (dict): Configuration options for the Lightning Fabric setup.
        - batches (int): Number of batches.
        - gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating model weights.
        - dist (bool): Enable distributed training.
    """

    name = config['name']
    dataset_config = config.get('dataset_config')
    split = config.get('split', 'train')
    split_ratio = config.get('split_ratio', 0.0)
    seed = config.get('seed', 42)
    lr = config.get('lr', 1e-4)
    num_epochs = config.get('epochs', 1)
    batch_size = config.get('batch_size', 1)
    val_split=config.get('val_split', 0.1)
    weight_decay = config.get('weight_decay', 0.01)
    ckpt_dir = Path(config.get('ckpt_dir', cfg.ckpt_dir))
    save_interval = config.get('save_interval', 1)
    fabric_config = config['fabric_config']
    num_batches = config.get('batches', -1)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
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
        'dataset_name': name,
        'split': split,
        'split_ratio': split_ratio,
        'seed': seed
    }

    if dataset_config:
        loader_args['dataset_config'] = dataset_config
    
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
            torch.save(model.state_dict(), ckpt_dir / cfg.md_file)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Periodic checkpointing
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch+1}.pt")
            print(f"Saved epoch {epoch+1} checkpoint")

def main():
    parser = argparse.ArgumentParser(description="Train the MD Model")
    
    # Dataset configuration
    parser.add_argument("--name", type=str, 
                        help="Dataset name")
    parser.add_argument("--config", type=str, default=None,
                        help="Dataset configuration name")
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
    parser.add_argument("--ckpt_dir", type=str, default=cfg.ckpt_dir,
                        help="Checkpoint directory")
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

    if args.name is None:
        args.name = default_dataset
    
    config = {
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
        'ckpt_dir': args.ckpt_dir,
        'save_interval': args.save_interval,
        'fabric_config': {
            'accelerator': cfg.accelerator,
            'precision': cfg.precision
        }
    }

    if args.config:
        config['dataset_config'] = args.config

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
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()