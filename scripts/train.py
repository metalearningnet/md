import sys
import torch
import argparse
import numpy as np
import lightning as L
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD
from loader import MDLoader
from utils import md_train, md_validate, cfg, add_dist_config

def train(config: dict):
    """
    config:
        - name: Dataset name (str)
        - dataset_config: Dataset configuration name (str)
        - split: Dataset split name (e.g., "train") (str)
        - split_ratio: Proportion of the dataset to be allocated for training or testing (float)
        - seed: Random seed (int)
        - lr: Learning rate (float)
        - epochs: Number of epochs (int)
        - batch_size: Training batch size (int)
        - val_split: Proportion of the training set to be used for validation (float)
        - weight_decay: Weight decay (float)
        - save_dir: Checkpoint directory (str)
        - save_interval: Checkpoint frequency (int)
        - fabric_config: Configuration options for the Lightning Fabric setup (dict)
    """
    name = config['name']
    fabric = L.Fabric(**config['fabric_config'])
    fabric.launch()

    # Initialize model
    model = MD.from_pretrained()

    # Configure optimizer
    trainable_params = []
    for param_group in model.get_trainable_parameters().values():
        trainable_params += [p for p in param_group if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get('lr'),
        weight_decay=config.get('weight_decay')
    )

    model, optimizer = fabric.setup(model, optimizer)
    
    # Configure dataset loader
    loader_args = {
        'dataset_name': name,
        'split': config.get('split'),
        'split_ratio': config.get('split_ratio'),
        'seed': config.get('seed')
    }

    if config.get('dataset_config'):
        loader_args['dataset_config'] = config['dataset_config']
    
    loader = MDLoader(**loader_args)
    
    # Create dataloaders
    train_loader, val_loader = loader.get_dataloaders(
        batch_size=config.get('batch_size'),
        val_split=config.get('val_split'),
        seed=config.get('seed')
    )
    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=True)
    if val_loader:
        val_loader = fabric.setup_dataloaders(val_loader, use_distributed_sampler=True)
   
    # Training utilities
    if hasattr(optimizer, 'optimizer'):  # For DeepSpeed wrapped optimizers
        scheduler_optimizer = optimizer.optimizer
    else:
        scheduler_optimizer = optimizer
    nr_epochs = config.get('epochs')
    scheduler = CosineAnnealingLR(scheduler_optimizer, T_max=nr_epochs)
    
    # Training loop
    best_val_loss = np.inf
    save_dir = Path(config.get('save_dir', cfg.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(nr_epochs):
        print(f"\nEpoch {epoch+1}/{nr_epochs}")
        train_metrics = md_train(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            scheduler=scheduler,
            fabric=fabric
        )
        
        log_info = [
            f"Train Loss: {train_metrics['total_loss']:.4f}",
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        ]
        
        val_metrics = md_validate(model, val_loader, fabric)
        if fabric.is_global_zero:
            log_info.append(f"Val Loss: {val_metrics.get('total_loss', 'N/A')}")
        
        print(" | ".join(log_info))
        
        # Save best checkpoint
        if val_metrics.get('total_loss', np.inf) < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'metrics': val_metrics,
                'config': config
            }, save_dir / cfg.md_file)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Periodic checkpointing
        if (epoch + 1) % config.get('save_interval', 1) == 0:
            torch.save(
                model.state_dict(),
                save_dir / f"epoch_{epoch+1}.pt"
            )
            print(f"Saved epoch {epoch+1} checkpoint")

def main():
    parser = argparse.ArgumentParser(description="Train the MD Model")
    
    # Dataset configuration
    parser.add_argument("--name", type=str, default="princeton-nlp/gemma2-ultrafeedback-armorm", 
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
    parser.add_argument("--save_dir", type=str, default=None,
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
        'save_interval': args.save_interval,
        'fabric_config': {
            'accelerator': cfg.accelerator,
            'precision': cfg.precision
        }
    }
    
    if args.save_dir:
        config['save_dir'] = args.save_dir

    if args.config:
        config['dataset_config'] = args.config

    if args.dist:
        add_dist_config(
            config['fabric_config'],
            main_addr=args.addr,
            main_port=args.port,
            num_nodes=args.nodes,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    train(config)

if __name__ == '__main__':
    main()