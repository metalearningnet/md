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
from utils import md_train, md_validate, cfg

def train(name: str, config: dict) -> None:
    """
    Main training function with unified configuration
    
    Args:
        name: Hugging Face dataset name
        config: Configuration dictionary containing:
            - dataset_config: HF dataset configuration name (str)
            - split: Dataset split name (str)
            - split_ratio: Train/val split ratio (float)
            - seed: Random seed (int)
            - lr: Learning rate (float)
            - epochs: Number of epochs (int)
            - batch_size: Training batch size (int)
            - val_split: Validation split ratio (float)
            - weight_decay: Weight decay (float)
            - save_dir: Checkpoint directory (str)
            - save_interval: Checkpoint frequency (int)
    """
    
    fabric = L.Fabric(accelerator=cfg.accelerator, precision=cfg.precision)
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
    val_loader = fabric.setup_dataloaders(val_loader, use_distributed_sampler=True)
   
    # Training utilities
    nr_epochs = config.get('epochs')
    scheduler = CosineAnnealingLR(optimizer, T_max=nr_epochs)
    
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
        
        val_metrics = md_validate(model, val_loader, fabric) if val_loader else {}
        if fabric.is_global_zero:
            log_info.append(f"Val Loss: {val_metrics.get('total_loss', 'N/A')}")
        
        print(" | ".join(log_info))
        
        # Save best checkpoint
        if val_loader and val_metrics.get('total_loss', np.inf) < best_val_loss:
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
    parser = argparse.ArgumentParser(description="Train MD Model")
    
    # Dataset configuration
    parser.add_argument("--name", required=True, 
                      help="HuggingFace dataset name")
    parser.add_argument("--config", default=None,
                      help="HF dataset configuration name")
    parser.add_argument("--split", type=str, default=None,
                      help="Predefined dataset split")
    parser.add_argument("--split_ratio", type=float, default=0.0,
                      help="Train/val split ratio for datasets without predefined splits")
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

    args = parser.parse_args()
    
    # Create comprehensive config dictionary
    config = {
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
        'save_interval': args.save_interval
    }
    
    if args.save_dir:
        config['save_dir'] = args.save_dir

    if args.config:
        config['dataset_config'] = args.config
    
    train(name=args.name, config=config)

if __name__ == "__main__":
    main()