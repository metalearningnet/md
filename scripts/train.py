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
from utils import (
    MD_TAG, LABEL_PAD_TOKEN_ID, md_train, md_validate,
    get_strategy, get_trainer, get_fabric_config, get_num_devices,
    default_dataset_path, clear_directory, info, set_dist_config, cfg
)

def train(config: dict):
    """
    config:
        - path: Dataset path.
        - name: Config name.
        - split: Dataset split name (e.g., 'train').
        - val_split: Proportion of the training set to be used for validation.
        - seed: Random seed.
        - epochs: Number of epochs.
        - batch_size: Training batch size.
        - samples: Number of samples.
        - lr: Learning rate.
        - eps: Numerical stability term for AdamW optimizer.
        - betas: Betas for AdamW optimizer.
        - weight_decay: Weight decay.
        - precision: Numerical Precision.
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
        dataset_path = config['path']
        config_name = config.get('name')
        split = config.get('split', 'train')
        val_split = config.get('val_split', 0.1)
        seed = config.get('seed', 42)
        num_epochs = config.get('epochs', 1)
        batch_size = config.get('batch_size', 1)
        num_samples = config.get('samples', -1)
        lr = config.get('lr', 3e-5)
        eps = config.get('eps', 1e-6)
        betas = config.get('betas', (0.9, 0.95))
        weight_decay = config.get('weight_decay', 0.1)
        precision = config.get('precision', 'bf16-mixed')
        ckpt_path = Path(config.get('ckpt_path', ''))
        save_interval = config.get('save_interval', 1)
        dist = config.get('dist', False)
        fabric_config = config['fabric_config']
        restore = config.get('restore', False)
        has_log = config.get('log', False)
        log_dir = Path(config.get('log_dir', ''))
        log_interval = config.get('log_interval', 1)
        
        val_samples = max(int(num_samples * val_split), 1) if num_samples != -1 else -1
        
        if 'strategy' not in fabric_config:
            strategy = get_strategy(precision)
            if strategy:
                fabric_config.update({'strategy': strategy})
        
        if get_num_devices() > 1:
            dist = True
        
        fabric = L.Fabric(**fabric_config)
        fabric.launch()
        if not restore:
            model = MD(dist=dist)
        else:
            info(f"Restore the model...")
            if ckpt_path:
                model = MD.from_pretrained(checkpoint_path=ckpt_path, dist=dist)
            else:
                model = MD.from_pretrained(dist=dist)

        trainable_params = [p for p in model.get_trainable_parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            eps=eps,
            betas=betas,
            weight_decay=weight_decay
        )
        model, optimizer = fabric.setup(model, optimizer)
        
        if model.has_anno:
            model.mark_forward_method('annotate')
        
        if cfg.sft:
            info(f"Training process (dataset: {dataset_path}, sft: enabled)")
        else:
            info(f"Training process (dataset: {dataset_path})")
        
        loader = MDLoader(
            path=dataset_path,
            name=config_name,
            split=split,
            is_encoder_decoder=model.config.is_encoder_decoder,
            label_pad_token_id=LABEL_PAD_TOKEN_ID
        )
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

        if num_samples > 0:
            steps = min(num_samples, len(train_loader))
        else:
            steps = len(train_loader)
        total_steps = num_epochs * steps
        model.set_max_steps(total_steps)
        trainer = get_trainer(model)
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
                trainer=trainer
            )
            
            log_info = [
                f"Train Loss: {train_metrics['total_loss']:.4f}"
            ]
            
            if (epoch + 1) % save_interval == 0:
                if ckpt_dir:
                    torch.save(model.state_dict(), ckpt_dir / f"{MD_TAG}_epoch_{epoch+1}.pt")
                    print(f"Saved epoch {epoch+1} checkpoint")
            
            val_metrics = md_validate(model, val_loader, fabric, num_samples=val_samples, trainer=trainer)
            
            if fabric.is_global_zero:
                log_info.append(f"Val Loss: {val_metrics.get('total_loss', 'N/A')}")
            
            print(" | ".join(log_info))
            
            if val_metrics.get('total_loss', np.inf) < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                if ckpt_path:
                    torch.save(model.state_dict(), ckpt_path)
                print(f"Saved best model with val loss: {best_val_loss:.4f}")
    
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default="train",
                        help="Predefined dataset split")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--samples", type=int, default=-1,
                        help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")

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
        'seed': args.seed,
        'split': args.split,
        'val_split': args.val_split,
        
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'samples': args.samples,

        'lr': cfg.lr,
        'eps': cfg.eps,
        'betas': cfg.betas,
        'weight_decay': cfg.weight_decay,
        'precision': cfg.precision,

        'ckpt_path': args.ckpt_path,
        'save_interval': args.save_interval,

        'log': args.log,
        'log_dir': args.log_dir,
        'log_interval': args.log_interval,

        'dist': args.dist,
        'restore': args.restore,
        'fabric_config': get_fabric_config(dist=args.dist)
    }

    if args.dist:
        set_dist_config(
            config,
            main_addr=args.addr,
            main_port=args.port,
            num_nodes=args.nodes
        )

    train(config)

if __name__ == '__main__':
    main()
