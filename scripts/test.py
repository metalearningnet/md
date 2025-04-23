import sys
import torch
import argparse
import lightning as L
from pathlib import Path

_src_dir = Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD
from loader import MDLoader
from utils import md_validate, cfg, add_dist_config

def test(config: dict):
    """
    config:
        - name: Dataset name (str)
        - dataset_config: Dataset configuration name (str)
        - split: Dataset split name (e.g., "test") (str)
        - model_path: Model path (str)
        - batch_size: Testing batch size (int)
        - fabric_config: Configuration options for the Lightning Fabric setup (dict)
        - batches: Number of batches (int)
    """
    
    dataset_name = config['name']
    split = config.get('split', 'test')
    fabric_config = config['fabric_config']
    num_batches = config.get('batches', -1)
    batch_size = config.get('batch_size', 1)
    dataset_config = config.get('dataset_config')
    model_path = config.get('model_path', cfg.ckpt_dir / cfg.md_file)
    
    fabric = L.Fabric(**fabric_config)
    fabric.launch()

    checkpoint = torch.load(model_path)
    model_state_dict = checkpoint['model']
    model = MD.from_pretrained()
    model.load_state_dict(model_state_dict)
    model = fabric.setup(model)
    
    loader = MDLoader(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split
    )
    
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if cfg.log:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.test_log
        log_interval = cfg.log_interval
    else:
        log_path = None
        log_interval = 0

    test_loader = loader.get_dataloader(batch_size=batch_size, shuffle=False)
    test_loader = fabric.setup_dataloaders(test_loader, use_distributed_sampler=True)
    test_metrics = md_validate(model, test_loader, fabric, num_batches=num_batches, log_path=log_path, log_interval=log_interval)

    print("\nTest Results:")
    for k, v in test_metrics.items():
        print(f"{k:20}: {v:.4f}")
    
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Test the MD Model")

    # Testing configuration
    parser.add_argument("--name", type=str, default="princeton-nlp/gemma2-ultrafeedback-armorm", 
                        help="Dataset name")
    parser.add_argument("--config", type=str, default=None,
                        help="Dataset configuration name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split name")
    parser.add_argument("--ckpt_dir", type=str, default=cfg.ckpt_dir,
                        help="Checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Testing batch size")

    # Distributed testing configuration
    parser.add_argument("--dist", action="store_true", default=False,
                        help="Enable distributed testing")
    parser.add_argument("--addr", type=str, default=None,
                        help="Master address for distributed testing")
    parser.add_argument("--port", type=int, default=None,
                        help="Master port for distributed testing")
    parser.add_argument("--nodes", type=int, default=None,
                        help="The number of nodes for distributed testing")

    args = parser.parse_args()
    
    ckpt_dir = Path(args.ckpt_dir)
    model_path = ckpt_dir / cfg.md_file
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    fabric_config = {
        'accelerator': cfg.accelerator,
        'precision': cfg.precision
    }
    
    if args.dist:
        add_dist_config(
            fabric_config, 
            main_addr=args.addr, 
            main_port=args.port, 
            num_nodes=args.nodes
        )
    
    config = {
        'name': args.name,
        'dataset_config': args.config,
        'split': args.split,
        'model_path': model_path,
        'batch_size': args.batch_size,
        'fabric_config': fabric_config
    }

    test(config)

if __name__ == '__main__':
    main()