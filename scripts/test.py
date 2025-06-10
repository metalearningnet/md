import sys
import torch
import argparse
import lightning as L
from pathlib import Path

_src_dir = Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD
from loader import MDLoader
from utils import md_validate, cfg, add_dist_config, default_dataset_path

def test(config: dict):
    """
    config:
        - path (str): Dataset path.
        - name (str): Dataset name.
        - split (str): Dataset split name (e.g., "test").
        - batch_size (int): Testing batch size.
        - samples (int): Number of samples.
        - ckpt (str): Checkpoint path.
        - fabric_config (dict): Configuration options for the Lightning Fabric setup.
    """
    try:
        dataset_path = config['path']
        dataset_name = config.get('name')
        split = config.get('split', 'test')
        batch_size = config.get('batch_size', 1)
        num_samples = config.get('samples', -1)
        ckpt_path = config.get('ckpt')
        fabric_config = config['fabric_config']
        
        fabric = L.Fabric(**fabric_config)
        fabric.launch()

        if ckpt_path:
            model = MD.from_pretrained(checkpoint_path=ckpt_path)
        else:
            model = MD.from_pretrained()
        
        model = fabric.setup(model)

        if model.enable_annotation:
            model.mark_forward_method('annotate')

        loader = MDLoader(
            path=dataset_path,
            name=dataset_name,
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
        test_metrics = md_validate(model, test_loader, fabric, num_samples=num_samples, log_path=log_path, log_interval=log_interval)

        print("\nTest Results:")
        for k, v in test_metrics.items():
            print(f"{k:20}: {v:.4f}")
        
        return test_metrics
    
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Test the MD Model")

    # Testing configuration
    parser.add_argument("--path", type=str, default=default_dataset_path,
                        help="Dataset path")
    parser.add_argument("--name", type=str, default=None,
                        help="Dataset name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split name")
    parser.add_argument("--ckpt", type=str, default=cfg.ckpt_path,
                        help="Checkpoint path")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Testing batch size")
    parser.add_argument("--samples", type=int, default=-1,
                        help="Number of testing samples")

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
    
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpointn file not found: {args.ckpt}")
    
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
        'path': args.path,
        'name': args.name,
        'split': args.split,
        'batch_size': args.batch_size,
        'samples': args.samples,
        'ckpt': args.ckpt,
        'fabric_config': fabric_config
    }

    test(config)
    
if __name__ == '__main__':
    main()
