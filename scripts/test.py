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

def test(model_path: Path, batch_size: int, dataset_name: str, dataset_config: str = None, fabric_config: dict = {}):
    """Main evaluation function"""
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
        split='test'
    )
    
    test_loader = loader.get_dataloader(batch_size=batch_size, shuffle=False)
    test_loader = fabric.setup_dataloaders(test_loader, use_distributed_sampler=True)
    test_metrics = md_validate(model, test_loader, fabric)

    print("\nTest Results:")
    for k, v in test_metrics.items():
        print(f"{k:20}: {v:.4f}")
    
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Test the MD Model")

    # Testing configuration
    parser.add_argument("--name", type=str, required=True, 
                        help="HuggingFace dataset name")
    parser.add_argument("--config", type=str, default=None,
                        help="HF dataset configuration name")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
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
    
    save_dir = Path(args.save_dir)
    model_path = save_dir / cfg.md_file
    
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
    
    test(
        model_path=model_path,
        batch_size=args.batch_size,
        dataset_name=args.name, 
        dataset_config=args.config,
        fabric_config=fabric_config
    )

if __name__ == '__main__':
    main()