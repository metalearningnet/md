import sys
import torch
import argparse
from pathlib import Path

_src_dir = Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD
from loader import MDLoader
from utils import md_validate, cfg, device

def test(model_path: Path, dataset_name: str, dataset_config: str = None):
    """Main evaluation function"""
    model = MD.from_pretrained()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Prepare test dataset with test split
    loader = MDLoader(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split="test"
    )
    test_loader = loader.get_dataloader(shuffle=False)
    
    # Use md_validate instead of md_test
    test_metrics = md_validate(model, test_loader, device)
    
    # Rename metrics for consistency (optional)
    test_metrics['action_accuracy'] = test_metrics.pop('action_acc', 0.0)
    
    print("\nTest Results:")
    for k, v in test_metrics.items():
        print(f"{k:20}: {v:.4f}")
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Test MD Model")
    parser.add_argument("--name", required=True, 
                      help="HuggingFace dataset name")
    parser.add_argument("--config", default=None,
                      help="HF dataset configuration name")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                      help="Checkpoint directory")
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    model_path = save_dir / cfg.md_file
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    test(model_path=model_path,
        dataset_name=args.name, 
        dataset_config=args.config)

if __name__ == "__main__":
    main()