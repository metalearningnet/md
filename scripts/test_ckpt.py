import sys
import torch
import unittest
from pathlib import Path

_script_dir = Path(__file__).parent
sys.path.append(str(_script_dir))
from test import test

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD
from utils import cfg

class TestCkpt(unittest.TestCase):
    def setUp(self):
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = cfg.save_dir / cfg.md_file
        
        # Create dummy model and save
        self.dummy_model = MD()
        torch.save({'model': self.dummy_model.state_dict()}, self.save_path)
        
        # Common test arguments
        self.test_args = {
            'model_path': self.save_path,
            'batch_size': 16,
            'dataset_name': 'ag_news',
            'fabric_config': {
                'accelerator': cfg.accelerator,
                'precision': cfg.precision
            }
        }

    def test_evaluation(self):
        # Call the test function
        metrics = test(**self.test_args)
        
        # Verify metrics were returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_loss', metrics)

if __name__ == '__main__':
    unittest.main()