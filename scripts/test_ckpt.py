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
        
        self.dummy_model = MD()
        torch.save({'model': self.dummy_model.state_dict()}, self.save_path)
        
        self.test_args = {
            'name': 'princeton-nlp/gemma2-ultrafeedback-armorm',
            'split': 'test',
            'model_path': self.save_path,
            'batch_size': 1,
            'num_batches': 1,
            'fabric_config': {
                'accelerator': cfg.accelerator,
                'precision': cfg.precision
            }
        }

    def test_evaluation(self):
        metrics = test(self.test_args)
        self.assertIsInstance(metrics, dict)

if __name__ == '__main__':
    unittest.main()