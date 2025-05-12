import sys
import unittest
from pathlib import Path

_script_dir = Path(__file__).parent
sys.path.append(str(_script_dir))
from train import train
from test import test

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from utils import cfg, default_dataset

class TestCkpt(unittest.TestCase):
    def setUp(self):
        self.train_args = {
            'name': default_dataset,
            'split': 'train',
            'epochs': 1,
            'batches': 1,
            'batch_size': 1,
            'dummy': True,
            'fabric_config': {
                'accelerator': cfg.accelerator,
                'precision': cfg.precision
            }
        }

        self.test_args = {
            'name': default_dataset,
            'split': 'test',
            'model_path': cfg.ckpt_dir / cfg.md_file,
            'batches': 1,
            'batch_size': 1,
            'fabric_config': {
                'accelerator': cfg.accelerator,
                'precision': cfg.precision
            }
        }

    def test_evaluation(self):
        train(self.train_args)
        test_metrics = test(self.test_args)
        self.assertIsInstance(test_metrics, dict)

if __name__ == '__main__':
    unittest.main()