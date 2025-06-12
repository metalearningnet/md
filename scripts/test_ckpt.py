import sys
import unittest
from pathlib import Path

_script_dir = Path(__file__).parent
sys.path.append(str(_script_dir))
from train import train
from test import test

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from utils import cfg, default_dataset_path

NR_EPOCHS = 3
NR_SAMPLES = 16

class TestCkpt(unittest.TestCase):
    def setUp(self):
        self.train_args = {
            'path': default_dataset_path,
            'split': 'train',
            'epochs': NR_EPOCHS,
            'samples': NR_SAMPLES,
            'batch_size': 1,
            'dummy': True,
            'fabric_config': {
                'precision': cfg.precision
            }
        }

        self.test_args = {
            'path': default_dataset_path,
            'split': 'test',
            'model_path': cfg.ckpt_dir / cfg.md_file,
            'samples': 1,
            'batch_size': 1,
            'fabric_config': {
                'precision': cfg.precision
            }
        }

    def test_evaluation(self):
        train(self.train_args)
        test_metrics = test(self.test_args)
        self.assertIsInstance(test_metrics, dict)

if __name__ == '__main__':
    unittest.main()
