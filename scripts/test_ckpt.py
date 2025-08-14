import sys
import unittest
from pathlib import Path

_script_dir = Path(__file__).parent
sys.path.append(str(_script_dir))
from train import train
from test import test

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from utils import cfg, default_dataset_path, get_fabric_config

NR_EPOCHS = 1
NR_SAMPLES = 1

class TestCkpt(unittest.TestCase):
    def setUp(self):
        self.train_args = {
            'path': default_dataset_path,
            'split': 'train',
            'epochs': NR_EPOCHS,
            'samples': NR_SAMPLES,
            'ckpt_path': cfg.ckpt_path,
            'batch_size': 1,
            'fabric_config': get_fabric_config()
        }

        self.test_args = {
            'path': default_dataset_path,
            'split': 'test',
            'model_path': cfg.ckpt_path,
            'samples': 1,
            'batch_size': 1,
            'fabric_config': get_fabric_config()
        }

    def test_evaluation(self):
        train(self.train_args)
        test_metrics = test(self.test_args)
        self.assertIsInstance(test_metrics, dict)

if __name__ == '__main__':
    unittest.main()
