import sys
import torch
import unittest
from pathlib import Path
from datasets import Dataset
from torch.utils.data import DataLoader

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from loader import MDLoader
from utils import default_dataset_path

class TestLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_config = {
            'path': default_dataset_path,
            'max_length': 128,
            'batch_size': 2
        }

        cls.loader = MDLoader(
            path=cls.test_config['path'],
            max_length=cls.test_config['max_length']
        )

    def test_initialization(self):
        self.assertIsNotNone(self.loader.tokenizer)
        self.assertGreater(len(self.loader), 0)
        
        self.assertIsNotNone(self.loader.tokenizer.pad_token,
                            "Pad token should be defined")
        
        if self.loader.tokenizer.eos_token is not None:
            self.assertTrue(
                self.loader.tokenizer.pad_token == self.loader.tokenizer.eos_token or
                self.loader.tokenizer.pad_token in ['<pad>', '<|endoftext|>'],
                f"Invalid pad token {self.loader.tokenizer.pad_token}"
            )

    def test_collate_fn_padding(self):
        dummy_batch = [
            {'input_ids': torch.tensor([1, 2]), 'labels': torch.tensor([2, 3])},
            {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4])}
        ]
        
        collated = self.loader.collate_fn(dummy_batch)
        
        if 'input_ids' in collated and 'labels' in collated:
            self.assertEqual(collated['input_ids'].shape, (2, 2))
            self.assertEqual(collated['labels'].shape, (2, 2))
            
            self.assertTrue(torch.equal(
                collated['input_ids'],
                torch.tensor([[1, 2], [1, 2]])
            ))
            self.assertTrue(torch.equal(
                collated['labels'],
                torch.tensor([[3, -100], [3, 4]])
            ))
            self.assertTrue(torch.equal(
                collated['attention_mask'],
                torch.tensor([[1, 1], [1, 1]])
            ))

    def test_edge_cases(self):
        empty_loader = MDLoader(path="", dataset=Dataset.from_dict({}))
        with self.assertRaises(ValueError):
            empty_loader.get_dataloaders(batch_size=self.test_config['batch_size'], val_split=0.2)

    def test_dataset_splitting(self):
        full_loader = MDLoader(
            path=self.test_config['path'],
            max_length=self.test_config['max_length']
        )
        
        for split_ratio in [0.2, 0.5, 0.8]:
            train_loader, val_loader = full_loader.get_dataloaders(
                batch_size=self.test_config['batch_size'],
                val_split=split_ratio
            )
            
            total = len(full_loader)
            val_size = int(total * split_ratio)
            train_size = total - val_size
            
            self.assertEqual(len(train_loader.dataset), train_size)
            self.assertEqual(len(val_loader.dataset), val_size)
            
            train_indices = set(train_loader.dataset.indices)
            val_indices = set(val_loader.dataset.indices)
            self.assertTrue(train_indices.isdisjoint(val_indices))
            
            self.assertEqual(
                len(train_loader.dataset) + len(val_loader.dataset),
                total
            )

    def test_factory_method(self):
        main_loader = MDLoader(
            path=self.test_config['path'],
            max_length=self.test_config['max_length']
        )
        train_loader, val_loader = main_loader.get_dataloaders(
            batch_size=self.test_config['batch_size'],
            val_split=0.3
        )
        
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertEqual(
            len(train_loader.dataset) + len(val_loader.dataset), 
            len(main_loader)
        )

    def test_split_reproducibility(self):
        loader = MDLoader(
            path=self.test_config['path'],
            max_length=self.test_config['max_length']
        )
        
        l1_train, l1_val = loader.get_dataloaders(batch_size=self.test_config['batch_size'], val_split=0.2, seed=42)
        l2_train, l2_val = loader.get_dataloaders(batch_size=self.test_config['batch_size'], val_split=0.2, seed=42)
        
        self.assertEqual(l1_train.dataset.indices, l2_train.dataset.indices)
        self.assertEqual(l1_val.dataset.indices, l2_val.dataset.indices)
        
        _, l3_val = loader.get_dataloaders(batch_size=self.test_config['batch_size'], val_split=0.2, seed=24)
        self.assertNotEqual(l1_val.dataset.indices, l3_val.dataset.indices)

if __name__ == '__main__':
    unittest.main()
