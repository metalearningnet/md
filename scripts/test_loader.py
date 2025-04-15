import sys
import torch
import unittest
from pathlib import Path
from datasets import Dataset
from torch.utils.data import DataLoader

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from loader import MDLoader

class TestLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_config = {
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-2-raw-v1',
            'max_length': 128,
            'state_window': 2,
            'batch_size': 2
        }

        cls.loader = MDLoader(
            dataset_name=cls.test_config['dataset_name'],
            dataset_config=cls.test_config['dataset_config'],
            max_length=cls.test_config['max_length'],
            state_window=cls.test_config['state_window']
        )

    def test_initialization(self):
        """Test loader initialization and properties"""
        self.assertIsNotNone(self.loader.tokenizer)
        self.assertIsNotNone(self.loader.embed_model)
        self.assertGreater(len(self.loader), 0)
        
        # Test tokenizer settings
        self.assertIsNotNone(self.loader.tokenizer.pad_token,
                            "Pad token should be defined")
        
        # Verify either pad == eos or special handling
        if self.loader.tokenizer.eos_token is not None:
            self.assertTrue(
                self.loader.tokenizer.pad_token == self.loader.tokenizer.eos_token or
                self.loader.tokenizer.pad_token in ['<pad>', '<|endoftext|>'],
                f"Invalid pad token {self.loader.tokenizer.pad_token}"
            )

    def test_single_sample_structure(self):
        """Test structure of a single sample"""
        sample = self.loader[0]
        
        # Check keys
        expected_keys = {'raw_text', 'states', 'input_ids', 'labels'}
        self.assertEqual(set(sample.keys()), expected_keys)
        
        # Check dimensions
        seq_len = sample['input_ids'].shape[0]
        self.assertEqual(sample['states'].shape[0], seq_len)
        self.assertEqual(sample['labels'].shape[0], seq_len)

    def test_state_generation(self):
        """Test state generation with valid sequence length"""
        test_text = "This is a sample text that will generate enough tokens for testing."
        
        # Verify token count meets minimum requirements
        tokens = self.loader.tokenizer.encode(test_text)
        self.assertGreaterEqual(
            len(tokens), 
            self.loader.min_sequence_length,
            "Test text is too short for current configuration"
        )
        
        # Process the example
        sample = self.loader._process_example({'text': test_text})
        
        # Validate output structure
        self.assertIn('states', sample)
        self.assertIn('input_ids', sample)
        self.assertEqual(len(sample['input_ids']), len(tokens)-1)

    def test_collate_fn_padding(self):
        """Test batch padding functionality"""
        # Create dummy batch with varying lengths
        dummy_batch = [
            {
                'states': torch.randn(2, 768),
                'input_ids': torch.tensor([1, 2]),
                'labels': torch.tensor([2, 3])
            },
            {
                'states': torch.randn(3, 768),
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([2, 3, 4])
            }
        ]
        
        collated = self.loader.collate_fn(dummy_batch)
        
        # Test padding dimensions
        self.assertEqual(collated['input_ids'].shape, (2, 3))
        self.assertEqual(collated['states'].shape, (2, 3, 768))
        self.assertEqual(collated['labels'].shape, (2, 3))
        
        # Test padding values
        self.assertEqual(collated['input_ids'][0][-1].item(), self.loader.tokenizer.pad_token_id)
        self.assertEqual(collated['labels'][0][-1].item(), -100)
        
        # Test masks
        self.assertTrue(torch.all(collated['attention_mask'][0] == torch.tensor([1, 1, 0])))
        self.assertTrue(torch.all(collated['state_mask'][0] == torch.tensor([1, 1, 0])))

    def test_dataloader_integration(self):
        """Test integration with DataLoader"""
        # Create filtered dataset excluding empty texts
        filtered_dataset = self.loader.dataset.filter(
            lambda x: len(x['text'].strip()) > 0,
            num_proc=4
        )
        
        # Skip test if filtered dataset is too small
        min_required = self.test_config['batch_size'] * 2
        if len(filtered_dataset) < min_required:
            self.skipTest(f"Filtered dataset too small (needs {min_required}+ samples)")
        
        # Create new loader with filtered data
        filtered_loader = MDLoader(
            dataset_name=self.test_config['dataset_name'],
            dataset_config=self.test_config['dataset_config'],
            max_length=self.test_config['max_length'],
            state_window=self.test_config['state_window'],
            dataset=filtered_dataset
        )
        
        dataloader = filtered_loader.get_dataloader(
            batch_size=self.test_config['batch_size'],
            shuffle=False,
            drop_last=True  # Ensure consistent batch sizes
        )
        
        full_batches = 0
        for batch_idx, batch in enumerate(dataloader):
            # Verify batch dimensions
            self.assertEqual(batch['input_ids'].shape[0], self.test_config['batch_size'])
            self.assertEqual(batch['states'].shape[0], self.test_config['batch_size'])
            full_batches += 1
            
            # Check batch structure
            self.assertIn('states', batch)
            self.assertIn('input_ids', batch)
            self.assertIn('labels', batch)
            self.assertIn('attention_mask', batch)
            self.assertIn('state_mask', batch)
            
            # Check mask consistency
            input_mask = (batch['input_ids'] != filtered_loader.tokenizer.pad_token_id)
            self.assertTrue(torch.all(input_mask == (batch['attention_mask'] == 1)))
            
            # Check valid states (non-padded states should match state_mask)
            state_non_zero = (batch['states'].abs().sum(dim=-1)) != 0
            self.assertTrue(torch.all(state_non_zero == (batch['state_mask'] == 1)))
            
            # Verify sequence lengths
            seq_lengths = batch['attention_mask'].sum(dim=1).long()
            for i in range(self.test_config['batch_size']):
                seq_len = seq_lengths[i]
                
                # Check valid states match sequence length
                self.assertEqual(
                    batch['state_mask'][i][:seq_len].sum(),
                    seq_len,
                    msg=f"State mask mismatch in batch {batch_idx} sample {i}"
                )
                
                # Verify label alignment
                self.assertTrue(torch.all(
                    batch['labels'][i][:seq_len-1] == batch['input_ids'][i][1:seq_len]
                ))
        
        # Verify we got at least 2 full batches
        self.assertGreaterEqual(full_batches, 2, "Insufficient full batches for testing")
        
        # Verify total samples handled
        total_samples = full_batches * self.test_config['batch_size']
        self.assertLessEqual(total_samples, len(filtered_dataset))

    def test_edge_cases(self):
        """Test handling of edge cases"""
        # Empty dataset split
        empty_loader = MDLoader(dataset_name="", dataset=Dataset.from_dict({}))
        with self.assertRaises(ValueError):
            empty_loader.get_dataloaders(batch_size=self.test_config['batch_size'], val_split=0.2)

    def test_embeddings_consistency(self):
        """Test if states match MDLoader's EXACT generation pipeline"""
        # Find a sample with non-empty states
        valid_samples = [
            i for i in range(len(self.loader))
            if len(self.loader[i]['states']) > 0
        ]
        if not valid_samples:
            self.skipTest("No valid samples with states")
        
        sample = self.loader[valid_samples[0]]
        
        # Reconstruct EXACT tokens used by MDLoader
        tokens = self.loader.tokenizer.encode(
            sample['raw_text'],
            max_length=self.test_config['max_length'],
            truncation=True,
            return_tensors='pt'
        )[0]
        
        with torch.no_grad():
            # 1. Get raw embeddings
            raw_embeddings = self.loader.embed_model(tokens)
            # 2. Apply projection
            embeddings = self.loader.embed_proj(raw_embeddings)
        
        # Replicate MDLoader's state generation pipeline step-by-step
        if len(tokens) > 1:
            states = torch.nn.functional.avg_pool1d(
                embeddings.unsqueeze(0).transpose(1, 2),
                kernel_size=min(self.test_config['state_window'], len(tokens)-1),
                stride=1,
                padding=0
            ).squeeze(0).transpose(0, 1)
            
            if len(states) < len(tokens) - 1:
                padding = embeddings[len(states):-1].mean(dim=0, keepdim=True)
                states = torch.cat([states, padding])
        else:
            states = torch.zeros(0, embeddings.size(-1))
        
        # Convert to same device as sample
        states = states.to(sample['states'].device)
        
        # Validate equivalence
        self.assertTrue(
            torch.allclose(
                sample['states'],
                states,
                atol=1e-5,
                rtol=1e-3
            ),
            msg="State mismatch - verify pooling parameters and padding logic"
        )

    def test_dataset_splitting(self):
        """Test manual train/validation splitting"""
        full_loader = MDLoader(
            dataset_name=self.test_config['dataset_name'],
            dataset_config=self.test_config['dataset_config'],
            max_length=self.test_config['max_length'],
            state_window=self.test_config['state_window']
        )
        
        # Test different split ratios
        for split_ratio in [0.2, 0.5, 0.8]:
            train_loader, val_loader = full_loader.get_dataloaders(
                batch_size=self.test_config['batch_size'],
                val_split=split_ratio
            )
            
            total = len(full_loader)
            val_size = int(total * split_ratio)
            train_size = total - val_size
            
            # Validate split sizes
            self.assertEqual(len(train_loader.dataset), train_size)
            self.assertEqual(len(val_loader.dataset), val_size)
            
            # Verify no overlap
            train_indices = set(train_loader.dataset.indices)
            val_indices = set(val_loader.dataset.indices)
            self.assertTrue(train_indices.isdisjoint(val_indices))
            
            # Verify total preservation
            self.assertEqual(
                len(train_loader.dataset) + len(val_loader.dataset),
                total
            )

    def test_factory_method(self):
        """Test factory method"""
        main_loader = MDLoader(
            dataset_name=self.test_config['dataset_name'],
            dataset_config=self.test_config['dataset_config'],
            max_length=self.test_config['max_length'],
            state_window=self.test_config['state_window']
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
        """Test split reproducibility with seed control"""
        loader = MDLoader(
            dataset_name=self.test_config['dataset_name'],
            dataset_config=self.test_config['dataset_config'],
            max_length=self.test_config['max_length'],
            state_window=self.test_config['state_window']
        )
        
        # Get two splits with same seed
        l1_train, l1_val = loader.get_dataloaders(batch_size=self.test_config['batch_size'], val_split=0.2, seed=42)
        l2_train, l2_val = loader.get_dataloaders(batch_size=self.test_config['batch_size'], val_split=0.2, seed=42)
        
        # Compare indices
        self.assertEqual(l1_train.dataset.indices, l2_train.dataset.indices)
        self.assertEqual(l1_val.dataset.indices, l2_val.dataset.indices)
        
        # Test different seeds produce different splits
        _, l3_val = loader.get_dataloaders(batch_size=self.test_config['batch_size'], val_split=0.2, seed=24)
        self.assertNotEqual(l1_val.dataset.indices, l3_val.dataset.indices)

    def test_predefined_splits(self):
        """Test loading of predefined splits"""
        for split in ['train', 'validation', 'test']:
            try:
                loader = MDLoader(
                    dataset_name=self.test_config['dataset_name'],
                    dataset_config=self.test_config['dataset_config'],
                    split=split
                )
                self.assertGreater(len(loader), 0)
            except ValueError:
                pass

if __name__ == '__main__':
    unittest.main()