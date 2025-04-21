import sys
import torch
import unittest
from typing import Dict
from pathlib import Path

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD

class TestMD(unittest.TestCase):
    def setUp(self):
        self.model = MD(flash_attn=False)
        self.seq_len = 10
        self.batch_size = 2
    
    def _create_dummy_inputs(self, batch_size: int = None) -> Dict:
        batch = batch_size or self.batch_size
        return {
            'input_ids': torch.randint(
                0, self.model.config.vocab_size, (batch, self.seq_len)
            )
        }
    
    def _create_dummy_inputs(self, batch_size: int = None) -> Dict:
        batch = batch_size or self.batch_size
        input_ids = torch.randint(
            0, self.model.config.vocab_size, (batch, self.seq_len)
        )
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids)
        }
    
    def test_sample_generation(self):
        # Generate test inputs
        input_ids = torch.randint(0, self.model.config.vocab_size, (self.batch_size, self.seq_len))
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Verify output shapes
        self.assertEqual(outputs['logits'].shape, 
                        (self.batch_size, self.seq_len * 2, self.model.config.vocab_size))
        self.assertEqual(outputs['action_logits'].shape,
                        (self.batch_size, self.seq_len, self.model.skill_memory.action_dim))
    
    def test_parameter_update(self):
        params = []
        for param_list in self.model.get_trainable_parameters().values():
            params.extend(param_list)
        optimizer = torch.optim.Adam(params)
        initial_params = {n: p.clone() for n, p in self.model.named_parameters()}
        input_ids = torch.randint(
            0, 
            self.model.config.vocab_size,
            (self.batch_size, self.seq_len)
        )

        # Forward pass
        optimizer.zero_grad()
        output = self.model(input_ids)
        loss = output['logits'].mean() + output['action_logits'].mean()
        loss.backward()
        optimizer.step()

        # Verify parameter updates
        changed = False
        for name, param in self.model.named_parameters():
            if not torch.allclose(initial_params[name], param):
                changed = True
                break
        self.assertTrue(changed, "No parameters updated during training")

    def test_forward_pass_dimensions(self):
        inputs = self._create_dummy_inputs()
        outputs = self.model(**inputs)
        
        # Verify LM output dimensions
        self.assertEqual(
            outputs['logits'].shape,
            (self.batch_size, self.seq_len * 2, self.model.config.vocab_size)
        )
        
        # Check action projection dimensions
        self.assertEqual(
            outputs['action_logits'].shape,
            (self.batch_size, self.seq_len, self.model.skill_memory.action_dim)
        )
    
    def test_parameter_freezing(self):
        # Check LM parameters frozen
        for name, param in self.model.lm.named_parameters():
            self.assertFalse(param.requires_grad, f"LLM parameter {name} should be frozen")
            
        # Verify trainable components
        components = ['action_proj', 'skill_memory']
        for comp in components:
            for param in getattr(self.model, comp).parameters():
                self.assertTrue(param.requires_grad, f"{comp} parameter should be trainable")

    def test_training_dynamics(self):
        params = []
        for param_list in self.model.get_trainable_parameters().values():
            params.extend(param_list)
        optimizer = torch.optim.Adam(params)
        initial_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        
        # Training step
        inputs = self._create_dummy_inputs()
        outputs = self.model(**inputs)
        loss = outputs['logits'].mean() + outputs['action_logits'].mean()
        loss.backward()
        optimizer.step()
        
        # Check parameter updates
        updated = False
        for name, param in self.model.named_parameters():
            if param.requires_grad and not torch.allclose(param, initial_params[name]):
                updated = True
                break
        self.assertTrue(updated, "Trainable parameters should update")

    def test_generation_interface(self):
        max_length = self.seq_len * 2 + 1
        inputs = self._create_dummy_inputs()
        generated = self.model.generate(**inputs, max_length=max_length)
        
        # Verify generation dimensions
        self.assertEqual(generated.shape[0], self.batch_size)
        self.assertTrue(generated.shape[1] == 1, "Invalid number of generated tokens")

    def test_edge_cases(self):
        with self.subTest("Variable batch sizes"):
            inputs = self._create_dummy_inputs(batch_size=1)
            outputs = self.model(**inputs)
            self.assertEqual(outputs['logits'].shape[0], 1)

    def test_device_compatibility(self):
        inputs = self._create_dummy_inputs()
        devices = [
            torch.device('cpu'),
            torch.device('cuda:0') if torch.cuda.is_available() else None,
            torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else None
        ]
        
        for device in devices:
            if device is None:
                continue
                
            try:
                # Move model and inputs to target device
                self.model.to(device)
                device_inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**device_inputs)
                
                # Verify output device matches
                self.assertEqual(outputs['logits'].device.type, device.type,
                            f"Output device {outputs['logits'].device.type} "
                            f"doesn't match expected {device.type}")
                
                # Verify CUDA device index when applicable
                if device.type == 'cuda':
                    self.assertEqual(outputs['logits'].device.index, 0,
                                "CUDA device index should be 0")
                
                # Verify MPS specific checks if needed
                if device.type == 'mps':
                    # Add any MPS-specific verification here
                    pass
                    
            finally:
                # Clean up by moving model back to CPU
                self.model.cpu()
                # Clear any gradients that might have been created
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = None

    def test_mixed_precision(self):
        inputs = self._create_dummy_inputs()
        
        # Test CUDA if available
        if torch.cuda.is_available():
            device_type = 'cuda'
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=True
            ):
                try:
                    outputs = self.model.to(device_type)(**{k: v.to(device_type) for k, v in inputs.items()})
                    loss = outputs['logits'].mean() + outputs['action_logits'].mean()
                    loss.backward()
                    
                    # Verify no NaNs in gradients
                    for param in self.model.parameters():
                        if param.grad is not None:
                            self.assertFalse(torch.isnan(param.grad).any(), 
                                        "Found NaN in CUDA gradients")
                finally:
                    # Clean up by moving model back to CPU
                    self.model.cpu()
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = None

if __name__ == '__main__':
    unittest.main()
