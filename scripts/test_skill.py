import sys
import torch
import unittest
from pathlib import Path

_root_dir = Path(__file__).parent.parent
_src_dir = _root_dir / 'src'
sys.path.append(str(_src_dir))

from skill import SkillMemory

class TestSkill(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 10
        self.state_dim = 128
        self.action_dim = 4
        self.hidden_dim = 64
        
        # Test data (now using continuous states)
        self.states = torch.randn(self.batch_size, self.seq_len, self.state_dim)
        self.actions = torch.randint(0, self.action_dim, (self.batch_size, self.seq_len))
        
        # Updated MD configuration
        self.skill_memory = SkillMemory(
            action_dim=self.action_dim,
            state_embed_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            mac_depth=1,
            mac_segment_len=4,
            mac_persistent_mem_tokens=4
        )
    
    def test_initialization(self):
        """Test proper initialization of all components"""
        # Check critical components exist
        self.assertTrue(hasattr(self.skill_memory, 'mac'))
        self.assertTrue(hasattr(self.skill_memory, 'prior_net'))
        self.assertTrue(hasattr(self.skill_memory, 'policy'))
        
        # Check output dimensions
        self.assertEqual(self.skill_memory.policy[-1].out_features, self.action_dim)
        self.assertEqual(self.skill_memory.state_embedding.out_features, self.hidden_dim)

    def test_input_type_handling(self):
        """Test float vs long input handling"""
        # Float states
        float_state = torch.randn(self.state_dim)
        _, m = self.skill_memory.generate(float_state)
        self.assertEqual(m.shape[-1], self.hidden_dim)
        
        # Integer states (simulating tokens)
        int_state = torch.randint(0, 100, (self.state_dim,))
        _, m = self.skill_memory.generate(int_state)
        self.assertEqual(m.shape[-1], self.hidden_dim)

    def test_input_type_handling(self):
        """Test float vs long input handling"""
        # Float states
        float_state = torch.randn(self.state_dim)
        action_logits, _ = self.skill_memory.generate(float_state)
        self.assertEqual(action_logits.squeeze().shape, (self.action_dim,))
        self.assertTrue(
            action_logits.dtype in (torch.float32, torch.float64),
            "Tensor should be float32 or float64"
        )
        
        # Integer states (simulating tokens)
        int_state = torch.randint(0, 100, (self.state_dim,))
        action_logits, _ = self.skill_memory.generate(int_state)
        self.assertEqual(action_logits.squeeze().shape, (self.action_dim,))
        self.assertTrue(
            action_logits.dtype in (torch.float32, torch.float64),
            "Tensor should be float32 or float64"
        )
        
    def test_loss_computation(self):
        """Test loss components are computed"""
        batch = (self.states, self.actions)
        losses = self.skill_memory.compute_losses(batch)
        
        # Check all loss components exist
        required_losses = ['mi_loss', 'entropy', 'adv_loss', 'kl_loss']
        self.assertIn('total_loss', losses)
        self.assertIn('loss_components', losses)
        for loss_name in required_losses:
            self.assertIn(loss_name, losses['loss_components'])
        
        # Check loss values are valid
        for k, v in losses['loss_components'].items():
            self.assertTrue(isinstance(v, float))

    def test_action_generation(self):
        """Test action generation"""
        state = torch.randn(self.state_dim)
        action_logits, m = self.skill_memory.generate(state)
        self.assertEqual(action_logits.squeeze().shape, (self.action_dim,))
        self.assertEqual(m.squeeze().shape, (self.hidden_dim,))

    def test_gradient_flow(self):
        """Test gradients flow through all components"""
        self.skill_memory.train()
        batch = (
            self.states,
            self.actions
        )
        
        losses = self.skill_memory.compute_losses(batch)
        losses['total_loss'].backward()
        
        # Components that MUST have gradients
        critical_components = [
            'attn', 'ff', 'policy', 'forward_model',
            'disc_', 'skill_pred', 'mmi_', 'mem'
        ]
        
        # Components that MAY have zero gradients (conditional)
        optional_components = [
            'dynamic_alpha_scale', 'gate', 'hyper_conn'
        ]

        exclude_params = [
            'mac.layers.0.0.dynamic_alpha_scale',
            'mac.layers.0.1.dynamic_alpha_scale',
            'mac.layers.0.2.dynamic_alpha_scale',
            'mac.layers.0.4.memory_model.norm.gamma',
            'mac.layers.0.4.memory_model.model.weights.0',
            'mac.layers.0.4.memory_model.model.weights.1',
        ]
        
        for name, param in self.skill_memory.named_parameters():
            if name not in exclude_params:
                # Check critical components
                if any(comp in name for comp in critical_components):
                    self.assertIsNotNone(param.grad, f"No gradient for {name}")
                    if param.grad is not None:
                        self.assertFalse(
                            torch.allclose(param.grad, torch.zeros_like(param.grad)),
                            f"Zero gradient for critical param: {name}"
                        )
                
                # Warn about optional components (don't fail test)
                elif any(comp in name for comp in optional_components):
                    if param.grad is None:
                        print(f"Warning: No gradient for optional param {name}")
                    elif torch.allclose(param.grad, torch.zeros_like(param.grad)):
                        print(f"Warning: Zero gradient for conditional param {name}")

    def test_edge_cases(self):
        """Test edge cases"""
        # Single timestep
        single_out = self.skill_memory(
            torch.randn(self.batch_size, 1, self.state_dim)
        )
        self.assertEqual(single_out['m_seq'].shape, (self.batch_size, 1, 1, self.hidden_dim))
        
        # Empty sequence (should handle gracefully)
        empty_state = torch.randn(self.batch_size, 0, self.state_dim)
        with self.assertRaisesRegex(AssertionError, "seq_len > 0"):
            self.skill_memory(empty_state)

    def test_variable_sequence_lengths(self):
        """Test handling of different sequence lengths"""
        for seq_len in [1, 4, 7, 10, 15]:  # Test various lengths including boundary cases
            states = torch.randn(self.batch_size, seq_len, self.state_dim)
            outputs = self.skill_memory(states)
            self.assertEqual(outputs['m_seq'].shape, (self.batch_size, seq_len, 1, self.hidden_dim),
                         f"Failed for seq_len={seq_len}")

    def test_extreme_input_values(self):
        """Test numerical stability with extreme inputs"""
        extreme_states = torch.randn(self.batch_size, self.seq_len, self.state_dim) * 1e6
        outputs = self.skill_memory(extreme_states)
        
        # Check for NaN/Inf in outputs
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                self.assertFalse(torch.isnan(v).any(), f"NaN in {k}")
                self.assertFalse(torch.isinf(v).any(), f"Inf in {k}")

    def test_mixed_precision_handling(self):
        """Test model works with mixed precision"""
        try:
            # Convert model and inputs to half precision
            half_model = self.skill_memory.half()
            half_states = self.states.half()
            
            # Forward pass should complete without errors
            outputs = half_model(half_states)
            self.assertEqual(outputs['m_seq'].dtype, torch.float16,
                          "Output should maintain half precision")
        except RuntimeError as e:
            self.fail(f"Mixed precision failed: {str(e)}")

    def test_device_portability(self):
        """Test model can move between devices"""
        if torch.cuda.is_available():
            try:
                # Move to GPU
                gpu_model = self.skill_memory.cuda()
                gpu_states = self.states.cuda()
                
                # Forward pass on GPU
                outputs = gpu_model(gpu_states)
                self.assertEqual(outputs['m_seq'].device.type, 'cuda',
                              "Output should be on GPU")
                
                # Move back to CPU
                cpu_model = gpu_model.cpu()
                cpu_outputs = cpu_model(self.states)
                self.assertEqual(cpu_outputs['m_seq'].device.type, 'cpu',
                              "Output should be on CPU")
            except RuntimeError as e:
                self.fail(f"Device portability failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()
