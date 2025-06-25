import sys
import torch
import unittest
import torch.nn as nn
from pathlib import Path

_root_dir = Path(__file__).parent.parent
_src_dir = _root_dir / 'src'
sys.path.append(str(_src_dir))

from skill import SkillMemory

class TestSkill(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 10
        self.state_dim = 64
        self.action_dim = 4
        self.hidden_dim = 64
        
        self.states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)    
        self.skill_memory = SkillMemory(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            mac_depth=1,
            mac_segment_len=4,
            mac_persistent_mem_tokens=4
        )
    
    def test_initialization(self):
        self.assertTrue(hasattr(self.skill_memory, 'mac'))
        self.assertTrue(hasattr(self.skill_memory, 'prior_net'))
        self.assertTrue(hasattr(self.skill_memory, 'policy'))
        
        output_layer = self.skill_memory.policy.output_layer
        if isinstance(output_layer, nn.Sequential):
            final_layer = output_layer[-1]
            self.assertIsInstance(
                final_layer, nn.Linear,
                f"Final layer should be Linear, got {type(final_layer)}"
            )
            self.assertEqual(
                final_layer.out_features, self.action_dim,
                f"Output dimension mismatch. Expected {self.action_dim}, got {final_layer.out_features}"
            )
        elif isinstance(output_layer, nn.Linear):
            self.assertEqual(
                output_layer.out_features, self.action_dim,
                f"Output dimension mismatch. Expected {self.action_dim}, got {output_layer.out_features}"
            )
        else:
            self.fail(
                f"Unexpected output_layer type: {type(output_layer)}. "
                "Should be nn.Linear or nn.Sequential ending with Linear layer"
            )

    def test_action_logits(self):
        float_state = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        action_logits = self.skill_memory.forward(float_state)['action_logits']
        self.assertEqual(action_logits.squeeze().shape, (self.batch_size, self.seq_len, self.action_dim))
        self.assertTrue(
            action_logits.dtype in (torch.float32, torch.float64),
            "Tensor should be float32 or float64"
        )
    
    def test_loss_computation(self):
        outputs = self.skill_memory(self.states)
        losses = self.skill_memory.compute_losses(outputs)
        
        required_losses = ['mi_loss', 'entropy', 'adv_loss', 'kl_loss']
        self.assertIn('total_loss', losses)
        self.assertIn('loss_components', losses)
        for loss_name in required_losses:
            self.assertIn(loss_name, losses['loss_components'])
        
        for k, v in losses['loss_components'].items():
            self.assertTrue(isinstance(v, float))

    def test_action_generation(self):
        action_logits = self.skill_memory.forward(self.states)['action_logits']
        self.assertEqual(action_logits.squeeze().shape, (self.batch_size, self.seq_len, self.action_dim))

    def test_gradient_flow(self):
        self.skill_memory.train()
        outputs = self.skill_memory(self.states)
        losses = self.skill_memory.compute_losses(outputs)
        losses['total_loss'].backward()
        
        critical_components = [
            'attn', 'ff', 'policy', 'forward_model',
            'disc_', 'skill_pred', 'mmi_', 'mem'
        ]
        
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
                if any(comp in name for comp in critical_components):
                    self.assertIsNotNone(param.grad, f"No gradient for {name}")
                    if param.grad is not None:
                        self.assertFalse(
                            torch.allclose(param.grad, torch.zeros_like(param.grad)),
                            f"Zero gradient for critical param: {name}"
                        )
                elif any(comp in name for comp in optional_components):
                    if param.grad is None:
                        print(f"Warning: No gradient for optional param {name}")
                    elif torch.allclose(param.grad, torch.zeros_like(param.grad)):
                        print(f"Warning: Zero gradient for conditional param {name}")

    def test_edge_cases(self):
        single_out = self.skill_memory(
            torch.randn(self.batch_size, 1, self.hidden_dim)
        )
        self.assertEqual(single_out['m'].shape, (self.batch_size, 1, self.hidden_dim))
        
        empty_state = torch.randn(self.batch_size, 0, self.hidden_dim)
        with self.assertRaisesRegex(AssertionError, "seq_len > 0"):
            self.skill_memory(empty_state)

    def test_variable_sequence_lengths(self):
        for seq_len in [1, 4, 7, 10, 15]:
            states = torch.randn(self.batch_size, seq_len, self.hidden_dim)
            outputs = self.skill_memory(states)
            self.assertEqual(outputs['m'].shape, (self.batch_size, seq_len, self.hidden_dim),
                         f"Failed for seq_len={seq_len}")

    def test_extreme_input_values(self):
        extreme_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim) * 1e6
        outputs = self.skill_memory(extreme_states)
        
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                self.assertFalse(torch.isnan(v).any(), f"NaN in {k}")
                self.assertFalse(torch.isinf(v).any(), f"Inf in {k}")

    def test_device_portability(self):
        if torch.cuda.is_available():
            try:
                gpu_model = self.skill_memory.cuda()
                gpu_states = self.states.cuda()
                
                outputs = gpu_model(gpu_states)
                self.assertEqual(outputs['m'].device.type, 'cuda', "Output should be on GPU")
                
                cpu_model = gpu_model.cpu()
                cpu_outputs = cpu_model(self.states)
                self.assertEqual(cpu_outputs['m'].device.type, 'cpu', "Output should be on CPU")
            except RuntimeError as e:
                self.fail(f"CUDA device portability failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()
