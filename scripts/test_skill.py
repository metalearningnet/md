import sys
import torch
import unittest
import torch.nn as nn
from pathlib import Path

_root_dir = Path(__file__).parent.parent
_src_dir = _root_dir / 'src'
sys.path.append(str(_src_dir))

from skill import SkillMemory

MEM_TYPE = 'mal'

class TestSkill(unittest.TestCase):
    def setUp(self):
        self.seq_len = 10
        self.batch_size = 4
        self.state_dim = 128
        self.action_dim = 4
        self.hidden_dim = 64
        
        self.states = torch.randn(self.batch_size, self.seq_len, self.state_dim)
        self.skill_memory = SkillMemory(
            mem_type=MEM_TYPE,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
    
    def test_action_logits(self):
        float_state = torch.randn(self.batch_size, self.seq_len, self.state_dim)
        action_logits = self.skill_memory.forward(float_state)['action_logits']
        self.assertEqual(action_logits.squeeze().shape, (self.batch_size, self.seq_len, self.action_dim))
        self.assertTrue(
            action_logits.dtype in (torch.float32, torch.float64),
            "Tensor should be float32 or float64"
        )
    
    def test_loss_computation(self):
        outputs = self.skill_memory(self.states)
        losses = self.skill_memory.compute_losses(outputs)
        self.assertIn('total_loss', losses)

    def test_action_generation(self):
        action_logits = self.skill_memory.forward(self.states)['action_logits']
        self.assertEqual(action_logits.squeeze().shape, (self.batch_size, self.seq_len, self.action_dim))
    
    def test_edge_cases(self):
        single_out = self.skill_memory(
            torch.randn(self.batch_size, 1, self.state_dim)
        )
        self.assertEqual(single_out['m'].shape, (self.batch_size, 1, self.skill_memory.hidden_dim))
        
        empty_state = torch.randn(self.batch_size, 0, self.state_dim)
        with self.assertRaisesRegex(AssertionError, "seq_len > 0"):
            self.skill_memory(empty_state)

    def test_variable_sequence_lengths(self):
        for seq_len in [1, 4, 7, 10, 15]:
            states = torch.randn(self.batch_size, seq_len, self.state_dim)
            outputs = self.skill_memory(states)
            self.assertEqual(outputs['m'].shape, (self.batch_size, seq_len, self.skill_memory.hidden_dim),
                         f"Failed for seq_len={seq_len}")

    def test_extreme_input_values(self):
        extreme_states = torch.randn(self.batch_size, self.seq_len, self.state_dim) * 1e6
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
