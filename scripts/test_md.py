import sys
import torch
import unittest
from typing import Dict
from pathlib import Path
from unittest.mock import patch

_src_dir =  Path(__file__).parent.parent / 'src'
sys.path.append(str(_src_dir))

from md import MD

FAST_TEST = False
SHOW_RESULTS = False

def show_results(results):
    if SHOW_RESULTS:
        print(results)

class TestMD(unittest.TestCase):
    def setUp(self):
        self.model = MD(attn='sdpa')
        self.seq_len = 10
        self.batch_size = 2
        self.params = self.model.get_trainable_parameters()
    
    def _create_dummy_inputs(self, batch_size: int = None) -> Dict:
        batch = batch_size or self.batch_size
        input_ids = torch.randint(0, self.model.config.vocab_size, (batch, self.seq_len))
        return {'input_ids': input_ids}
    
    def _create_inputs(self):
        """Create inputs without SEP tokens to trigger annotation"""
        vocab_size = self.model.lm_num_tokens
        input_ids = torch.randint(0, vocab_size, (self.batch_size, self.seq_len))
        
        # Ensure no SEP tokens in first position
        input_ids[:, 0] = torch.where(
            input_ids[:, 0] == self.model.token_sep_id,
            torch.randint(1, vocab_size, (self.batch_size,)),
            input_ids[:, 0]
        )
        return input_ids
    
    def test_begin_token_generation(self):
        if FAST_TEST:
            return

        if self.model.enable_annotation:
            input_ids = self._create_inputs()
            original_sample = self.model._get_next_token
            
            def patched_sample(logits, temperature=1.0):
                new_logits = torch.zeros_like(logits).fill_(-float('inf'))
                new_logits[:, self.model.token_sep_id] = 1e9
                return original_sample(new_logits, temperature)
            
            with patch.object(self.model, '_get_next_token', new=patched_sample):
                generated_ids = self.model.generate(input_ids)
            
            begin_found = any(self.model.token_sep_id in seq.tolist() for seq in generated_ids)
            self.assertTrue(begin_found, "No SEP tokens generated in any sequence")
    
    def test_annotation_content(self):
        """Verify annotation contains only reasoning tokens"""
        if self.model.enable_annotation:
            input_ids = self._create_inputs()
            generated_ids = self.model.generate(input_ids)
            reasoning_tokens = set(self.model.anno_token_ids)
            
            for seq in generated_ids:
                seq = seq.tolist()
                if self.model.token_sep_id in seq:
                    begin_idx = seq.index(self.model.token_sep_id)
                    end_idx = seq.index(self.model.token_sep_id, begin_idx + 1)
                
                    for token in seq[begin_idx + 1:end_idx]:
                        self.assertIn(token, reasoning_tokens,
                                    f"Token {token} not in reasoning tokens")
                
                show_results(self.model.tokenizer.decode(seq))
    
    def test_skillmemory_diversity(self):
        """Ensure SkillMemory produces diverse outputs"""
        if FAST_TEST:
            return
        
        if self.model.enable_annotation:
            import torch.nn.functional as F
            from torch.distributions import Categorical
            skill = self.model.skill_memory
            
            states = torch.randn(5, 10, self.model.lm_hidden_size)
            
            with torch.no_grad():
                outputs = skill(states)
                logits = outputs['action_logits']
            
            entropy = Categorical(logits=logits).entropy()
            self.assertGreater(entropy.mean(), 1.0, "Low entropy in SkillMemory outputs")
            
            diff_states = torch.randn_like(states)
            diff_outputs = skill(diff_states)
            diff_logits = diff_outputs['action_logits']
            
            similarity = F.cosine_similarity(logits.flatten(), diff_logits.flatten(), dim=0)
            show_results(f"\nEntropy: {entropy.mean().item():.4f}\nOutput similarity: {similarity.item():.4f}")
            self.assertLess(similarity, 0.8, "Outputs not sensitive to input changes")
    
    def test_annotation_boundaries(self):
        """Verify annotation placement and length constraints"""
        if FAST_TEST:
            return
        
        if self.model.enable_annotation:
            input_ids = self._create_inputs()
            generated_ids = self.model.generate(input_ids)
            
            for seq in generated_ids:
                seq = seq.tolist()
                try:
                    begin_idx = seq.index(self.model.token_sep_id)
                    end_idx = seq.index(self.model.token_sep_id, begin_idx + 1)
                    annotation_length = end_idx - begin_idx - 1
                    show_results(f'Annotation: begin_idx={begin_idx} end_idx={end_idx}\nContext:\n{self.model.tokenizer.decode(seq)}')
                    self.assertGreaterEqual(annotation_length, 1, "Annotation too short")
                    self.assertLessEqual(annotation_length, self.model.anno_max_length + 2,
                                        "Annotation exceeds max length")
                except ValueError:
                    if self.model.token_sep_id in seq:
                        self.fail("SEP tokens must be paired")
    
    def test_annotation_token_constraints(self):
        """Verify only special tokens appear in annotations"""
        if FAST_TEST:
            return
        
        if self.model.enable_annotation:
            input_ids = self._create_inputs()
            generated_ids = self.model.generate(input_ids)
            
            allowed_tokens = set(self.model.anno_token_ids) | {
                self.model.token_sep_id
            }
            
            for seq in generated_ids:
                seq = seq.tolist()
                if self.model.token_sep_id in seq:
                    begin_idx = seq.index(self.model.token_sep_id)
                    end_idx = seq.index(self.model.token_sep_id, begin_idx + 1)
                    
                    for token in seq[begin_idx:end_idx + 1]:
                        self.assertIn(token, allowed_tokens, 
                                    f"Invalid token {token} in annotation")
    
    def test_annotation_encapsulation(self):
        """Verify all annotations are properly encapsulated"""
        if FAST_TEST:
            return
        
        if self.model.enable_annotation:
            input_ids = self._create_inputs()
            generated_ids = self.model.generate(input_ids)
            
            for seq in generated_ids:
                seq = seq.tolist()

                if self.model.token_sep_id in seq:
                    begin_idx = seq.index(self.model.token_sep_id)
                    end_idx = None
                    for i in range(begin_idx + 1, len(seq)):
                        if seq[i] == self.model.token_sep_id:
                            end_idx = i
                            break
                    
                    self.assertIsNotNone(end_idx, "SEP tokens must be paired")
    
    def test_sample_generation(self):
        if FAST_TEST:
            return
        
        input_ids = torch.randint(0, self.model.config.vocab_size, (self.batch_size, self.seq_len))
        outputs = self.model(input_ids)
        
        if self.model.enable_annotation:
            batch_size, output_seq_len, vocab_size = outputs['logits'].shape

            min_expected = self.seq_len
            max_expected = self.seq_len + self.model.anno_max_length + 2
            
            self.assertEqual(batch_size, self.batch_size)
            self.assertEqual(vocab_size, self.model.lm_num_tokens)
            self.assertTrue(min_expected <= output_seq_len <= max_expected,
                            f"Output length {output_seq_len} not in range "
                            f"[{min_expected}, {max_expected}]")
        else:
            self.assertEqual(outputs['logits'].shape, 
                            (self.batch_size, self.seq_len, self.model.lm_num_tokens))

    def test_forward_pass_dimensions(self):
        if FAST_TEST:
            return

        inputs = self._create_dummy_inputs()
        outputs = self.model(**inputs)
        
        if self.model.enable_annotation:
            batch_size, output_seq_len, vocab_size = outputs['logits'].shape

            min_expected = self.seq_len
            max_expected = self.seq_len + self.model.anno_max_length + 2
            
            self.assertEqual(batch_size, self.batch_size)
            self.assertEqual(vocab_size, self.model.lm_num_tokens)
            self.assertTrue(min_expected <= output_seq_len <= max_expected,
                            f"Output length {output_seq_len} not in range "
                            f"[{min_expected}, {max_expected}]")
        else:
            self.assertEqual(
                outputs['logits'].shape,
                (self.batch_size, self.seq_len, self.model.lm_num_tokens)
            )
    
    def test_parameter_freezing(self):
        if FAST_TEST:
            return

        for name, param in self.model.lm.named_parameters():
            self.assertTrue(param.requires_grad, f"LLM parameter {name} should be trainable")
        
        components = ['skill_memory']
        for comp in components:
            for param in getattr(self.model, comp).parameters():
                self.assertTrue(param.requires_grad, f"{comp} parameter should be trainable")
    
    def test_generation_interface(self):
        if FAST_TEST:
            return
        
        inputs = self.model.tokenizer("Hello, how are you?", return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs['input_ids'])
        self.assertEqual(outputs.shape[0], 1)

    def test_device_compatibility(self):
        if FAST_TEST:
            return

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
                    pass
                    
            finally:
                # Clean up by moving model back to CPU
                self.model.cpu()
                for param in self.params:
                    if param.grad is not None:
                        param.grad = None

if __name__ == '__main__':
    show = "--show" in sys.argv
    fast = "--fast" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in ("--show", "--fast")]
    if show:
        SHOW_RESULTS = True
    if fast:
        FAST_TEST = True
    unittest.main()
