import os
import torch
import torch.nn as nn
from skill import SkillMemory
from typing import Dict, Optional
from utils import info, cfg, get_device
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

class MD(nn.Module):
    def __init__(self, 
                 config = cfg, 
                 attn: str = None):
        super().__init__()
        self.lm_dir = config.model_dir
        self.lm_coef = config.lm_coef
        self.adapter = config.adapter
        self.skill_coef = config.skill_coef
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_dir)
        self.config = AutoConfig.from_pretrained(self.lm_dir)
        self.checkpoint_pretrained = config.checkpoint_pretrained
        self.lm_hidden_size = self.config.hidden_size
        self.lm_num_tokens = self.config.vocab_size
        self.max_length = self.config.max_length
        self.config.use_cache = config.use_cache
        self.skill_config = config.skill_config
        self.suffix_start = config.suffix_start
        self.attn = config.attn if attn is None else attn 
        self.skill_memory = self._init_skill_memory()
        self.lm = self._init_lm()
        self._init_action_projection()
        self._init_params(freeze_pretrained=config.freeze_pretrained)
        info(f"LM {self.config.model_type} (hidden_size: {self.lm_hidden_size} vocab_size: {self.config.vocab_size})")

    def _init_lm(self):
        lm = AutoModelForCausalLM.from_pretrained(
            self.lm_dir,
            torch_dtype='auto',
            config=self.config,
            trust_remote_code=True,
            attn_implementation=self.attn
        )
        if self.checkpoint_pretrained:
            lm.gradient_checkpointing_enable()
        return lm

    def _init_skill_memory(self) -> nn.Module:
        """Initialize SkillMemory with LM-compatible dimensions"""
        self.skill_config.update({
            'num_tokens': self.lm_num_tokens,
            'hidden_dim': self.lm_hidden_size,
            'action_dim': self.skill_config.get('action_dim', self.lm_hidden_size)
        })
        return SkillMemory(**self.skill_config)
    
    def _init_action_projection(self):
        """Adapter between SkillMemory and LM"""
        proj_scale = self.adapter.get('proj_scale', 4)
        min_dim = self.adapter.get('min_proj_dim', self.config.hidden_size)
        proj_dim = max(self.skill_memory.action_dim * proj_scale, min_dim)

        norm_pos = self.adapter.get('norm_position', 'post')
        assert norm_pos in ['pre', 'post'], f"Invalid norm_position: {norm_pos}"
        
        layers = [
            nn.Linear(self.skill_memory.action_dim, proj_dim)
        ]
        
        if norm_pos == 'pre':
            layers += [nn.LayerNorm(proj_dim), nn.GELU()]
        else:
            layers += [nn.GELU(), nn.LayerNorm(proj_dim)]
        
        layers += [
            nn.Linear(proj_dim, self.config.hidden_size),
            nn.Dropout(self.adapter.get('proj_dropout', 0.1))
        ]
        
        self.action_proj = nn.Sequential(*layers)
        
        gelu_gain = 1.5957696
        nn.init.xavier_normal_(self.action_proj[0].weight, gain=gelu_gain)
        nn.init.zeros_(self.action_proj[0].bias)

        nn.init.xavier_normal_(self.action_proj[-2].weight, gain=1.0)
        nn.init.zeros_(self.action_proj[-2].bias)
        
        assert self.action_proj[-2].out_features == self.config.hidden_size

    def _init_params(self, freeze_pretrained):
        """Freeze LM parameters while keeping adapters trainable"""
        lm_requires_grad = not freeze_pretrained
        for param in self.lm.parameters():
            param.requires_grad = lm_requires_grad
        for param in self.action_proj.parameters():
            param.requires_grad = True
        for param in self.skill_memory.parameters():
            param.requires_grad = True

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.skill_coef:
            state_embeds = self.lm.get_input_embeddings()(input_ids)
            
            # Process through SkillMemory
            action_logits = self.skill_memory.generate(state_embeds)
            
            # Adapt actions
            action_embeds = self.action_proj(action_logits)
            
            # Combine with text inputs
            inputs_embeds = self._combine_inputs(state_embeds, action_embeds)
            position_ids = self._create_position_ids(input_ids, action_embeds)
            attention_mask = self._create_attention_mask(input_ids, action_embeds, attention_mask)
        
            # Forward through LM
            lm_out = self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
        else:
            state_embeds = None
            action_logits = None
            lm_out = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        return {
            'states': state_embeds,
            'action_logits': action_logits,
            'logits': lm_out.logits
        }

    def _create_position_ids(self, input_ids, action_embeds):
        """Create position IDs accounting for action embeddings"""
        original_len = input_ids.shape[1]
        action_len = action_embeds.shape[1]
        batch_size = input_ids.size(0)
        position_ids = torch.cat([
            torch.arange(original_len, device=input_ids.device),
            torch.arange(self.suffix_start, self.suffix_start + action_len, device=input_ids.device)
        ]).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        return position_ids
    
    def _create_attention_mask(self, input_ids, action_embeds, attention_mask=None):
        attention_mask = (
            (input_ids != self.tokenizer.pad_token_id).int()
            if attention_mask is None
            else attention_mask.to(input_ids.device)
        )
        batch_size = attention_mask.size(0)
        ones_tensor = torch.ones(batch_size, action_embeds.shape[1], device=input_ids.device)
        combined_mask = torch.cat([attention_mask, ones_tensor], dim=1)
        return combined_mask
    
    def _combine_inputs(self,
        state_embeds: torch.Tensor,
        action_embeds: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([state_embeds, action_embeds], dim=1)

    def _generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        device = input_ids.device
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        eos_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.max_length - input_ids.shape[1]):
            if eos_flags.all():
                break
            
            model_inputs = {
                'input_ids': generated_ids,
            }

            if attention_mask is not None:
                current_length = generated_ids.shape[1]
                extended_attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, current_length - attention_mask.shape[1]), device=device)
                ], dim=1)
                model_inputs['attention_mask'] = extended_attention_mask

            with torch.no_grad():
                outputs = self.forward(**model_inputs)
            
            next_token_logits = outputs['logits'][:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            next_tokens = next_tokens.masked_fill(eos_flags, self.tokenizer.eos_token_id)
            eos_flags = eos_flags | (next_tokens == self.tokenizer.eos_token_id)
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)

        return generated_ids

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **generation_kwargs
    ) -> torch.Tensor:
        if self.skill_coef:
            return self._generate(input_ids, attention_mask)
        else:
            return self.lm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str = cfg.ckpt_path,
        config = cfg,
        attn: str = None,
        **kwargs
    ) -> 'MD':
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")
        model = cls(config=config, attn=attn, **kwargs)
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {str(e)}") from e
        if 'model' in state_dict:
            state_dict = state_dict['model']
        with torch.device('cpu'):
            model.load_state_dict(state_dict, strict=False)
        info(f"Loaded pre-trained model from {checkpoint_path}")
        return model.to(get_device())

    def get_trainable_parameters(self) -> Dict[str, nn.Parameter]:
        return {
            'skill_memory': list(self.skill_memory.parameters()),
            'action_proj': list(self.action_proj.parameters())
        }
