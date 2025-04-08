import torch
import torch.nn as nn
from pathlib import Path
from utils import info, cfg
from skill import SkillMemory
from typing import Dict, Optional
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

class MD(nn.Module):
    def __init__(
        self,
        pretrained_model_dir: Path = cfg.model_dir,
        freeze_pretrained: bool = True
    ):
        super().__init__()

        # 1. Load pretrained model configuration first
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
        self.lm_config = AutoConfig.from_pretrained(pretrained_model_dir)
        self.lm_hidden_size = self.lm_config.hidden_size
        self.lm_vocab_size = self.lm_config.vocab_size
        self.lm_config.use_sdpa = cfg.use_sdpa
        self.state_embed_dim = self.lm_hidden_size
        info(f"Base LLM (name: {cfg.model_name} hidden_size: {self.lm_hidden_size} vocab_size: {self.lm_vocab_size})")
     
        # 2. Initialize SkillMemory with compatible dimensions
        self.skill_memory = self._init_skill_memory()

        # 3. Load pretrained model weights
        self.lm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_dir,
            config=self.lm_config
        )
        
        # 4. Configure model components
        self._init_action_projection()
        
        if freeze_pretrained:
            self._freeze_pretrained()

    def _init_skill_memory(self) -> nn.Module:
        """Initialize SkillMemory with LM-compatible dimensions"""
        skill_params = cfg.skill_memory.copy()
        skill_params.update({
            'action_dim': self.lm_hidden_size,
            'state_embed_dim': self.lm_hidden_size,
            'hidden_dim': self.lm_hidden_size
        })
        return SkillMemory(**skill_params)

    def _init_action_projection(self):
        """Adapter between SkillMemory and LM"""
        self.action_proj = nn.Linear(self.skill_memory.action_dim, self.lm_config.hidden_size)

    def _freeze_pretrained(self):
        """Freeze LM parameters while keeping adapters trainable"""
        for param in self.lm.parameters():
            param.requires_grad = False
        for param in self.action_proj.parameters():
            param.requires_grad = True
        for param in self.skill_memory.parameters():
            param.requires_grad = True

    def forward(
        self,
        states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        seq_len = input_ids.shape[1]
        if states.shape[1] != seq_len:
            raise ValueError(
                f"States length {states.shape[1]} "
                f"must match input_ids length {seq_len}"
            )
        
        # 1. Process through SkillMemory
        action_logits, m = self.skill_memory.generate(states)
        
        # 2. Adapt actions
        action_embeds = self.action_proj(action_logits)
        
        # 3. Combine with text inputs
        inputs_embeds = self._combine_inputs(input_ids, action_embeds)
        position_ids = self._create_position_ids(input_ids, action_embeds)
        attention_mask = self._create_attention_mask(input_ids, action_embeds, attention_mask)
        
        # 4. Forward through LM
        lm_out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True
        )
        
        return {
            'skill_memory': m,
            'lm_logits': lm_out.logits,
            'action_logits': action_embeds
        }

    def _create_position_ids(self, input_ids, action_embeds):
        """Create position IDs accounting for action embeddings"""
        original_len = input_ids.shape[1]
        action_len = action_embeds.shape[1]
        batch_size = input_ids.size(0)
        position_ids = torch.cat([
            torch.arange(original_len, device=input_ids.device),
            torch.arange(cfg.action_start, cfg.action_start + action_len, device=input_ids.device)
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
        
    def _combine_inputs(
        self,
        input_ids: torch.Tensor,
        actioin_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Fuse memory context with text inputs using position-aware encoding"""
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        return torch.cat([text_embeds, actioin_embeds], dim=1)

    def generate(
        self,
        states: torch.Tensor,
        input_ids: torch.Tensor,
        **generation_kwargs
    ) -> torch.Tensor:
        with torch.no_grad():
            attention_mask = generation_kwargs.get('attention_mask')
        
            # 1. Get memory context
            action_logits, _ = self.skill_memory.generate(states)
            action_embeds = self.action_proj(action_logits)
            
            # 2. Prepare inputs
            inputs_embeds = self._combine_inputs(input_ids, action_embeds)
            position_ids = self._create_position_ids(input_ids, action_embeds)
            attention_mask = self._create_attention_mask(input_ids, action_embeds, attention_mask)
            generation_kwargs['attention_mask'] = attention_mask
            
            # 3. Generate with adjusted mask
            return self.lm.generate(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                **generation_kwargs
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_dir: Path = cfg.model_dir,
        **kwargs
    ) -> 'MD':
        return cls(
            pretrained_model_dir=pretrained_model_dir,
            **kwargs
        )

    def get_trainable_parameters(self) -> Dict[str, nn.Parameter]:
        """Utility method for training"""
        return {
            'skill_memory': list(self.skill_memory.parameters()),
            'action_proj': list(self.action_proj.parameters())
        }
