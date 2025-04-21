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

        # Load pretrained model configuration first
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
        self.config = AutoConfig.from_pretrained(pretrained_model_dir)
        self.lm_hidden_size = self.config.hidden_size
        self.lm_num_tokens = self.config.vocab_size
        self.lm_dir = pretrained_model_dir
        self.config.use_sdpa = cfg.use_sdpa
        info(f"Base LM (name: {cfg.lm_name} hidden_size: {self.lm_hidden_size} vocab_size: {self.config.vocab_size})")
     
        # Initialize SkillMemory with compatible dimensions
        self.skill_memory = self._init_skill_memory()

        # Load pretrained model weights
        self.lm = self._init_lm()
        
        # Configure model components
        self._init_action_projection()

        if freeze_pretrained:
            self._freeze_pretrained()

    def _init_lm(self):
        return AutoModelForCausalLM.from_pretrained(
            self.lm_dir,
            config=self.config,
            trust_remote_code=True
        )

    def _init_skill_memory(self) -> nn.Module:
        """Initialize SkillMemory with LM-compatible dimensions"""
        skill_params = cfg.skill
        skill_params.update({
            'num_tokens': self.lm_num_tokens,
            'hidden_dim': self.lm_hidden_size,
            'action_dim': self.lm_hidden_size
        })
        return SkillMemory(**skill_params)

    def _init_action_projection(self):
        """Adapter between SkillMemory and LM"""
        self.action_proj = nn.Linear(self.skill_memory.action_dim, self.config.hidden_size)

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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
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
        
        return {
            'states': state_embeds,
            'logits': lm_out.logits,
            'action_logits': action_logits
        }

    def _create_position_ids(self, input_ids, action_embeds):
        """Create position IDs accounting for action embeddings"""
        original_len = input_ids.shape[1]
        action_len = action_embeds.shape[1]
        batch_size = input_ids.size(0)
        position_ids = torch.cat([
            torch.arange(original_len, device=input_ids.device),
            torch.arange(cfg.suffix_start, cfg.suffix_start + action_len, device=input_ids.device)
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

    def generate(
        self,
        input_ids: torch.Tensor,
        **generation_kwargs
    ) -> torch.Tensor:
        with torch.no_grad():
            attention_mask = generation_kwargs.get('attention_mask')
            state_embeds = self.lm.get_input_embeddings()(input_ids)
        
            # Get memory context
            action_logits = self.skill_memory.generate(state_embeds)
            action_embeds = self.action_proj(action_logits)
            
            # Prepare inputs
            input_embeds = self._combine_inputs(state_embeds, action_embeds)
            position_ids = self._create_position_ids(input_ids, action_embeds)
            attention_mask = self._create_attention_mask(input_ids, action_embeds, attention_mask)
            generation_kwargs['attention_mask'] = attention_mask
            
            # Generate with adjusted mask
            return self.lm.generate(
                inputs_embeds=input_embeds,
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
        return {
            'skill_memory': list(self.skill_memory.parameters()),
            'action_proj': list(self.action_proj.parameters())
        }
