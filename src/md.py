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
        self.attn = config.attn if attn is None else attn
        self.skill_memory = self._init_skill_memory()

        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * self.lm_hidden_size, self.lm_hidden_size),
            nn.Sigmoid()
        )
        self.state_norm = nn.LayerNorm(self.lm_hidden_size)
        self.action_norm = nn.LayerNorm(self.lm_hidden_size)

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
            'state_dim': self.lm_hidden_size,
            'action_dim': self.skill_config.get('action_dim', self.lm_num_tokens),
            'hidden_dim': self.skill_config.get('hidden_dim', self.lm_hidden_size)
        })
        return SkillMemory(**self.skill_config)
    
    def _init_action_projection(self):
        """Adapter between SkillMemory and LM"""
        proj_scale = self.adapter.get('proj_scale', 2)
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
        for param in self.fusion_gate.parameters():
            param.requires_grad = True
        for param in [self.state_norm, self.action_norm]:
            param.requires_grad = True

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        device = input_ids.device
        
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).int().to(device)
        
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if self.skill_coef:
            state_embeds = self.lm.get_input_embeddings()(input_ids)
            action_logits = self.skill_memory.generate(state_embeds)
            action_embeds = self.action_proj(action_logits)
            fused_embeds = self._fuse_features(state_embeds, action_embeds)
            
            lm_out = self.lm(
                inputs_embeds=fused_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
        else:
            state_embeds = None
            action_logits = None
            lm_out = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        return {
            'states': state_embeds,
            'action_logits': action_logits,
            'logits': lm_out.logits
        }

    def _fuse_features(self, state_embeds, action_embeds):
        """Enhanced feature fusion with learnable normalization"""
        state_norm = self.state_norm(state_embeds)
        action_norm = self.action_norm(action_embeds)
        
        combined = torch.cat([state_norm, action_norm], dim=-1)
        gate = self.fusion_gate(combined)
        return gate * state_norm + (1 - gate) * action_norm

    def _generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = input_ids.device
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        eos_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        start_len = input_ids.shape[1]
        total_steps = self.max_length - start_len
        step = 0
        
        if attention_mask is None:
            current_attention_mask = (input_ids != self.tokenizer.pad_token_id).int().to(device)
        else:
            current_attention_mask = attention_mask.to(device)
        
        position_ids = torch.arange(start_len, device=device).expand(batch_size, -1)
        
        while step < total_steps:
            current_length = generated_ids.shape[1]
            if eos_flags.all():
                break
            
            if current_attention_mask.shape[1] < current_length:
                new_mask = torch.ones(
                    batch_size, 
                    current_length - current_attention_mask.shape[1],
                    device=device,
                    dtype=current_attention_mask.dtype
                )
                current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)
            
            if position_ids.shape[1] < current_length:
                new_positions = torch.arange(
                    position_ids.shape[1], 
                    current_length,
                    device=device
                ).expand(batch_size, -1)
                position_ids = torch.cat([position_ids, new_positions], dim=1)
            
            current_state_embeds = self.lm.get_input_embeddings()(generated_ids)
            
            if self.skill_coef:
                current_action_logits = self.skill_memory.generate(current_state_embeds)
                current_action_embeds = self.action_proj(current_action_logits)
                fused_embeds = self._fuse_features(current_state_embeds, current_action_embeds)
            else:
                fused_embeds = current_state_embeds
            
            with torch.no_grad():
                lm_out = self.lm(
                    inputs_embeds=fused_embeds,
                    attention_mask=current_attention_mask,
                    position_ids=position_ids
                )
            
            next_token_logits = lm_out.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            next_tokens = next_tokens.masked_fill(eos_flags, self.tokenizer.eos_token_id)
            eos_flags = eos_flags | (next_tokens == self.tokenizer.eos_token_id)
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            step += 1
            
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
        
        model_state = state_dict.get('model', state_dict)
        model_state = model_state.get('state_dict', model_state)
        
        model_keys = set(dict(model.named_parameters()).keys())
        filtered_state = {k: v for k, v in model_state.items() if k in model_keys}
        
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        info(f"Loaded pre-trained model from {checkpoint_path}")
        info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        return model.to(get_device())

    def get_trainable_parameters(self) -> Dict[str, nn.Parameter]:
        return {
            'skill_memory': list(self.skill_memory.parameters()),
            'action_proj': list(self.action_proj.parameters()),
            'fusion_gate': list(self.fusion_gate.parameters()),
            'normalization': list(self.state_norm.parameters()) + list(self.action_norm.parameters())
        }
