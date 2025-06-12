import os
import torch
import torch.nn as nn
from skill import SkillMemory
from typing import Dict, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from utils import RESERVED_TOKENS, SPECIAL_TOKEN_SEP, info, cfg, get_device, get_special_token_by_index, load_peft_config

class MD(nn.Module):
    def __init__(self, 
                 config = cfg, 
                 attn: str = None):
        super().__init__()
        self.use_cache = config.use_cache
        self.adapter_config = config.adapter
        self.temperature = config.temperature
        self.attn = config.attn if attn is None else attn
        self.label_pad_token_id = config.label_pad_token_id
        self.dtype = torch.float16 if config == "16-mixed" else torch.bfloat16

        self.max_annotations = config.max_annotations
        self.num_reasoning_tokens = config.num_reasoning_tokens
        self.max_reasoning_length = config.max_reasoning_length
        
        self.lm_coef = config.lm_coef
        self.lm_dir = config.model_dir
        self.lm_freeze = config.lm_freeze
        self.lm_checkpoint = config.lm_checkpoint

        self.skill_coef = config.skill_coef
        self.skill_config = config.skill_config
        self.skill_checkpoint = config.skill_checkpoint
        self.skill_integration_strategy = config.skill_integration_strategy

        self.enable_fusion = self.skill_coef and self.skill_integration_strategy == 'fusion'
        self.enable_annotation = self.skill_coef and self.skill_integration_strategy == 'annotation'

        self._init_lm()
        self._init_skill()
        self._init_params()
        info(f"LM {self.config.model_type} (hidden_size: {self.lm_hidden_size} vocab_size: {self.config.vocab_size})")

    def _init_lm(self):
        config = AutoConfig.from_pretrained(self.lm_dir)
        config.use_cache = self.use_cache
        
        model = AutoModelForCausalLM.from_pretrained(
            self.lm_dir,
            config=config,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation=self.attn
        )
        
        self._init_tokenizer(model)
        
        if not self.lm_freeze:
            peft_config_dict = load_peft_config()
            peft_config = LoraConfig(**peft_config_dict)
            peft_model = get_peft_model(model, peft_config)
            model = peft_model.to(self.dtype)
        
        self.lm = model
        self.config = config
        self.max_length = self.config.max_length
        self.lm_hidden_size = self.config.hidden_size
        
        if self.lm_checkpoint:
            self.lm.gradient_checkpointing_enable()

    def _init_skill(self) -> nn.Module:
        """Initialize SkillMemory with LM-compatible dimensions"""
        if self.skill_integration_strategy == 'annotation':
            self.action_dim = self.num_reasoning_tokens + 1
        else:
            self.action_dim = self.skill_config.get('action_dim', self.lm_num_tokens)
            
        self.skill_config.update({
            'num_tokens': self.lm_num_tokens,
            'state_dim': self.lm_hidden_size,
            'action_dim': self.action_dim,
            'hidden_dim': self.skill_config.get('hidden_dim', self.lm_hidden_size),
            'checkpoint': self.skill_checkpoint
        })
        self.skill_memory = SkillMemory(**self.skill_config)
    
    def _init_action_projection(self):
        """Adapter between SkillMemory and LM"""
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * self.lm_hidden_size, self.lm_hidden_size),
            nn.Sigmoid()
        )
        self.state_norm = nn.LayerNorm(self.lm_hidden_size)
        self.action_norm = nn.LayerNorm(self.lm_hidden_size)
        proj_scale = self.adapter_config.get('proj_scale', 2)
        min_dim = self.adapter_config.get('min_proj_dim', self.config.hidden_size)
        proj_dim = max(self.skill_memory.action_dim * proj_scale, min_dim)

        norm_pos = self.adapter_config.get('norm_position', 'post')
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
            nn.Dropout(self.adapter_config.get('proj_dropout', 0.1))
        ]
        
        self.action_proj = nn.Sequential(*layers)
        
        gelu_gain = 1.5957696
        nn.init.xavier_normal_(self.action_proj[0].weight, gain=gelu_gain)
        nn.init.zeros_(self.action_proj[0].bias)

        nn.init.xavier_normal_(self.action_proj[-2].weight, gain=1.0)
        nn.init.zeros_(self.action_proj[-2].bias)
        
        assert self.action_proj[-2].out_features == self.config.hidden_size

    def _init_params(self):
        assert self.skill_integration_strategy in  ['fusion', 'annotation'], f"Invalid skill integration strategy: {self.skill_integration_strategy}"
       
        if self.skill_integration_strategy == 'fusion':
            self._init_action_projection()
    
        if self.lm_freeze:
            for param in self.lm.parameters():
                param.requires_grad = False
        
        trainable_params = self.get_trainable_parameters()

        for param in trainable_params:
            param.requires_grad = True
    
    def _init_tokenizer(self, model):
        """Add specialized tokens ensuring that all are newly added."""
        reasoning_tokens = [get_special_token_by_index(i) for i in range(self.num_reasoning_tokens)]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_dir)
        self.tokenizer.add_special_tokens({'additional_special_tokens': RESERVED_TOKENS})
        self.tokenizer.add_special_tokens({'additional_special_tokens': reasoning_tokens})
        self.token_sep_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN_SEP)
        self.reasoning_token_ids = self.tokenizer.convert_tokens_to_ids(reasoning_tokens)

        device = next(model.parameters()).device
        reasoning_ids_tensor = torch.tensor(self.reasoning_token_ids, device=device)
        sep_id_tensor = torch.tensor([self.token_sep_id], device=device)

        embedding_layer = model.get_input_embeddings()
        reasoning_embeddings = embedding_layer(reasoning_ids_tensor)
        sep_embed = embedding_layer(sep_id_tensor)

        self.register_buffer('reasoning_embeddings', reasoning_embeddings.detach().clone())
        self.register_buffer('reasoning_token_ids_tensor', reasoning_ids_tensor)
        self.register_buffer('sep_embed', sep_embed.detach().clone())
        self.register_buffer('sep_id_tensor', sep_id_tensor)

        self.lm_num_tokens = len(self.tokenizer)
        model.resize_token_embeddings(self.lm_num_tokens)
        
        assert self.token_sep_id not in self.reasoning_token_ids, \
            "SEP token cannot be a reasoning token"
        assert len(set(self.reasoning_token_ids)) == self.num_reasoning_tokens, \
            "Duplicate reasoning token IDs detected"
    
    def _has_sep(self, context_embeds: torch.Tensor) -> torch.Tensor:
        """Determine if SkillMemory would generate SEP token at the end of context"""
        skill_output = self.skill_memory(context_embeds)
        step_logits = skill_output['action_logits'][:, -1, :]
        
        scaled_logits = step_logits / self.temperature
        
        if self.training:
            probs = torch.softmax(scaled_logits, dim=-1)
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            sampled_indices = torch.multinomial(probs, 1)
        else:
            sampled_indices = torch.argmax(scaled_logits, dim=-1, keepdim=True)
        
        sep_index = self.action_dim - 1
        return sampled_indices.squeeze(-1) == sep_index
    
    def _get_annotation(self, context_embeds: torch.Tensor, ids: bool = True):
        """Auto-regressively generate annotation embeddings"""
        if context_embeds.dim() == 2:
            context_embeds = context_embeds.unsqueeze(0)
        batch_size, ctx_len, hidden_size = context_embeds.shape
        device = context_embeds.device
        
        annotation_embeds = []
        annotation_ids = [] if ids else None
        
        current_embeds = context_embeds
        sep_count = torch.zeros(batch_size, dtype=torch.int, device=device)
        
        sep_embed = self.sep_embed.squeeze(0)
        reasoning_embeddings = self.reasoning_embeddings
        reasoning_token_ids = self.reasoning_token_ids_tensor
        
        for step_idx in range(self.max_reasoning_length):
            skill_output = self.skill_memory(current_embeds)
            step_logits = skill_output['action_logits'][:, -1, :]
            
            if self.training:
                probs = torch.softmax(step_logits / self.temperature, dim=-1)
                sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
            else:
                sampled_indices = torch.argmax(step_logits, dim=-1)
            
            if step_idx == self.max_reasoning_length - 1:
                force_sep_mask = (sep_count == 0)
                sampled_indices = torch.where(
                    force_sep_mask,
                    torch.tensor(self.action_dim - 1, device=device),
                    sampled_indices
                )
            
            is_sep = (sampled_indices == self.action_dim - 1)
            sep_count += is_sep.int()
            
            safe_indices = sampled_indices.masked_fill(is_sep, 0)
            non_sep_embeds = reasoning_embeddings[safe_indices]
            
            next_embeds = torch.where(
                is_sep.unsqueeze(-1),
                sep_embed.expand_as(non_sep_embeds),
                non_sep_embeds
            )
            
            if ids:
                non_sep_ids = reasoning_token_ids[safe_indices]
                next_ids = torch.where(
                    is_sep,
                    torch.tensor(self.token_sep_id, device=device),
                    non_sep_ids
                ).unsqueeze(1)
                annotation_ids.append(next_ids)
            
            annotation_embeds.append(next_embeds.unsqueeze(1))
            
            if (sep_count >= 1).all():
                break
            
            current_embeds = torch.cat([current_embeds, next_embeds.unsqueeze(1)], dim=1)
        
        annotation_embeds = torch.cat(annotation_embeds, dim=1) if annotation_embeds else \
            torch.empty(batch_size, 0, hidden_size, device=device)
        
        if ids:
            annotation_ids = torch.cat(annotation_ids, dim=1) if annotation_ids else \
                torch.empty(batch_size, 0, dtype=torch.long, device=device)
        
        return annotation_embeds, annotation_ids if ids else None
    
    def _sample_next_token(self, logits, temperature = None):
        if temperature is None:
            temperature = self.temperature
        temperature = max(temperature, 1e-5)
        original_logits = logits.clone()
        
        top_k = 3
        top_logits, _ = torch.topk(logits, top_k, dim=-1)
        min_val = top_logits[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_val, torch.tensor(-1e8).to(logits.device), logits)
        
        all_masked = torch.all(logits < -1e7)
        if all_masked:
            logits = original_logits
        
        logits = torch.clamp(logits, min=-1e4, max=1e4)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
    
    def _fuse_features(self, state_embeds, action_embeds):
        """Enhanced feature fusion with learnable normalization"""
        state_embeds = state_embeds.to(self.dtype)
        action_embeds = action_embeds.to(self.dtype)
        
        with torch.amp.autocast(get_device().type, enabled=False):
            state_norm = self.state_norm(state_embeds.float()).to(self.dtype)
            action_norm = self.action_norm(action_embeds.float()).to(self.dtype)

        with torch.amp.autocast(get_device().type, dtype=self.dtype):
            combined = torch.cat([state_norm, action_norm], dim=-1)
            gate = self.fusion_gate(combined.to(self.fusion_gate[0].weight.dtype))
            return gate * state_norm + (1 - gate) * action_norm

    def annotate(
        self, 
        input_ids: torch.Tensor,
        input_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        assert self.enable_annotation, "Annotation must be enabled"
        
        input_ids = input_ids.clamp(0, self.lm_num_tokens - 1)
        state_embeds = self.lm.get_input_embeddings()(input_ids)
        batch_size, seq_len, hidden_size = state_embeds.shape
        
        sep_detected = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        total_seps = 0
        
        for pos in range(1, seq_len):
            context_embeds = state_embeds[:, :pos]
            sep = self._has_sep(context_embeds)
            sep_detected[:, pos] = sep
            
            batch_seps = sep.sum().item()
            total_seps += batch_seps
            
            if self.max_annotations > 0 and total_seps >= self.max_annotations:
                break
        
        all_embeds = []
        all_labels = []
        annotation_cache = {}
        
        sep_embed = self.sep_embed.unsqueeze(0)
        
        for i in range(batch_size):
            pos_indices = sep_detected[i].nonzero().squeeze(-1).tolist()
            
            current_embeds = [state_embeds[i]]
            current_labels = [input_labels[i]]
            
            for pos in reversed(pos_indices):
                context_key = tuple(input_ids[i, :pos+1].tolist())
                
                if context_key not in annotation_cache:
                    context = torch.cat([state_embeds[i, :pos+1], sep_embed.squeeze(0)], dim=0)
                    annotation, _ = self._get_annotation(context.unsqueeze(0))
                    
                    if annotation.size(1) > 0:
                        full_annotation = torch.cat([sep_embed.squeeze(0), annotation.squeeze(0)])
                        annotation_cache[context_key] = full_annotation
                
                if context_key in annotation_cache:
                    annotation = annotation_cache[context_key]
                    current_embeds.insert(pos+1, annotation)
                    current_labels.insert(pos+1, 
                        torch.full_like(annotation[:, 0], self.label_pad_token_id))
            
            all_embeds.append(torch.cat(current_embeds))
            all_labels.append(torch.cat(current_labels))
        
        max_len = max(e.size(0) for e in all_embeds)
        full_embeds = torch.zeros(batch_size, max_len, hidden_size, 
                                  device=input_ids.device)
        full_labels = torch.full((batch_size, max_len), self.label_pad_token_id,
                                 dtype=torch.long, device=input_ids.device)
        
        for i, (embeds, labels) in enumerate(zip(all_embeds, all_labels)):
            full_embeds[i, :embeds.size(0)] = embeds
            full_labels[i, :labels.size(0)] = labels
        
        lm_out = self.lm(inputs_embeds=full_embeds.to(self.dtype))
        
        return {
            'labels': full_labels,
            'states': full_embeds,
            'logits': lm_out.logits
        }

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if torch.any(input_ids >= self.lm_num_tokens):
            input_ids = input_ids.clamp(0, self.lm_num_tokens - 1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        
        state_embeds = self.lm.get_input_embeddings()(input_ids)
        batch_size, seq_len, hidden_size = state_embeds.shape
        
        if self.enable_annotation:
            sep_detected = self._has_sep(state_embeds)
            
            new_embeds = []
            new_labels = []
            
            for i in range(batch_size):
                seq_embeds = state_embeds[i]
                seq_input_ids = input_ids[i]
                new_embeds_list = [seq_embeds]
                new_labels_list = [seq_input_ids]
                
                if sep_detected[i].item():
                    seq_embeds = torch.cat([seq_embeds, self.sep_embed])
                    annotation_embeds, _ = self._get_annotation(seq_embeds)
                    
                    if annotation_embeds.size(1) > 1:
                        annotation_embeds = torch.cat([self.sep_embed, annotation_embeds[0]])
                        new_embeds_list.append(annotation_embeds)
                        new_labels_list.append(torch.full((annotation_embeds.size(0),), self.label_pad_token_id, 
                                                          device=input_ids.device))
                
                new_embeds.append(torch.cat(new_embeds_list))
                new_labels.append(torch.cat(new_labels_list))
            
            max_len = max(tensor.size(0) for tensor in new_embeds)
            full_embeds = torch.zeros(batch_size, max_len, hidden_size, 
                                      device=state_embeds.device)
            full_labels = torch.full((batch_size, max_len), self.label_pad_token_id, 
                                     dtype=torch.long, device=input_ids.device)
            full_mask = torch.zeros(batch_size, max_len, 
                                    dtype=attention_mask.dtype, device=attention_mask.device)
            
            for i in range(batch_size):
                seq_len_i = new_embeds[i].size(0)
                full_embeds[i, :seq_len_i] = new_embeds[i]
                full_labels[i, :seq_len_i] = new_labels[i]
                full_mask[i, :seq_len_i] = 1

            lm_out = self.lm(inputs_embeds=full_embeds.to(self.dtype))
        
        else:
            if self.enable_fusion:
                with torch.amp.autocast(get_device().type, enabled=False):
                    skill_output = self.skill_memory(state_embeds)
                
                action_logits = skill_output['action_logits']
                action_embeds = self.action_proj(action_logits)
                state_embeds = self._fuse_features(state_embeds, action_embeds)
            
            lm_out = self.lm(inputs_embeds=state_embeds.to(self.dtype))
        
        return {
            'states': state_embeds,
            'logits': lm_out.logits
        }

    def _generate(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor: 
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        sequences = [input_ids[i].clone() for i in range(batch_size)]
        eos_flags = [False] * batch_size
        active_indices = list(range(batch_size))

        for step in range(self.max_length - input_ids.shape[1]):
            if not active_indices:
                break
            
            for idx in list(active_indices):
                if eos_flags[idx]:
                    active_indices.remove(idx)
                    continue
                
                context = sequences[idx].unsqueeze(0)
                context_embeds = self.lm.get_input_embeddings()(context)
                
                if self.enable_annotation and self._has_sep(context_embeds).item():
                    sep_embed = self.sep_embed.unsqueeze(0)
                    embeds = torch.cat([context_embeds, sep_embed], dim=1)
                    annotation_embeds, annotation_ids = self._get_annotation(embeds, ids=True)

                    if annotation_embeds.size(1) > 1:
                        sequences[idx] = torch.cat([sequences[idx], self.sep_id_tensor])
                        sequences[idx] = torch.cat([sequences[idx], annotation_ids.squeeze(0)])
                        context_embeds = torch.cat([embeds, annotation_embeds], dim=1)
                
                else:
                    with torch.no_grad():
                        if self.enable_fusion:
                            skill_output = self.skill_memory(context_embeds)
                            action_logits = skill_output['action_logits']
                            action_embeds = self.action_proj(action_logits)
                            context_embeds = self._fuse_features(context_embeds, action_embeds)
                        
                lm_out = self.lm(inputs_embeds=context_embeds.to(self.dtype))
                next_token = self._sample_next_token(lm_out.logits[:, -1, :])[0].item()
                sequences[idx] = torch.cat([sequences[idx], torch.tensor([next_token], device=device)])
                
                if next_token == self.tokenizer.eos_token_id:
                    eos_flags[idx] = True
                    active_indices.remove(idx)
        
        return torch.nn.utils.rnn.pad_sequence(
            sequences, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
    
    def generate(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.skill_coef:
            return self._generate(input_ids)
        else:
            return self.lm.generate(input_ids=input_ids)
    
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

    def get_trainable_parameters(self):
        trainable_params = []

        if self.enable_fusion:
            trainable_params.extend([
                *self.action_proj.parameters(),
                *self.fusion_gate.parameters(),
                *self.state_norm.parameters(),
                *self.action_norm.parameters()
            ])
        
        trainable_params.extend(self.skill_memory.parameters())
        
        if not self.lm_freeze:
            trainable_params.extend(self.lm.parameters())
        
        return trainable_params
