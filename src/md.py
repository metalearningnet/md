import os
import math
import torch
import torch.nn as nn
from frontend import Frontend
from skill import SkillMemory
import torch.nn.functional as F
from typing import Dict, Optional
from huggingface_hub import model_info
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from utils import (
    SEP_TOKEN, RESERVED_TOKENS, BOUNDARY_TOKENS, LABEL_PAD_TOKEN_ID,
    LogitsDecoder, get_device, load_peft_config, get_special_token_by_index, info, warn, cfg
)

class SafeEmbeddingWrapper(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
    
    def forward(self, input):
        # Ensure no in-place modifications
        input = input.detach().clone()
        return self.embedding_layer(input)

class AdaptedEmbedding(nn.Module):
    def __init__(self, original_embedding, peft_config):
        super().__init__()
        self.original = original_embedding
        vocab_size, embedding_dim = original_embedding.weight.shape
        
        self.lora_A = nn.Parameter(
            torch.empty(peft_config.r, vocab_size)
        )
        self.lora_B = nn.Parameter(
            torch.empty(embedding_dim, peft_config.r)
        )
        
        if getattr(peft_config, 'init_lora_weights', 'gaussian') == 'gaussian':
            nn.init.normal_(self.lora_A, std=1/peft_config.r)
            nn.init.zeros_(self.lora_B)
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        
        self.peft_config = peft_config
        self.scaling = peft_config.lora_alpha / peft_config.r
    
    def forward(self, input):
        orig_out = self.original(input)
        one_hot = F.one_hot(input, num_classes=self.original.num_embeddings).float()
        lora_out = (one_hot @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return orig_out + lora_out

class MD(nn.Module):
    def __init__(self,
                 config = cfg,
                 attn: str = None,
                 dist: bool = False):
        super().__init__()
        
        self.noise_scale = 0.15
        self.noise_floor = 0.05
        self.min_sep_bias = 1.0
        self.anno_temperature = 0.9
        self.anno_noise_scale = 0.05
        self.density_strength = 2.0
        self.diversity_strength = 0.8
        self.bias_growth_power = 1.2
        self.noise_decay_power = 0.8

        self.max_steps = 0
        self.current_step = 0
        self.label_pad_token_id = LABEL_PAD_TOKEN_ID

        self.dist = dist
        self.device = get_device()
        self.lm_dir = config.lm_dir
        self.use_cache = config.use_cache
        self.max_length = config.max_length
        self.has_mem = config.mem_coef != 0.0
        self.has_frontend = config.has_frontend
        self.attn = config.attn if attn is None else attn
        self.dtype = torch.float16 if config.precision == '16-mixed' else torch.bfloat16
        
        self._init_lm(config)
        self._init_mem(config)
        self._init_params()

        info(f"MD (hidden_dim: {self.hidden_size}, mem: {config.mem_info}, special_tokens: {self.num_special_tokens}, preference_optimizer: {config.po})")

    def _init_backend_fusion_adapter(self):
        self.backend_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )
        self.backend_state_norm = nn.LayerNorm(self.hidden_size)
        self.backend_action_norm = nn.LayerNorm(self.hidden_size)
        min_dim = self.backend_fusion_adapter_config.get('min_proj_dim', -1)
        proj_scale = self.backend_fusion_adapter_config.get('proj_scale', -1)
        proj_dim = max(self.mem_backend.action_dim * proj_scale, min_dim, self.hidden_size)

        norm_pos = self.backend_fusion_adapter_config.get('norm_position', 'post')
        assert norm_pos in ['pre', 'post'], f"Invalid norm_position: {norm_pos}"
        
        layers = [
            nn.Linear(self.mem_backend.action_dim, proj_dim)
        ]
        
        if norm_pos == 'pre':
            layers += [nn.LayerNorm(proj_dim), nn.GELU()]
        else:
            layers += [nn.GELU(), nn.LayerNorm(proj_dim)]
        
        layers += [
            nn.Linear(proj_dim, self.hidden_size),
            nn.Dropout(self.backend_fusion_adapter_config.get('proj_dropout', 0.1))
        ]
        
        self.backend_action_proj = nn.Sequential(*layers)
        
        nn.init.xavier_normal_(self.backend_action_proj[0].weight, gain=1.5957696)
        nn.init.zeros_(self.backend_action_proj[0].bias)

        nn.init.xavier_normal_(self.backend_action_proj[-2].weight, gain=1.0)
        nn.init.zeros_(self.backend_action_proj[-2].bias)
        
        assert self.backend_action_proj[-2].out_features == self.hidden_size
    
    def _init_mem_backend(self, config):
        self.backend_sep_logit_bias = self.min_sep_bias
        self.backend_sep_noise_scale = self.noise_scale
        self.backend_mem_type = config.backend_mem_type
        self.backend_max_hints = config.backend_max_hints
        self.backend_checkpoint = config.backend_checkpoint
        self.backend_mem_config = config.backend_mem_config
        self.backend_gen_sep = self.backend_enable_annotation
        self.backend_min_interval = config.backend_min_interval
        self.backend_skill_config = config.backend_skill_config
        self.backend_max_sep_bias = config.backend_sep_logit_bias
        self.backend_update_memory = config.backend_update_memory
        self.backend_context_window = config.backend_context_window
        self.backend_sep_temperature = config.backend_sep_temperature
        self.backend_max_annotations = config.backend_max_annotations
        self.backend_anno_max_length = config.backend_anno_max_length
        self.backend_non_sep_temp = self.backend_sep_temperature * 1.5
        self.backend_fusion_adapter_config = config.backend_fusion_adapter
        self.backend_sentence_alignment = config.backend_sentence_alignment
        self.backend_anno = self.backend_enable_hint or self.backend_enable_annotation
        self.backend_tune_special_token_embeddings = config.backend_tune_special_token_embeddings
        
        if self.backend_enable_annotation:
            self.backend_max_annos = self.backend_max_annotations
        else:
            self.backend_max_annos = self.backend_max_hints

        assert self.backend_strategy in ['fusion', 'annotation', 'hint'], f"Invalid skill integration strategy: {self.backend_strategy}"
        
        if self.backend_strategy in ['hint', 'annotation']:
            self.backend_action_dim = self.backend_special_tokens + 1
        else:
            self.backend_action_dim = self.num_tokens
        
        if self.backend_strategy == 'fusion':
            self.mem_state_dim = self.hidden_size
        else:
            self.mem_state_dim = self.backend_skill_config.get('state_dim', self.hidden_size)
        
        self.backend_hidden_dim = self.backend_skill_config.get('hidden_dim', self.hidden_size)

        self.mem_state_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.mem_state_dim*2),
            nn.GELU(),
            nn.Linear(self.mem_state_dim*2, self.mem_state_dim),
            nn.LayerNorm(self.mem_state_dim)
        )

        self.backend_skill_config.update({
            'num_tokens': self.num_tokens,
            'mem_type': self.backend_mem_type,
            'state_dim': self.mem_state_dim,
            'action_dim': self.backend_action_dim,
            'hidden_dim': self.backend_hidden_dim,
            'checkpoint': self.backend_checkpoint,
            'mem_config': self.backend_mem_config,
            'update_memory': self.backend_update_memory,
        })

        self.mem_backend = SkillMemory(**self.backend_skill_config)

        if self.backend_strategy == 'fusion':
            self._init_backend_fusion_adapter()
    
    def _init_mem_frontend(self, config):
        self.frontend_mem_config = config.frontend_mem_config
        self.frontend_mem_type = config.frontend_mem_type
        self.frontend_update_memory = config.frontend_update_memory
        self.frontend_config = {
            'num_tokens': self.num_tokens,
            'mem_type': self.frontend_mem_type,
            'state_dim': self.mem_state_dim,
            'mem_config': self.frontend_mem_config,
            'update_memory': self.frontend_update_memory,
        }
        self.mem_frontend = Frontend(**self.frontend_config)

    def _init_mem(self, config):
        self.mem_coef = config.mem_coef
        self._init_mem_backend(config)
        if self.has_frontend:
            self._init_mem_frontend(config)

    def _init_peft(self, model):
        embedding_layer = model.model.embed_tokens
        base_config = load_peft_config()
        target_modules = list(set(base_config.get('target_modules', []) + ['embed_tokens']))
        modules_to_save = list(set(base_config.get('modules_to_save', []) + ['embed_tokens']))
        
        info(f'target_modules: {", ".join(target_modules)}')
        info(f'modules_to_save: {", ".join(modules_to_save)}')

        peft_config = LoraConfig(
            r=base_config.get('r', 8),
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            bias=base_config.get('bias', 'none'),
            use_dora=base_config.get('use_dora', False),
            lora_alpha=base_config.get('lora_alpha', 16),
            use_rslora=base_config.get('use_rslora', True),
            lora_dropout=base_config.get('lora_dropout', 0.1),
            task_type=base_config.get('task_type', 'CAUSAL_LM'),
            fan_in_fan_out=base_config.get('fan_in_fan_out', False),
            init_lora_weights=base_config.get('init_lora_weights', 'gaussian')
        )
        
        try:
            return get_peft_model(model, peft_config)
        except Exception as e:
            try:
                warn(f"Using adapted embeddings fallback: {str(e)}")
                model.model.embed_tokens = AdaptedEmbedding(embedding_layer, peft_config)
                return model
            except Exception as e:
                warn(f"Using minimal trainable embeddings: {str(e)}")
                for param in model.parameters():
                    param.requires_grad = False
                self._make_special_token_embeddings_trainable(model)
                return model

    def _init_lm(self, config):
        self.lm_path = config.lm_path
        self.lm_coef = config.lm_coef
        self.lm_freeze = config.lm_freeze
        self.lm_checkpoint = config.lm_checkpoint
        self.lm_temperature = config.lm_temperature
    
        if not os.path.exists(self.lm_dir):
            raise FileNotFoundError(f"Language model path {self.lm_dir} does not exist")
        
        lm_config = AutoConfig.from_pretrained(self.lm_dir)
        lm_config.use_cache = self.use_cache

        info = model_info(self.lm_path)
        model_type = info.config['model_type']
        if lm_config.model_type != model_type:
            raise ValueError(f"Expected model type {model_type}, got {lm_config.model_type}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.lm_dir,
            config=lm_config,
            trust_remote_code=True,
            attn_implementation=self.attn
        )
        
        self.lm_config = lm_config
        self._init_tokenizer(model, config)
        
        if self.lm_checkpoint:
            model.gradient_checkpointing_enable()
        
        if not self.lm_freeze:
            model = self._init_peft(model)
        
        self.lm = model
        self.hidden_size = getattr(
            self.lm_config,
            'hidden_size',
            getattr(
                getattr(self.lm_config, 'text_config', None),
                'hidden_size',
                None
            )
        )

        if self.hidden_size is None:
            raise ValueError("Could not find hidden_size")
        
        self.logits_decoder = LogitsDecoder(self.lm_config, self.tokenizer, self.lm_temperature)
    
    def _make_special_token_embeddings_trainable(self, model):
        if self.backend_tune_special_token_embeddings:
            embedding_layer = model.get_input_embeddings()
            embedding_layer.weight.requires_grad_(False)
            
            with torch.no_grad():
                for token_id in self.token_extended_ids:
                    embedding_layer.weight[token_id].requires_grad_(True)
    
    def _init_vocab(self, config):
        self.backend_vocab = config.backend_vocab
        self.backend_strategy = config.backend_strategy
        self.backend_hint_category = config.backend_hint_category
        self.backend_special_tokens = config.backend_special_tokens
        self.backend_enable_hint = self.has_mem and self.backend_strategy == 'hint'
        self.backend_enable_fusion = self.has_mem and self.backend_strategy == 'fusion'
        self.backend_enable_annotation = self.has_mem and self.backend_strategy == 'annotation'
    
    def _init_tokenizer(self, model, config):
        self._init_vocab(config)

        if self.backend_enable_annotation:
            special_tokens = [get_special_token_by_index(i) for i in range(self.backend_special_tokens)]
        elif self.backend_enable_hint:
            assert self.backend_hint_category in self.backend_vocab, \
                f"Invalid hint category '{self.backend_hint_category}'. Valid options: {list(self.backend_vocab.keys())}"
            special_tokens = self.backend_vocab[self.backend_hint_category]
        else:
            special_tokens = []
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_dir)
        
        if special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': RESERVED_TOKENS})
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

            self.token_sep_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
            self.token_special_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)
            self.token_extended_ids =  self.token_special_ids + [self.token_sep_id]

            self.boundary_tokens = [self.tokenizer.tokenize(t) for t in BOUNDARY_TOKENS]
            device = next(model.parameters()).device
            token_sep_id_tensor = torch.tensor([self.token_sep_id], device=device)
            token_special_ids_tensor = torch.tensor(self.token_special_ids, device=device)
            sentence_boundary_ids_tensor = torch.tensor([
                self.tokenizer.convert_tokens_to_ids(t)
                for t in self.boundary_tokens
                if self.tokenizer.convert_tokens_to_ids(t) != self.tokenizer.unk_token_id
            ], device=device)
            
            self.register_buffer('token_sep_id_tensor', token_sep_id_tensor)
            self.register_buffer('token_special_ids_tensor', token_special_ids_tensor)
            self.register_buffer('sentence_boundary_ids_tensor', sentence_boundary_ids_tensor)
            
            model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)
            self.num_tokens = model.get_output_embeddings().weight.shape[0]
            self.lm_config.vocab_size = self.num_tokens

            info(f"LM (model_type: {self.lm_config.model_type}, vocab_size: {self.num_tokens})")
            
            assert self.token_sep_id not in self.token_special_ids, \
                f"SEP token ID {self.token_sep_id} conflicts with special token IDs {self.token_special_ids}"

            assert len(set(self.token_special_ids)) == len(self.token_special_ids), \
                f"Duplicate special token IDs detected: {self.token_special_ids}"
            
            assert len(self.token_special_ids) == self.backend_special_tokens, \
                f"Expected {self.backend_special_tokens} special tokens, got {len(self.token_special_ids)}"
        else:
            self.num_tokens = model.get_output_embeddings().weight.shape[0]

    def _init_params(self):
        if self.lm_freeze:
            for param in self.lm.parameters():
                param.requires_grad = False
            self._make_special_token_embeddings_trainable(self.lm)
        
        trainable_params = self.get_trainable_parameters()
        for param in trainable_params:
            param.requires_grad = True
        
        self.embedding_layer = SafeEmbeddingWrapper(self.lm.get_input_embeddings())
    
    def _has_sep(
            self,
            context_embeds: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            pad_mask: Optional[torch.Tensor] = None,
            input_ids: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        assert context_embeds.dim() == 3

        if mask is not None:
            context_embeds = context_embeds * mask.unsqueeze(-1)
        
        with torch.autocast(self.device.type, enabled=False):
            states = self.mem_state_proj(context_embeds)
            if self.has_frontend:
                frontend_states = self.mem_frontend(states)
                skill_output = self.mem_backend(frontend_states)
            else:
                skill_output = self.mem_backend(states)

        sep_index = self.mem_backend.action_dim - 1
        step_logits = skill_output['action_logits'][:, -1, :]
        if self.backend_sep_logit_bias:
            step_logits = step_logits.clone()
            step_logits[:, sep_index] += self.backend_sep_logit_bias

        temp_scaling = torch.full_like(step_logits, self.backend_non_sep_temp).clamp(min=1e-7)
        temp_scaling[:, sep_index] = self.backend_sep_temperature
        step_logits = step_logits / temp_scaling
        
        batch_size, seq_len, _ = context_embeds.shape
        device = context_embeds.device
        
        has_content = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if self.backend_sentence_alignment:
            has_boundary = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if pad_mask is None:
            pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        window_size = min(seq_len, self.backend_min_interval) if seq_len > 0 else 0
        if window_size > 0:
            window_pad_mask = pad_mask[:, -window_size:]
            has_content = ~window_pad_mask.any(dim=1)
        
        if input_ids is not None and window_size > 0 and self.backend_sentence_alignment:
            recent_tokens = input_ids[:, -window_size:]
            recent_pad_mask = pad_mask[:, -window_size:]

            # Find last non-padded token in window
            col_indices = torch.arange(window_size-1, -1, -1, device=device).expand(batch_size, window_size)
            non_pad_mask = ~recent_pad_mask
            col_indices = col_indices.masked_fill(~non_pad_mask, -1)
            last_token_index = col_indices.max(dim=1)[0]
            
            # Get the actual last token position
            valid_mask = (last_token_index >= 0)
            last_tokens = torch.full((batch_size,), self.tokenizer.pad_token_id, device=device)
            last_tokens[valid_mask] = recent_tokens[
                torch.arange(batch_size, device=device)[valid_mask],
                last_token_index[valid_mask].clamp(min=0)
            ]
            
            # Check if last token is a sentence boundary
            has_boundary = torch.isin(last_tokens, self.sentence_boundary_ids_tensor)
            if self.backend_enable_hint and has_boundary.any():
                special_token_mask = torch.isin(input_ids, self.token_special_ids_tensor)
                last_special_pos = special_token_mask.long().argmax(dim=1)
                token_counts = input_ids.size(1) - last_special_pos
                has_sufficient_distance = token_counts >= self.backend_min_interval
                has_boundary &= has_sufficient_distance

        if self.training:
            final_tau = 0.5
            initial_tau = 1.0
            if self.max_steps:
                tau = initial_tau - (initial_tau - final_tau) * (self.current_step / self.max_steps)
            else:
                tau = initial_tau
        
            noise_mask = torch.ones_like(step_logits, dtype=torch.bool)
            noise_mask[:, sep_index] = False
            noisy_logits = step_logits + (torch.randn_like(step_logits) * self.backend_sep_noise_scale * noise_mask)
            
            gumbel_samples = F.gumbel_softmax(
                noisy_logits,
                tau=tau,
                hard=False,
                dim=-1
            )
            
            has_sep = (gumbel_samples[:, sep_index] > 0.5) & has_content
        else:
            probs = F.softmax(step_logits, dim=-1)
            sep_pred = torch.multinomial(probs, num_samples=1).squeeze(-1) == sep_index
            has_sep = sep_pred & has_content
        
        return has_sep if not self.backend_sentence_alignment else has_sep & has_boundary
    
    def _get_annotation(self,
                        context_embeds: torch.Tensor,
                        return_ids: bool = True,
                        return_loss: bool = False):
        if context_embeds.dim() == 2:
            context_embeds = context_embeds.unsqueeze(0)
        batch_size, _, hidden_size = context_embeds.shape
        device = context_embeds.device
        
        loss_list = []
        annotation_embeds = []
        annotation_ids = [] if return_ids else None
        
        current_embeds = context_embeds
        if self.backend_enable_annotation:
            sep_count = torch.zeros(batch_size, dtype=torch.int, device=device)
        
        special_ids = self.token_special_ids_tensor

        with torch.no_grad():
            special_embeddings = self.embedding_layer(special_ids)
            sep_embed = self.embedding_layer(self.token_sep_id_tensor) if self.backend_enable_annotation else None
        
        if self.backend_enable_annotation:
            max_length = self.backend_anno_max_length + 1
        else:
            max_length = 1
        
        for step_idx in range(max_length):
            with torch.autocast(self.device.type, enabled=False):
                states = self.mem_state_proj(current_embeds)
                if self.has_frontend:
                    frontend_states = self.mem_frontend(states)
                    skill_output = self.mem_backend(frontend_states)
                else:
                    skill_output = self.mem_backend(states)
            
            if return_loss:
                mem_loss = self.mem_backend.compute_losses(skill_output)['total_loss']
                loss_list.append(mem_loss)
            
            step_logits = skill_output['action_logits'][:, -1, :]

            # Prevent SEP token sampling
            if not self.backend_gen_sep:
                step_logits = step_logits.clone()
                step_logits[:, self.mem_backend.action_dim - 1] = -float('inf')
            
            anno_indices = self._get_anno_next_token(step_logits)
            
            if step_idx == max_length - 1 and self.backend_gen_sep:
                force_sep_mask = (sep_count == 0)
                anno_indices = torch.where(
                    force_sep_mask,
                    torch.tensor(self.mem_backend.action_dim - 1, device=device),
                    anno_indices
                )
            
            is_sep = (anno_indices == self.mem_backend.action_dim - 1)
            if self.backend_gen_sep:
                sep_count += is_sep.int()
            
            safe_indices = anno_indices.masked_fill(is_sep, 0)
            non_sep_embeds = special_embeddings[safe_indices]

            if self.backend_gen_sep:
                next_embeds = torch.where(
                    is_sep.unsqueeze(-1),
                    sep_embed,
                    non_sep_embeds
                )
            else:
                next_embeds = non_sep_embeds
            
            if return_ids:
                if self.backend_gen_sep:
                    non_sep_ids = special_ids[safe_indices]
                    next_ids = torch.where(
                        is_sep,
                        self.token_sep_id_tensor,
                        non_sep_ids
                    ).unsqueeze(1)
                else:
                    next_ids = special_ids[safe_indices].unsqueeze(1)
                annotation_ids.append(next_ids)
            
            annotation_embeds.append(next_embeds.unsqueeze(1))
            
            if self.backend_gen_sep and (sep_count >= 1).all():
                break
            
            current_embeds = torch.cat([current_embeds, next_embeds.unsqueeze(1)], dim=1)
        
        if annotation_embeds:
            annotation_embeds = torch.cat(annotation_embeds, dim=1)
        else:
            annotation_embeds = torch.empty(batch_size, 0, hidden_size, device=device)
        
        if return_ids and annotation_ids:
            annotation_ids = torch.cat(annotation_ids, dim=1)
        elif return_ids:
            annotation_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        
        losses = torch.stack(loss_list).mean() if loss_list else None
        return annotation_embeds, annotation_ids, losses
    
    def _get_anno_next_token(self, logits):
        if self.training:
            final_tau = 0.5
            initial_tau = 1.0
            if self.max_steps:
                tau = initial_tau - (initial_tau - final_tau) * (self.current_step / self.max_steps)
            else:
                tau = initial_tau

            noise_mask = torch.ones_like(logits)
            noise_mask[..., -1] = 0  # Do not perturb SEP token selection
            perturbed_logits = logits + (torch.randn_like(logits) * self.anno_noise_scale * noise_mask)
            
            return F.gumbel_softmax(
                perturbed_logits,
                tau=tau,
                hard=True,
                dim=-1
            ).argmax(-1)
        else:
            probs = F.softmax(logits / self.anno_temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _get_next_token(self, logits):
        return self.logits_decoder.decode_logits(logits)

    def _backend_fuse_features(self, state_embeds, action_embeds):
        state_embeds = state_embeds.to(self.dtype)
        action_embeds = action_embeds.to(self.dtype)
        
        state_norm = self.backend_state_norm(state_embeds.float()).to(self.dtype)
        action_norm = self.backend_action_norm(action_embeds.float()).to(self.dtype)

        combined = torch.cat([state_norm, action_norm], dim=-1)
        gate = self.backend_fusion_gate(combined.to(self.backend_fusion_gate[0].weight.dtype))
        return gate * state_norm + (1 - gate) * action_norm
    
    def _need_anno(self, count):
        return (
            (self.backend_enable_hint and (self.backend_max_hints < 0 or count < self.backend_max_hints)) or
            (self.backend_enable_annotation and (self.backend_max_annotations < 0 or count < self.backend_max_annotations))
        )

    @property
    def has_anno(self):
        return self.backend_anno
    
    @property
    def anno_max_length(self):
        return self.backend_anno_max_length
    
    @property
    def num_special_tokens(self):
        return self.backend_special_tokens
    
    @property
    def mem(self):
        return self.mem_backend
    
    @property
    def config(self):
        return self.lm_config
    
    def set_max_steps(self, max_steps):
        self.max_steps = max_steps
    
    def step(self):
        if self.current_step < self.max_steps:
            self.current_step += 1
            progress = self.current_step / self.max_steps
            self.backend_sep_noise_scale = max(
                self.noise_floor,
                self.noise_scale * (1 - progress)**self.noise_decay_power
            )
            self.backend_sep_logit_bias = min(
                self.backend_max_sep_bias,
                self.min_sep_bias + (self.backend_max_sep_bias - self.min_sep_bias) * progress**self.bias_growth_power
            )
    
    def calculate_adaptive_loss(self, annotations_added, seq_len, loss_list, device):
        if loss_list:
            insertion_loss = torch.stack(loss_list).mean()
        else:
            insertion_loss = torch.tensor(0., device=device, requires_grad=True)
        
        annotations_added_f = torch.tensor(float(annotations_added),
                                        dtype=torch.float32,
                                        device=device,
                                        requires_grad=True)
        
        min_annos_f = torch.tensor(1.0,
                                dtype=torch.float32,
                                device=device,
                                requires_grad=True)
        
        max_annos = min(self.backend_max_annos, seq_len // max(1, self.backend_min_interval))
        max_annos_f = torch.tensor(float(max_annos),
                                dtype=torch.float32,
                                device=device,
                                requires_grad=True)
        
        with torch.set_grad_enabled(True):
            diversity_weight = torch.sigmoid(annotations_added_f - min_annos_f)
            diversity_penalty = (1 - diversity_weight) * self.diversity_strength
            
            density_ratio = torch.where(
                max_annos_f > 0,
                annotations_added_f / max_annos_f,
                torch.tensor(0., device=device, requires_grad=True)
            )
            
            density_penalty = (
                F.relu(0.2 - density_ratio) * self.density_strength +
                F.relu(density_ratio - 0.8) * (self.density_strength * 0.5)
            )
            
            adaptive_penalty = 0.6 * diversity_penalty + 0.4 * density_penalty
            total_loss = insertion_loss + adaptive_penalty
            
            if self.training and torch.isnan(total_loss).any():
                return torch.tensor(0., device=device, requires_grad=True)
            
        return total_loss
    
    def annotate(
        self,
        input_ids: torch.Tensor,
        input_labels: torch.Tensor,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        assert self.backend_anno, "Annotation/Hint mode must be enabled"
        
        with torch.no_grad():
            state_embeds = self.embedding_layer(input_ids)
            sep_embed = self.embedding_layer(self.token_sep_id_tensor).view(1, 1, -1)
        
        device = input_ids.device
        batch_size, seq_len, hidden_size = state_embeds.shape

        loss_list = []
        all_embeds = []
        all_labels = []
        batch_annotations = []
        
        for i in range(batch_size):
            seq_embeds = []
            seq_labels = []
            annotations_added = 0
            
            token_embeds = state_embeds[i]
            token_labels = input_labels[i]
            
            pos = 0
            while pos < seq_len:
                if self._need_anno(annotations_added):
                    remaining_tokens = seq_len - pos
                    window_size = min(remaining_tokens, self.backend_context_window)
                    
                    if window_size == 0:
                        seq_embeds.append(token_embeds[pos].view(1, 1, -1))
                        seq_labels.append(token_labels[pos].view(1, 1))
                        pos += 1
                        continue
                    
                    current_context = torch.cat(seq_embeds, dim=1) if seq_embeds else torch.zeros(1, 0, hidden_size, device=device)
                    current_labels = torch.cat(seq_labels, dim=1).squeeze(0) if seq_labels else torch.zeros(0, dtype=torch.long, device=device)
                    
                    max_length = current_context.size(1) + window_size
                    padded_contexts = torch.zeros(window_size, max_length, hidden_size, device=device)
                    mask = torch.zeros(window_size, max_length, dtype=torch.bool, device=device)
                    pad_mask = torch.zeros(window_size, max_length, dtype=torch.bool, device=device)
                    input_ids_tensor = torch.full((window_size, max_length),
                        self.tokenizer.pad_token_id,
                        device=device)

                    for j in range(window_size):
                        context_length = current_context.size(1) + j + 1
                        padded_contexts[j, :context_length] = torch.cat([
                            current_context.squeeze(0),
                            token_embeds[pos:pos+j+1]
                        ])
                        mask[j, :context_length] = True
                        context_labels = torch.cat([
                            current_labels,
                            token_labels[pos:pos+j+1]
                        ])
                        pad_mask[j, :context_length] = (context_labels == self.label_pad_token_id)
                        input_ids_tensor[j, :context_length] = context_labels
                    
                    sep_flags = self._has_sep(padded_contexts, mask=mask, pad_mask=pad_mask, input_ids=input_ids_tensor)
                    sep_positions = (sep_flags == 1).nonzero(as_tuple=True)[0]

                    if sep_positions.numel() > 0:
                        first_sep_index = sep_positions[0].item()
                        window_size = first_sep_index + 1
                        has_sep = True
                    else:
                        has_sep = False
                    
                    for j in range(window_size):
                        seq_embeds.append(token_embeds[pos + j].view(1, 1, -1))
                        seq_labels.append(token_labels[pos + j].view(1, 1))
                    
                    pos += window_size
                    
                    if has_sep:
                        current_context = torch.cat(seq_embeds, dim=1)
                        context_with_sep = torch.cat([
                            current_context,
                            sep_embed
                        ], dim=1)

                        annotation_embeds, _, mem_loss = self._get_annotation(
                            context_with_sep,
                            return_loss=return_loss
                        )
                        
                        if mem_loss is not None:
                            loss_list.append(mem_loss)
                        
                        if self.is_valid_anno(annotation_embeds):
                            annotation_embeds = annotation_embeds.squeeze(0)
                            if self.backend_enable_annotation:
                                annotation_embeds = torch.cat([
                                    sep_embed.squeeze(0),
                                    annotation_embeds
                                ])
                            
                            anno_len = annotation_embeds.size(0)
                            seq_embeds.append(annotation_embeds.unsqueeze(0))
                            seq_labels.append(
                                torch.full((1, anno_len), self.label_pad_token_id, device=device)
                            )
                            annotations_added += 1
                else:
                    seq_embeds.append(token_embeds[pos].view(1, 1, -1))
                    seq_labels.append(token_labels[pos].view(1, 1))
                    pos += 1
            
            full_seq_embeds = torch.cat(seq_embeds, dim=1).squeeze(0)
            full_seq_labels = torch.cat(seq_labels, dim=1).squeeze(0)
            all_embeds.append(full_seq_embeds)
            all_labels.append(full_seq_labels)
            batch_annotations.append(annotations_added)
        
        max_len = max(e.size(0) for e in all_embeds)
        full_embeds = torch.zeros(batch_size, max_len, hidden_size, device=device)
        full_labels = torch.full(
            (batch_size, max_len),
            self.label_pad_token_id,
            dtype=torch.long,
            device=device
        )
        
        for i, (embeds, labels) in enumerate(zip(all_embeds, all_labels)):
            full_embeds[i, :embeds.size(0)] = embeds
            full_labels[i, :labels.size(0)] = labels
        
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            lm_out = self.lm(inputs_embeds=full_embeds)

        if return_loss:
            losses = self.calculate_adaptive_loss(
                annotations_added=sum(batch_annotations) // batch_size,
                seq_len=seq_len,
                loss_list=loss_list,
                device=device
            )
        else:
            losses = None

        return {
            'labels': full_labels,
            'states': full_embeds,
            'logits': lm_out.logits,
            'losses': losses
        }

    def is_valid_anno(self, annotation_embeds):
        seq_len = annotation_embeds.size(-2)
        if self.backend_enable_annotation:
            return seq_len > 1
        elif self.backend_enable_hint:
            return seq_len > 0
        else:
            return False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        if torch.any(input_ids >= self.num_tokens):
            input_ids = input_ids.clamp(0, self.num_tokens - 1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        
        state_embeds = self.embedding_layer(input_ids)
        batch_size, _, hidden_size = state_embeds.shape
        
        if self.backend_anno:
            pad_mask = (attention_mask == 0)
            sep_detected = self._has_sep(
                state_embeds, 
                pad_mask=pad_mask,
                input_ids=input_ids
            )
            
            new_embeds = []
            new_labels = []
            
            sep_embed = self.embedding_layer(self.token_sep_id_tensor).view(1, -1)

            for i in range(batch_size):
                seq_embeds = state_embeds[i]
                seq_input_ids = input_ids[i]
                new_embeds_list = [seq_embeds]
                new_labels_list = [seq_input_ids]
            
                if sep_detected[i].item():
                    seq_embeds = torch.cat([seq_embeds, sep_embed])
                    annotation_embeds, _, _ = self._get_annotation(seq_embeds)
                    
                    if self.is_valid_anno(annotation_embeds):
                        annotation_embeds = annotation_embeds.squeeze(0)
                        if self.backend_enable_annotation:
                            annotation_embeds = torch.cat([sep_embed, annotation_embeds])
                        anno_len = annotation_embeds.size(0)
                        new_embeds_list.append(annotation_embeds)
                        new_labels_list.append(torch.full((anno_len,), self.label_pad_token_id, device=device))
                
                new_embeds.append(torch.cat(new_embeds_list))
                new_labels.append(torch.cat(new_labels_list))
            
            max_len = max(tensor.size(0) for tensor in new_embeds)
            full_embeds = torch.zeros(batch_size, max_len, hidden_size, device=device)

            for i in range(batch_size):
                seq_len_i = new_embeds[i].size(0)
                full_embeds[i, :seq_len_i] = new_embeds[i]
            
            with torch.autocast(self.device.type, dtype=self.dtype):
                lm_out = self.lm(inputs_embeds=full_embeds)
        
            return {
                'logits': lm_out.logits,
                'states': state_embeds
            }
        else:
            if self.backend_enable_fusion:
                with torch.autocast(self.device.type, enabled=False):
                    skill_output = self.mem_backend(state_embeds)
                action_logits = skill_output['action_logits']
                action_embeds = self.backend_action_proj(action_logits)
                state_embeds = self._backend_fuse_features(state_embeds, action_embeds)
            
            with torch.autocast(self.device.type, dtype=self.dtype):
                lm_out = self.lm(inputs_embeds=state_embeds)

            return {
                'logits': lm_out.logits,
                **skill_output
            }

    def append_token(self, seq, token, device):
        if isinstance(token, torch.Tensor):
            if token.dim() == 0:
                return torch.cat([seq, token.view(1)])
            return torch.cat([seq, token.flatten()])
        return torch.cat([seq, torch.tensor([token], device=device)])
    
    def _generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        batch_size = input_ids.shape[0]
        sequences = [input_ids[i].clone() for i in range(batch_size)]
        current_embeds = [self.embedding_layer(seq).unsqueeze(0) for seq in sequences]
        
        eos_flags = [False] * batch_size
        annotations_added = [0] * batch_size
        active_indices = list(range(batch_size))
        
        sep_embed = self.embedding_layer(self.token_sep_id_tensor).view(1, 1, -1)
        pad_masks = [torch.zeros(seq.size(0), dtype=torch.bool, device=device) for seq in sequences]
        
        for _ in range(self.max_length - input_ids.shape[1]):
            if not active_indices:
                break
            
            next_active = []
            
            for idx in active_indices:
                if eos_flags[idx]:
                    continue
                
                current_pad_mask = pad_masks[idx].unsqueeze(0)
                has_sep = self._has_sep(
                    current_embeds[idx],
                    pad_mask=current_pad_mask,
                    input_ids=sequences[idx].unsqueeze(0)
                ).item()

                if self.backend_anno and has_sep and self._need_anno(annotations_added[idx]):
                    context_with_sep = torch.cat([
                        current_embeds[idx],
                        sep_embed
                    ], dim=1)
                    
                    annotation_embeds, annotation_ids, _ = self._get_annotation(
                        context_with_sep,
                        return_ids=True
                    )
                    
                    if self.is_valid_anno(annotation_embeds):
                        if self.backend_enable_annotation:
                            sequences[idx] = torch.cat([
                                sequences[idx],
                                self.token_sep_id_tensor.view(1),
                                annotation_ids.squeeze(0)
                            ])
                            current_embeds[idx] = torch.cat([
                                context_with_sep,
                                annotation_embeds
                            ], dim=1)
                            new_pad = torch.ones(annotation_ids.size(1) + 1, dtype=torch.bool, device=device)
                        else:
                            sequences[idx] = torch.cat([
                                sequences[idx],
                                annotation_ids.squeeze(0)
                            ])
                            current_embeds[idx] = torch.cat([
                                current_embeds[idx],
                                annotation_embeds
                            ], dim=1)
                            new_pad = torch.ones(annotation_ids.size(1), dtype=torch.bool, device=device)

                        pad_masks[idx] = torch.cat([pad_masks[idx], new_pad])
                        annotations_added[idx] += 1
                
                if len(sequences[idx]) >= self.max_length:
                    eos_flags[idx] = True
                    continue

                context_embeds = current_embeds[idx]
                if self.backend_enable_fusion:
                    with torch.no_grad():
                        skill_output = self.mem_backend(context_embeds)
                        action_logits = skill_output['action_logits']
                        action_embeds = self.backend_action_proj(action_logits)
                        context_embeds = self._backend_fuse_features(context_embeds, action_embeds)
                
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    lm_out = self.lm(inputs_embeds=context_embeds)
                
                logits = lm_out.logits[:, -1, :]
                if hasattr(self, 'token_extended_ids'):
                    logits[:, self.token_extended_ids] = -float('inf')
                next_token = self._get_next_token(logits)
                next_token_embed = self.embedding_layer(next_token).view(1, 1, -1)
                
                sequences[idx] = torch.cat([sequences[idx], next_token.view(1)])
                current_embeds[idx] = torch.cat([current_embeds[idx], next_token_embed], dim=1)
                pad_masks[idx] = torch.cat([pad_masks[idx], torch.zeros(1, dtype=torch.bool, device=device)])
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    eos_flags[idx] = True
                else:
                    next_active.append(idx)

            active_indices = next_active
        
        return torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
    
    def generate(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.has_mem:
            return self._generate(input_ids)
        else:
            return self.lm.generate(input_ids=input_ids)
    
    @classmethod
    def from_pretrained(
        cls,
        config = cfg,
        attn: str = None,
        dist: bool = False,
        checkpoint_path: str = cfg.ckpt_path,
        **kwargs
    ) -> 'MD':
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

        model = cls(config=config, attn=attn, dist=dist, **kwargs)

        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {str(e)}") from e
        
        model_state = state_dict.get('model', state_dict)
        model_state = model_state.get('state_dict', model_state)
        
        try:
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if missing:
                warn(f"Missing keys: {missing}")
            if unexpected:
                warn(f"Unexpected keys: {unexpected}")
        except Exception as e:
            raise RuntimeError(f"Error loading state dict: {str(e)}") from e

        info(f"Loaded pre-trained model from {checkpoint_path}")
        return model.to(get_device())

    def get_trainable_parameters(self):
        trainable_params = []
        
        if self.backend_enable_fusion:
            trainable_params.extend(self.backend_action_proj.parameters())
            trainable_params.extend(self.backend_fusion_gate.parameters())
            trainable_params.extend(self.backend_state_norm.parameters())
            trainable_params.extend(self.backend_action_norm.parameters())
        
        trainable_params.extend(self.mem_backend.parameters())
        trainable_params.extend(self.mem_state_proj.parameters())
        
        if self.has_frontend:
            trainable_params.extend(self.mem_frontend.parameters())

        trainable_params.extend(
            param for param in self.lm.parameters() 
            if param.requires_grad
        )
        
        return trainable_params
