import os
import math
import torch
import torch.nn as nn
from skill import SkillMemory
import torch.nn.functional as F
from typing import Dict, Optional
from huggingface_hub import model_info
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from utils import (
    cfg, info, warn, get_device, load_peft_config, get_special_token_by_index,
    LogitsDecoder, SEP_TOKEN, HINT_VOCAB, RESERVED_TOKENS
)

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
        
        if getattr(peft_config, "init_lora_weights", "gaussian") == "gaussian":
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
        self.dist = dist
        self.device = get_device()

        self.use_cache = config.use_cache
        self.model_dir = config.model_dir
        self.max_hints = config.max_hints
        self.max_length = config.max_length
        self.adapter_config = config.adapter
        self.temperature = config.temperature
        self.min_interval = config.min_interval
        self.hint_category = config.hint_category
        self.context_window = config.context_window
        self.max_annotations = config.max_annotations
        self.anno_max_length = config.anno_max_length
        self.num_special_words = config.num_special_words
        self.attn = config.attn if attn is None else attn
        self.label_pad_token_id = config.label_pad_token_id
        self.dtype = torch.float16 if config.precision == '16-mixed' else torch.bfloat16

        self.lm_path = config.lm_path
        self.lm_coef = config.lm_coef
        self.lm_freeze = config.lm_freeze
        self.lm_checkpoint = config.lm_checkpoint

        self.skill_coef = config.skill_coef
        self.skill_config = config.skill_config
        self.skill_checkpoint = config.skill_checkpoint
        self.skill_integration_strategy = config.skill_integration_strategy

        self.enable_hint = self.skill_coef and self.skill_integration_strategy == 'hint'
        self.enable_fusion = self.skill_coef and self.skill_integration_strategy == 'fusion'
        self.enable_annotation = self.skill_coef and self.skill_integration_strategy == 'annotation'

        self._init_lm()
        self._init_skill()
        self._init_params()

        self.sep_temperature = 0.7
        self.sep_noise_scale = 0.1
        self.anno_temperature = 0.7
        self.anno_noise_scale = 0.1
        self.density_strength = 1.0
        self.diversity_strength = 1.0
        self.sep_lookback_window = 16

        self.max_steps = 0
        self.current_step = 0
        self.has_anno = self.enable_hint or self.enable_annotation
        self.max_annos = self.max_annotations if self.enable_annotation else self.max_hints

        info(f"MD (hidden_size: {self.hidden_size}, skill: {config.skill_info}, special_tokens: {self.num_special_words}, preference_optimizer: {config.po})")

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps
    
    def step(self):
        if self.current_step < self.max_steps:
            self.current_step += 1
    
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

    def _init_lm(self):
        config = AutoConfig.from_pretrained(self.model_dir)
        config.use_cache = self.use_cache

        info = model_info(self.lm_path)
        model_type = info.config['model_type']
        if config.model_type != model_type:
            raise ValueError(f"Expected model type {model_type}, got {config.model_type}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            config=config,
            trust_remote_code=True,
            attn_implementation=self.attn
        )
        self.config = config
        self._init_tokenizer(model)
        
        if self.lm_checkpoint:
            model.gradient_checkpointing_enable()
        
        if not self.lm_freeze:
            model = self._init_peft(model)
        
        self.lm = model
        self.hidden_size = getattr(
            self.config,
            'hidden_size',
            getattr(
                getattr(self.config, 'text_config', None),
                'hidden_size',
                None
            )
        )

        if self.hidden_size is None:
            raise ValueError("Could not find hidden_size")
        
        if self.temperature is None:
            self.temperature = getattr(self.config, 'temperature', 1.0)
        
        self.logits_decoder = LogitsDecoder(self.config, self.tokenizer, self.temperature)

    def _init_skill(self) -> nn.Module:
        if self.skill_integration_strategy in ['hint', 'annotation']:
            self.action_dim = self.num_special_words + 1
        else:
            self.action_dim = self.skill_config.get('action_dim', self.num_tokens)
            
        self.skill_config.update({
            'num_tokens': self.num_tokens,
            'state_dim': self.hidden_size,
            'action_dim': self.action_dim,
            'hidden_dim': self.skill_config.get('hidden_dim', self.hidden_size),
            'checkpoint': self.skill_checkpoint
        })

        self.skill_memory = SkillMemory(**self.skill_config)
    
    def _init_adapter(self):
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )
        self.state_norm = nn.LayerNorm(self.hidden_size)
        self.action_norm = nn.LayerNorm(self.hidden_size)
        proj_scale = self.adapter_config.get('proj_scale', 2)
        min_dim = self.adapter_config.get('min_proj_dim', self.hidden_size)
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
            nn.Linear(proj_dim, self.hidden_size),
            nn.Dropout(self.adapter_config.get('proj_dropout', 0.1))
        ]
        
        self.action_proj = nn.Sequential(*layers)
        
        gelu_gain = 1.5957696
        nn.init.xavier_normal_(self.action_proj[0].weight, gain=gelu_gain)
        nn.init.zeros_(self.action_proj[0].bias)

        nn.init.xavier_normal_(self.action_proj[-2].weight, gain=1.0)
        nn.init.zeros_(self.action_proj[-2].bias)
        
        assert self.action_proj[-2].out_features == self.hidden_size

    def _make_special_token_embeddings_trainable(self, model):
        embedding_layer = model.get_input_embeddings()
        device = embedding_layer.weight.device
        embedding_layer.weight.requires_grad_(True)
        
        self.register_buffer('_trainable_mask', torch.zeros_like(embedding_layer.weight, dtype=torch.bool, device=device))
        
        with torch.no_grad():
            for token_id in self.token_extended_ids:
                self._trainable_mask[token_id] = True
        
        def _hook(grad):
            return grad * self._trainable_mask.to(grad.device)
        
        embedding_layer.weight.register_hook(_hook)
        
        with torch.no_grad():
            for i, param in enumerate(embedding_layer.weight):
                if i not in self.token_extended_ids:
                    param.requires_grad_(False)
    
    def _init_params(self):
        assert self.skill_integration_strategy in  ['fusion', 'annotation', 'hint'], f"Invalid skill integration strategy: {self.skill_integration_strategy}"
       
        if self.skill_integration_strategy == 'fusion':
            self._init_adapter()
        
        if self.lm_freeze:
            for param in self.lm.parameters():
                param.requires_grad = False
            self._make_special_token_embeddings_trainable(self.lm)
        
        trainable_params = self.get_trainable_parameters()
        for param in trainable_params:
            param.requires_grad = True
    
    def _init_tokenizer(self, model):
        if self.enable_annotation:
            special_tokens = [get_special_token_by_index(i) for i in range(self.num_special_words)]
        elif self.enable_hint:
            assert self.hint_category in HINT_VOCAB, \
                f"Invalid hint category '{self.hint_category}'. Valid options: {list(HINT_VOCAB.keys())}"
            special_tokens = HINT_VOCAB[self.hint_category]
        else:
            special_tokens = []
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        if special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': RESERVED_TOKENS})
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

            self.token_sep_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
            self.token_special_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)
            self.token_extended_ids =  self.token_special_ids + [self.token_sep_id]

            device = next(model.parameters()).device
            token_sep_id_tensor = torch.tensor([self.token_sep_id], device=device)
            token_special_ids_tensor = torch.tensor(self.token_special_ids, device=device)
            
            self.register_buffer('token_sep_id_tensor', token_sep_id_tensor)
            self.register_buffer('token_special_ids_tensor', token_special_ids_tensor)
            
            model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)
            self.num_tokens = model.get_output_embeddings().weight.shape[0]
            self.config.vocab_size = self.num_tokens

            info(f'LM (model_type: {self.config.model_type}, vocab_size: {self.num_tokens})')
            
            assert self.token_sep_id not in self.token_special_ids, \
                f"SEP token ID {self.token_sep_id} conflicts with special token IDs {self.token_special_ids}"

            assert len(set(self.token_special_ids)) == len(self.token_special_ids), \
                f"Duplicate special token IDs detected: {self.token_special_ids}"
            
            assert len(self.token_special_ids) == self.num_special_words, \
                f"Expected {self.num_special_words} special tokens, got {len(self.token_special_ids)}"

    def _has_sep(
            self,
            context_embeds: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            pad_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        assert context_embeds.dim() == 3

        if mask is not None:
            context_embeds = context_embeds * mask.unsqueeze(-1)
        
        with torch.autocast(self.device.type, enabled=False):
            skill_output = self.skill_memory(context_embeds)

        step_logits = skill_output['action_logits'][:, -1, :]
        sep_index = self.action_dim - 1
        batch_size, seq_len, _ = context_embeds.shape

        if pad_mask is None:
            pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=context_embeds.device)

        has_content = torch.zeros(batch_size, dtype=torch.bool, device=context_embeds.device)
        
        if seq_len > 0:
            window_size = min(seq_len, self.sep_lookback_window)
            window_pad_mask = pad_mask[:, -window_size:]
            has_content = ~window_pad_mask.any(dim=1)

        if self.training:
            final_tau = 0.5
            initial_tau = 1.0
            if self.max_steps:
                tau = initial_tau - (initial_tau - final_tau) * (self.current_step / self.max_steps)
            else:
                tau = self.sep_temperature
        
            noise_mask = torch.ones_like(step_logits, dtype=torch.bool)
            noise_mask[:, sep_index] = False
            noisy_logits = step_logits + (torch.randn_like(step_logits) * self.sep_noise_scale * noise_mask)
            
            gumbel_samples = F.gumbel_softmax(
                noisy_logits,
                tau=tau,
                hard=False,
                dim=-1
            )
            
            return (gumbel_samples[:, sep_index] > 0.5) & has_content
        else:
            probs = F.softmax(step_logits / self.sep_temperature, dim=-1)
            sep_pred = torch.multinomial(probs, num_samples=1).squeeze(-1) == sep_index
            return sep_pred & has_content
    
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
        if self.enable_annotation:
            sep_count = torch.zeros(batch_size, dtype=torch.int, device=device)
        
        embeding_layer = self.lm.get_input_embeddings()
        special_ids = self.token_special_ids_tensor
        special_embeddings = embeding_layer(special_ids)
        sep_embed = embeding_layer(self.token_sep_id_tensor) if self.enable_annotation else None
        
        max_length = self.anno_max_length + 1 if self.enable_annotation else 1
        for step_idx in range(max_length):
            with torch.autocast(self.device.type, enabled=False):
                skill_output = self.skill_memory(current_embeds)
            
            if return_loss:
                skill_loss = self.skill_memory.compute_losses(skill_output)['total_loss']
                loss_list.append(skill_loss)
            step_logits = skill_output['action_logits'][:, -1, :]

            # Prevent SEP token sampling in non-annotation mode
            if not self.enable_annotation:
                step_logits = step_logits.clone()
                step_logits[:, self.action_dim - 1] = -float('inf')
            
            anno_indices = self._get_anno_next_token(step_logits)
            
            if step_idx == max_length - 1 and self.enable_annotation:
                force_sep_mask = (sep_count == 0)
                anno_indices = torch.where(
                    force_sep_mask,
                    torch.tensor(self.action_dim - 1, device=device),
                    anno_indices
                )
            
            is_sep = (anno_indices == self.action_dim - 1)
            if self.enable_annotation:
                sep_count += is_sep.int()
            
            safe_indices = anno_indices.masked_fill(is_sep, 0)
            non_sep_embeds = special_embeddings[safe_indices]

            if self.enable_annotation:
                next_embeds = torch.where(
                    is_sep.unsqueeze(-1),
                    sep_embed,
                    non_sep_embeds
                )
            else:
                next_embeds = non_sep_embeds
            
            if return_ids:
                if self.enable_annotation:
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
            
            if self.enable_annotation and (sep_count >= 1).all():
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
                tau = self.anno_temperature

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

    def _fuse_features(self, state_embeds, action_embeds):
        state_embeds = state_embeds.to(self.dtype)
        action_embeds = action_embeds.to(self.dtype)
        
        state_norm = self.state_norm(state_embeds.float()).to(self.dtype)
        action_norm = self.action_norm(action_embeds.float()).to(self.dtype)

        combined = torch.cat([state_norm, action_norm], dim=-1)
        gate = self.fusion_gate(combined.to(self.fusion_gate[0].weight.dtype))
        return gate * state_norm + (1 - gate) * action_norm
    
    def _need_anno(self, count):
        return (
            (self.enable_hint and (self.max_hints < 0 or count < self.max_hints)) or
            (self.enable_annotation and (self.max_annotations < 0 or count < self.max_annotations))
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
        
        max_annos = min(self.max_annotations, seq_len // max(1, self.min_interval))
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
        assert self.has_anno, "Annotation/Hint mode must be enabled"
        
        device = input_ids.device
        embedding_layer = self.lm.get_input_embeddings()
        state_embeds = embedding_layer(input_ids)
        batch_size, seq_len, hidden_size = state_embeds.shape
        sep_embed = embedding_layer(self.token_sep_id_tensor).view(1, 1, -1)
        
        all_embeds = []
        all_labels = []
        loss_list = []
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
                    window_size = min(remaining_tokens, self.context_window)
                    
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
                    
                    sep_flags = self._has_sep(padded_contexts, mask=mask, pad_mask=pad_mask)
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

                        annotation_embeds, _, skill_loss = self._get_annotation(
                            context_with_sep,
                            return_loss=return_loss
                        )
                        
                        if skill_loss is not None:
                            loss_list.append(skill_loss)
                        
                        if self.is_valid_anno(annotation_embeds):
                            annotation_embeds = annotation_embeds.squeeze(0)
                            if self.enable_annotation:
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
        if self.enable_annotation:
            return seq_len > 1
        elif self.enable_hint:
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
        
        embedding_layer = self.lm.get_input_embeddings()
        state_embeds = embedding_layer(input_ids)
        batch_size, _, hidden_size = state_embeds.shape
        
        if self.has_anno:
            sep_detected = self._has_sep(state_embeds)
            
            new_embeds = []
            new_labels = []
            
            sep_embed = embedding_layer(self.token_sep_id_tensor).view(1, -1)

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
                        if self.enable_annotation:
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
            if self.enable_fusion:
                with torch.autocast(self.device.type, enabled=False):
                    skill_output = self.skill_memory(state_embeds)
                action_logits = skill_output['action_logits']
                action_embeds = self.action_proj(action_logits)
                state_embeds = self._fuse_features(state_embeds, action_embeds)
            
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

        embedding_layer = self.lm.get_input_embeddings()
        current_embeds = [embedding_layer(seq).unsqueeze(0) for seq in sequences]
        
        eos_flags = [False] * batch_size
        annotations_added = [0] * batch_size
        active_indices = list(range(batch_size))
        
        sep_embed = embedding_layer(self.token_sep_id_tensor).view(1, 1, -1)
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
                    pad_mask=current_pad_mask
                ).item()

                if self.has_anno and has_sep and self._need_anno(annotations_added[idx]):
                    context_with_sep = torch.cat([
                        current_embeds[idx],
                        sep_embed
                    ], dim=1)
                    
                    annotation_embeds, annotation_ids, _ = self._get_annotation(
                        context_with_sep,
                        return_ids=True
                    )
                    
                    if self.is_valid_anno(annotation_embeds):
                        if self.enable_annotation:
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
                if self.enable_fusion:
                    with torch.no_grad():
                        skill_output = self.skill_memory(context_embeds)
                        action_logits = skill_output['action_logits']
                        action_embeds = self.action_proj(action_logits)
                        context_embeds = self._fuse_features(context_embeds, action_embeds)
                
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    lm_out = self.lm(inputs_embeds=context_embeds)
                
                logits = lm_out.logits[:, -1, :]
                if hasattr(self, 'token_extended_ids'):
                    logits[:, self.token_extended_ids] = -float('inf')
                next_token = self._get_next_token(logits)
                next_token_embed = embedding_layer(next_token).view(1, 1, -1)
                
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
        if self.skill_coef:
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
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {str(e)}") from e
        
        model_state = state_dict.get('model', state_dict)
        model_state = model_state.get('state_dict', model_state)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        
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

        for param in self.lm.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        return trainable_params
