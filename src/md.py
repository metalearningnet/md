import os
import torch
import torch.nn as nn
from skill import SkillMemory
from typing import Dict, Optional
from huggingface_hub import model_info
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from utils import (
    cfg, info, get_device, load_peft_config, get_special_token_by_index,
    LogitsDecoder, SEP_TOKEN, HINT_VOCAB, RESERVED_TOKENS
)

class MD(nn.Module):
    def __init__(self, 
                 config = cfg, 
                 attn: str = None):
        super().__init__()
        self.device = get_device()
        self.use_cache = config.use_cache
        self.model_dir = config.model_dir
        self.max_hints = config.max_hints
        self.max_length = config.max_length
        self.adapter_config = config.adapter
        self.temperature = config.temperature
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

        self.has_anno = self.enable_hint or self.enable_annotation
        info(f"LM {self.config.model_type} (hidden_size: {self.lm_hidden_size})")

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

        self._init_tokenizer(model)
        
        if self.lm_checkpoint:
            model.gradient_checkpointing_enable()
        
        if not self.lm_freeze:
            peft_config_dict = load_peft_config()
            peft_config = LoraConfig(**peft_config_dict)
            model = get_peft_model(model, peft_config)
        else:
            for param in model.parameters():
                param.requires_grad = False
        
        self.lm = model
        self.config = config
        
        self.lm_hidden_size = getattr(
            self.config,
            'hidden_size',
            getattr(
                getattr(self.config, 'text_config', None),
                'hidden_size',
                None
            )
        )

        if self.lm_hidden_size is None:
            raise ValueError("Could not find hidden_size")
        
        if self.temperature is None:
            self.temperature = getattr(self.config, 'temperature', 1.0)
        
        self.logits_decoder = LogitsDecoder(self.config, self.tokenizer, self.temperature)

    def _init_skill(self) -> nn.Module:
        """Initialize SkillMemory with LM-compatible dimensions"""
        if self.skill_integration_strategy in ['hint', 'annotation']:
            self.action_dim = self.num_special_words + 1
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
        assert self.skill_integration_strategy in  ['fusion', 'annotation', 'hint'], f"Invalid skill integration strategy: {self.skill_integration_strategy}"
       
        if self.skill_integration_strategy == 'fusion':
            self._init_action_projection()
    
        self.ext_params = []
        
        if self.lm_freeze:
            for param in self.lm.parameters():
                param.requires_grad = False
            
            embedding_layer = self.lm.get_input_embeddings()
            for id in self.token_extended_ids:
                self.ext_params.append(embedding_layer.weight[id])
        
        trainable_params = self.get_trainable_parameters()

        for param in trainable_params:
            param.requires_grad = True
    
    def _init_tokenizer(self, model):
        """Add specialized tokens ensuring that all are newly added."""
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
            
            self.lm_num_tokens = len(self.tokenizer)
            model.resize_token_embeddings(self.lm_num_tokens)
            
            assert self.token_sep_id not in self.token_special_ids, \
                f"SEP token ID {self.token_sep_id} conflicts with special token IDs {self.token_special_ids}"
        
            assert len(set(self.token_special_ids)) == len(self.token_special_ids), \
                f"Duplicate special token IDs detected: {self.token_special_ids}"
            
            assert len(self.token_special_ids) == self.num_special_words, \
                f"Expected {self.num_special_words} special tokens, got {len(self.token_special_ids)}"
    
    def _has_sep(self, context_embeds: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Determine if SkillMemory would generate SEP token at the end of context"""
        if mask is not None:
            context_embeds = context_embeds * mask.unsqueeze(-1)
        
        skill_output = self.skill_memory(context_embeds)
        
        step_logits = skill_output['action_logits'][:, -1, :]
        anno_indices = self._get_anno_next_token(step_logits)
        sep_index = self.action_dim - 1
        return anno_indices == sep_index
    
    def _get_annotation(self, 
                        context_embeds: torch.Tensor, 
                        return_ids: bool = True, 
                        return_loss: bool = False):
        """Auto-regressively generate annotation embeddings"""
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
        
        special_ids = self.token_special_ids_tensor
        special_embeddings = self.lm.get_input_embeddings()(special_ids)
        sep_embed = self.lm.get_input_embeddings()(self.token_sep_id_tensor) if self.enable_annotation else None
        
        max_length = self.anno_max_length + 1 if self.enable_annotation else 1
        for step_idx in range(max_length):
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
        return torch.argmax(logits, dim=-1)
    
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

    def annotate(
        self, 
        input_ids: torch.Tensor,
        input_labels: torch.Tensor,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        assert self.has_anno, "Annotation/Hint mode must be enabled"
        
        state_embeds = self.lm.get_input_embeddings()(input_ids)
        batch_size, seq_len, hidden_size = state_embeds.shape
        sep_embed = self.lm.get_input_embeddings()(self.token_sep_id_tensor).view(1, 1, -1)
        
        all_embeds = []
        all_labels = []
        loss_list = []
        
        for i in range(batch_size):
            seq_embeds = []
            seq_labels = []
            annotations_added = 0
            
            token_embeds = state_embeds[i]
            token_labels = input_labels[i]
            
            pos = 0
            while pos < seq_len:
                seq_embeds.append(token_embeds[pos].view(1, 1, -1))
                seq_labels.append(token_labels[pos].view(1, 1))
                
                if self._need_anno(annotations_added):
                    current_context = torch.cat(seq_embeds, dim=1)
                    
                    if self._has_sep(current_context).item():
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
                            
                            seq_embeds.append(annotation_embeds.unsqueeze(0))
                            anno_len = annotation_embeds.size(0)
                            seq_labels.append(
                                torch.full((1, anno_len), self.label_pad_token_id, device=input_ids.device)
                            )
                            
                            annotations_added += 1
                            remaining_tokens = seq_len - pos - 1
                            
                            window_size = min(remaining_tokens, self.context_window)
                            if window_size > 0:
                                max_length = current_context.size(1) + window_size
                                padded_contexts = torch.zeros(window_size, max_length, hidden_size, 
                                                            device=input_ids.device)
                                
                                mask = torch.zeros(window_size, max_length, dtype=torch.bool, 
                                                device=input_ids.device)
                                
                                base_context = current_context.squeeze(0)
                                for j in range(window_size):
                                    context_length = base_context.size(0) + j + 1
                                    padded_contexts[j, :context_length] = torch.cat([
                                        base_context,
                                        token_embeds[pos+1:pos+j+2]
                                    ])
                                    mask[j, :context_length] = True
                                
                                sep_flags = self._has_sep(padded_contexts, mask=mask)
                                sep_positions = (sep_flags == 1).nonzero(as_tuple=True)[0]
                                if sep_positions.numel() > 0:
                                    first_sep_index = sep_positions[0].item()
                                    
                                    for j in range(first_sep_index + 1):
                                        seq_embeds.append(token_embeds[pos + j + 1].view(1, 1, -1))
                                        seq_labels.append(token_labels[pos + j + 1].view(1, 1))
                                    
                                    pos += first_sep_index + 1
                                    continue
                                else:
                                    for j in range(window_size):
                                        seq_embeds.append(token_embeds[pos + j + 1].view(1, 1, -1))
                                        seq_labels.append(token_labels[pos + j + 1].view(1, 1))
                                    pos += window_size
                                    continue
                
                pos += 1
            
            full_seq_embeds = torch.cat(seq_embeds, dim=1).squeeze(0)
            full_seq_labels = torch.cat(seq_labels, dim=1).squeeze(0)
            all_embeds.append(full_seq_embeds)
            all_labels.append(full_seq_labels)
        
        max_len = max(e.size(0) for e in all_embeds)
        full_embeds = torch.zeros(batch_size, max_len, hidden_size, device=input_ids.device)
        full_labels = torch.full(
            (batch_size, max_len), 
            self.label_pad_token_id, 
            dtype=torch.long, 
            device=input_ids.device
        )
        
        for i, (embeds, labels) in enumerate(zip(all_embeds, all_labels)):
            full_embeds[i, :embeds.size(0)] = embeds
            full_labels[i, :labels.size(0)] = labels
        
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            lm_out = self.lm(inputs_embeds=full_embeds)

        losses = torch.stack(loss_list).mean() if loss_list and return_loss else None

        return {
            'labels': full_labels,
            'states': full_embeds,
            'logits': lm_out.logits,
            'losses': losses
        }

    def is_valid_anno(self, annotation_embeds):
        if self.enable_annotation:
            return annotation_embeds.size(1) > 1
        elif self.enable_hint:
            return annotation_embeds.size(1) > 0
        else:
            return False

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
        batch_size, _, hidden_size = state_embeds.shape
        
        if self.has_anno:
            sep_detected = self._has_sep(state_embeds)
            
            new_embeds = []
            new_labels = []
            
            sep_embed = self.lm.get_input_embeddings()(self.token_sep_id_tensor).view(1, -1)

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
                        new_labels_list.append(torch.full((anno_len,), self.label_pad_token_id, device=input_ids.device))
                
                new_embeds.append(torch.cat(new_embeds_list))
                new_labels.append(torch.cat(new_labels_list))
            
            max_len = max(tensor.size(0) for tensor in new_embeds)
            full_embeds = torch.zeros(batch_size, max_len, hidden_size, device=state_embeds.device)

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
                with torch.amp.autocast(self.device.type, enabled=False):
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
    
    def _generate(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        device = input_ids.device
        batch_size = input_ids.shape[0]
        sequences = [input_ids[i].clone() for i in range(batch_size)]
        
        embeddings = self.lm.get_input_embeddings()
        current_embeds = [embeddings(seq).unsqueeze(0) for seq in sequences]
        
        eos_flags = [False] * batch_size
        active_indices = list(range(batch_size))
        sep_embed = embeddings(self.token_sep_id_tensor).view(1, 1, -1)
        
        for _ in range(self.max_length - input_ids.shape[1]):
            if not active_indices:
                break
                
            next_active = []
            for idx in active_indices:
                if eos_flags[idx]:
                    continue
                
                if self.has_anno and self._has_sep(current_embeds[idx]).item():
                    if self.enable_annotation:
                        seq_ids = self.append_token(sequences[idx], self.token_sep_id_tensor, device)
                    
                    seq_embeds = torch.cat([current_embeds[idx], sep_embed], dim=1)
                    annotation_embeds, annotation_ids, _ = self._get_annotation(
                        seq_embeds,
                        return_ids=True
                    )

                    if self.is_valid_anno(annotation_embeds):
                        if self.enable_annotation:
                            sequences[idx] = self.append_token(seq_ids, annotation_ids, device)
                            current_embeds[idx] = torch.cat([seq_embeds, annotation_embeds], dim=1)
                        else:
                            sequences[idx] = self.append_token(sequences[idx], annotation_ids, device)
                            current_embeds[idx] = torch.cat([current_embeds[idx], annotation_embeds], dim=1)
                
                context_embeds = current_embeds[idx]
                if self.enable_fusion:
                    with torch.no_grad():
                        skill_output = self.skill_memory(context_embeds)
                        action_logits = skill_output['action_logits']
                        action_embeds = self.action_proj(action_logits)
                        context_embeds = self._fuse_features(context_embeds, action_embeds)
                
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    lm_out = self.lm(inputs_embeds=context_embeds)
                
                next_token = self._get_next_token(lm_out.logits[:, -1, :])
                next_token_embed = embeddings(next_token).view(1, 1, -1)
                
                sequences[idx] = self.append_token(sequences[idx], next_token, device)
                current_embeds[idx] = torch.cat([current_embeds[idx], next_token_embed], dim=1)
                
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
        
        return trainable_params + self.ext_params
