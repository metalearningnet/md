MODEL = {
    # Language Model Configuration
    "lm": {
        "path": "Qwen/Qwen3-0.6B",                              # Pretrained model identifier from HuggingFace Hub
        "temperature": 0.7,                                     # Sampling temperature (0.0: deterministic, 1.0: creative)
        "freeze": True,                                         # Freeze base LM weights during training
        "max_length": 1280,                                     # Maximum total sequence length (prompt + response)
        "max_target_length": 1024,                              # Maximum generated output length in tokens
        "max_prompt_length": 512                                # Maximum input prompt length
    },

    "lm_coef": 0.8,                                             # Language modeling loss weight
    "mem_coef": 0.2,                                            # Memory system loss weight

    "sft": False,                                               # Supervised fine-tuning mode
    "frontend": False,                                          # Whether to enable the frontend memory processing module
    "use_initial_prompt": True,                                 # Whether to prepend an initial prompt to inputs
    
    "use_cache": False,                                         # Whether to enable KV caching for faster autoregressive decoding
}

# Memory System Configuration
MEMORY = {
    # Frontend Memory (Preprocessing)
    "frontend": {
        "mem_type": "mac",                                      # Memory architecture type (Memory-as-Context)

        "mac": {
            "depth": 1,                                         # Number of stacked MAC blocks
            "segment_len": 32,                                  # Segment length processed per MAC block
            "longterm_mem_tokens": 32,                          # Tokens allocated for extended memory retention
            "persistent_mem_tokens": 16,                        # Tokens reserved for long-lasting contextual memory

            "use_flex_attn": False,                             # Use FlexAttention for potentially faster/sliding window attention
            "sliding_window_attn": True,                        # Use sliding window attention for efficiency
            "manual_per_sample_grads": False,                   # Use manual per-sample gradient computation

            "neural_mem_heads": 4,                              # Number of attention heads in neural memory
            "neural_mem_head_dim": 64,                          # Dimension per head in neural memory
            "neural_mem_momentum_order": 1,                     # Highest order of momentum to calculate
            "neural_mem_step_transform_max_lr": 1e-1,           # Upper limit for the effective learning rate used in memory updates

            "neural_mem_momentum": True,                        # Enable momentum-based memory updates
            "neural_mem_qk_rmsnorm": True,                      # Apply RMSNorm to Q/K projections
            "neural_mem_weight_residual": True,                 # Adds residual connections in neural memory weight updates
            "neural_mem_attn_pool_chunks": True,                # whether to use attention pooling for chunk derived momentum
            "neural_mem_use_accelerated_scan": False,           # Use an optimized version of the associative scan for faster computation
            "neural_mem_qkv_receives_diff_views": True,         # Q/K/V projections come from different layers/views
            "neural_mem_spectral_norm_surprises": True,         # Apply spectral normalization to memory updates
            "neural_mem_per_parameter_lr_modulation": True,     # Parameter-specific learning rate adaptation
            "neural_mem_per_head_learned_parameters": False     # Independent parameters per memory head
        },

        "update_memory": True                                   # Allow memory updates during inference
    },

    # Backend Memory (Skill Integration)
    "backend": {
        "skill": {
            "state_dim": 256,                                   # Dimension of state embeddings
            "action_dim": 64,                                   # Dimension of action space
            "hidden_dim": 128,                                  # Hidden layer dimension

            "mi_coef": 0.6,                                     # Weight for state-memory mutual information
            "kl_coef": 0.03,                                    # Weight for memory consistency regularization
            "adv_coef": 0.15,                                   # Weight for action-memory disentanglement
            "entropy_coef": 0.05,                               # Weight for policy exploration incentive
        },

        "mem_type": "mac",                                      # Memory architecture type (Memory-as-Context)

        "mac": {
            "depth": 1,                                         # Number of stacked MAC blocks
            "segment_len": 32,                                  # Segment length processed per MAC block
            "longterm_mem_tokens": 32,                          # Tokens allocated for extended memory retention
            "persistent_mem_tokens": 16,                        # Tokens reserved for long-lasting contextual memory

            "use_flex_attn": False,                             # Use FlexAttention for potentially faster/sliding window attention
            "sliding_window_attn": True,                        # Use sliding window attention for efficiency
            "manual_per_sample_grads": False,                   # Use manual gradient computation

            "neural_mem_heads": 4,                              # Number of attention heads in neural memory
            "neural_mem_head_dim": 64,                          # Dimension per head in neural memory
            "neural_mem_momentum_order": 1,                     # Highest order of momentum to calculate
            "neural_mem_step_transform_max_lr": 1e-1,           # Upper limit for the effective learning rate used in memory updates

            "neural_mem_momentum": True,                        # Enable momentum-based memory updates
            "neural_mem_qk_rmsnorm": True,                      # Apply RMSNorm to Q/K projections
            "neural_mem_weight_residual": True,                 # Adds residual connections in neural memory weight updates
            "neural_mem_attn_pool_chunks": True,                # whether to use attention pooling for chunk derived momentum
            "neural_mem_use_accelerated_scan": False,           # Use an optimized version of the associative scan for faster computation
            "neural_mem_qkv_receives_diff_views": True,         # Q/K/V projections come from different layers/views
            "neural_mem_spectral_norm_surprises": True,         # Apply spectral normalization to memory updates
            "neural_mem_per_parameter_lr_modulation": True,     # Parameter-specific learning rate adaptation
            "neural_mem_per_head_learned_parameters": False     # Independent parameters per memory head
        },

        "strategy": "hint",                                     # Integration strategy with LLM: 'fusion' | 'annotation' | 'hint'

        "fusion": {
            "adapter": {
                "proj_scale": -1,                               # Hidden dimension expansion factor (-1 for auto)
                "proj_dropout": 0.1,                            # Dropout rate for adapter layers
                "min_proj_dim": -1,                             # Minimum hidden dimension size (-1 for auto)
                "norm_position": "post"                         # LayerNorm placement: 'pre' | 'post'
            }
        },

        "annotation": {
            "words": 8,                                         # Vocabulary size for annotations
            "max_length": 2,                                    # Maximum tokens per annotation
            "max_annotations": 4,                               # Maximum annotations per response (-1 for unlimited)
            "min_interval": 128,                                # Minimum tokens between annotations
            "tune": True                                        # Fine-tune annotation embeddings
        },

        "hint": {
            "category": "minimal",                              # Hint complexity level: 'minimal' | 'standard' | 'enhanced' | 'advanced'
            "max_hints": 64,                                    # Maximum hints per response (-1 for unlimited)
            "min_interval": 8,                                  # Minimum tokens between hints
            "tune": True,                                       # Fine-tune hint embeddings
            "sentence_alignment": True,                         # Trigger special token generation at sentence boundaries.
            "sep_logit_bias": 3.0,                              # Controls hint insertion frequency by biasing the [SEP] token logits that trigger hint generation
            "sep_temperature": 0.7                              # [SEP] insertion temperature (0.1=always trigger, 1.0=neutral)
        },

        "context_window": 4,                                    # Token lookahead window for generation

        "update_memory": True                                   # Allow memory updates during inference
    }
}

# Memory Optimization
CKPT = {
    "gradient": {
        "lm": False,                                            # Checkpoint LM
        "mem": False                                            # Checkpoint memory system
    }
}

# Training Configuration
OPTIMIZER = {
    "preference": "SimPO",                                      # Preference optimization method: 'SimPO' | 'NCA'

    "gradient": {
        "lr": 3e-5,                                             # Base learning rate
        "eps": 1e-6,                                            # Numerical stability term for AdamW optimizer
        "betas": (0.9, 0.95),                                   # Betas for AdamW optimizer
        "weight_decay": 0.1                                     # Regularization parameter to prevent overfitting
    }
}

# Data Handling
LOADER = {
    "truncation_mode": "keep_start"                             # Sequence truncation strategy: 'keep_start' | 'keep_end'
}

# Numerical Precision
PRECISION = "bf16-mixed"                                        # Mixed-precision training mode: '16-mixed' | 'bf16-mixed'

# Monitoring
LOG = True
