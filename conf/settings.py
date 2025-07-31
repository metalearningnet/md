MODEL = {
    # Language Model Configuration
    "lm": {
        "path": "Qwen/Qwen3-0.6B",         # Identifier for the pretrained language model (from HuggingFace)
        "temperature": 0.7,                # Controls randomness in token sampling (higher = more diverse outputs)
        "freeze": True,                    # Whether to freeze the pretrained LM weights during training
        "max_length": 384,                 # Maximum token length for input + output sequences
        "max_target_length": 256,          # Maximum token length allowed for output sequences
        "max_prompt_length": 128           # Max tokens allowed in prompt before truncation
    },

    # Skill Memory Configuration
    "skill": {
        "mem_type": "mac",                 # MAC (Memory as Context)

        # Loss Balancing Coefficients
        "mi_coef": 0.5,                    # Weight for mutual information maximization
        "kl_coef": 0.01,                   # Controls KL divergence penalty for prior-policy alignment
        "adv_coef": 0.1,                   # Scales adversarial loss component
        "entropy_coef": 0.3,               # Encourages exploration via policy entropy regularization
        "forward_coef": 0.2,               # Weight for forward prediction consistency loss

        # Action Space Configuration
        "action_dim": 128                  # Dimensionality of action embeddings
    },
    
    # Training Objective Weights
    "lm_coef": 1.0,                        # Proportional weight for language modeling loss
    "skill_coef": 0.05,                    # Proportional weight for skill learning objectives (0.0 = pure LM)

    # Integration Strategy for Skill Output into the Language Model
    "context_window": 4,                   # Lookahead window size for determining insertion position of skill output
    "skill_integration_strategy": "hint",  # Options: 'fusion' | 'annotation' | 'hint'

    # Inference-Time Behavior
    "use_cache": False,                    # Enable KV caching for faster autoregressive decoding
    "update_memory": True                  # Allow memory updates during inference
}

# Memory Architecture
MEMORY = {
    "mac": {
        "depth": 1,                                      # Number of stacked MAC blocks
        "segment_len": 32,                               # Segment length processed per MAC block
        "longterm_mem_tokens": 32,                       # Tokens allocated for extended memory retention
        "persistent_mem_tokens": 16,                     # Tokens reserved for long-lasting contextual memory

        "use_flex_attn": False,                          # Use FlexAttention for potentially faster/sliding window attention
        "sliding_window_attn": True,                     # Use sliding window attention for efficiency

        "neural_mem_heads": 4,                           # Number of attention heads in neural memory
        "neural_mem_head_dim": 64,                       # Dimension per head in neural memory
        "neural_mem_batch_size": 64,                     # set smaller to update the neural memory weights more often as it traverses the sequence
        "neural_mem_momentum_order": 1,                  # Highest order of momentum to calculate
        "neural_mem_step_transform_max_lr": 1e-1,        # Upper limit for the effective learning rate used in memory updates

        "neural_mem_momentum": True,                     # Enable momentum-based memory updates
        "neural_mem_qk_rmsnorm": True,                   # Apply RMSNorm to Q/K projections
        "neural_mem_weight_residual": True,              # Adds residual connections in neural memory weight updates
        "neural_mem_attn_pool_chunks": True,             # whether to use attention pooling for chunk derived momentum
        "neural_mem_use_accelerated_scan": False,        # Use an optimized version of the associative scan for faster computation
        "neural_mem_qkv_receives_diff_views": True,      # Q/K/V projections come from different layers/views
        "neural_mem_spectral_norm_surprises": True,      # Apply spectral normalization to memory updates
        "neural_mem_per_parameter_lr_modulation": True,  # Parameter-specific learning rate adaptation
        "neural_mem_per_head_learned_parameters": False  # Independent parameters per memory head
    }
}

# Fusion Settings (used when skill_integration_strategy == 'fusion')
FUSION = {
    # Skill-LM Adapter Configuration
    "adapter": {
        "proj_scale": 2,                   # Expansion factor for intermediate adapter dimensions
        "proj_dropout": 0.1,               # Dropout rate applied after adapter projections
        "min_proj_dim": 32,                # Minimum hidden dimension size in adapter layers
        "norm_position": "post"            # Position of LayerNorm: 'pre' (before) or 'post' (after) activation
    }
}

# Annotation Generation Settings (used when skill_integration_strategy == 'annotation')
ANNOTATION = {
    "words": 8,                            # Number of distinct word types allowed per annotation
    "max_length": 2,                       # Max token length per annotation
    "max_annotations": 4,                  # Max number of annotations per response (-1 for unlimited)
    "min_interval": 128,                   # Minimum token distance between annotations
    "tune": True                           # Whether annotation embeddings are fine-tuned during training
}

# Hint Generation Settings (used when skill_integration_strategy == 'hint')
HINT = {
    "category": "minimal",                 # Options: 'minimal' | 'standard' | 'enhanced' | 'advanced'
    "max_hints": 16,                       # Max number of hints per response (-1 for unlimited)
    "min_interval": 16,                    # Minimum token distance between hints
    "tune": True                           # Whether hint token embeddings are trainable
}

# Checkpointing Configuration
CKPT = {
    # Enables gradient checkpointing to reduce GPU memory usage
    "gradient": {
        "lm": False,                       # Gradient checkpointing for the Language Model
        "skill": False                     # Skill Memory
    }
}

# Optimization Configuration
OPTIMIZER = {
    # Preference optimization method
    "preference": "SimPO",                 # Options: 'SimPO' | 'NCA'

    # Gradient optimizer settings
    "gradient": {
        "lr": 3e-5,                        # Base learning rate
        "betas": (0.9, 0.98),              # Betas for AdamW optimizer
        "weight_decay": 0.05               # Regularization parameter to prevent overfitting
    }
}

# Data Processing Configuration
LOADER = {
    "truncation_mode": "keep_start"        # Truncation strategy; Options:  'keep_start' | 'keep_end'
}

# Numerical Precision Setting
PRECISION = "bf16-mixed"                   # Floating-point precision; Options: '16-mixed' | 'bf16-mixed'

# Logging Configuration
LOG = True
