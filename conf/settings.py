MODEL = {
    # Language Model Configuration
    "lm": {
        "path": "google/gemma-3-1b-it",    # Identifier for the pretrained language model (from HuggingFace)
        "temperature": 0.7,                # Controls randomness in token sampling (higher = more diverse outputs)
        "freeze": False,                   # Whether to freeze the pretrained LM weights during training
        "max_length": 256,                 # Maximum token length for input + output sequences
        "max_target_length": 128,          # Maximum token length allowed for output sequences
        "max_prompt_length": 128           # Max tokens allowed in prompt before truncation
    },

    # Skill Memory Configuration
    "skill": {
        # Memory Architecture Parameters (MAC - Memory as Context)
        "mac_persistent_mem_tokens": 32,   # Tokens reserved for long-lasting contextual memory
        "mac_longterm_mem_tokens": 32,     # Tokens allocated for extended memory retention
        "mac_depth": 1,                    # Number of stacked MAC blocks
        "mac_segment_len": 128,            # Segment length processed per MAC block
        "mac_neural_memory_qkv_receives_diff_views": True,  # If True, Q/K/V projections come from different layers/views
        "mac_neural_mem_weight_residual": True,  # Adds residual connections in neural memory weight updates

        # Loss Balancing Coefficients
        "mi_coef": 1.0,             # Weight for mutual information maximization
        "entropy_coef": 0.5,        # Encourages exploration via policy entropy regularization
        "adv_coef": 0.5,            # Scales adversarial loss component
        "kl_coef": 0.05,            # Controls KL divergence penalty for prior-policy alignment
        "forward_coef": 0.1,        # Weight for forward prediction consistency loss

        # Action Space Configuration
        "action_dim": 64            # Dimensionality of action embeddings
    },
    
    # Training Objective Weights
    "lm_coef": 1.0,                 # Proportional weight for language modeling loss
    "skill_coef": 0.25,             # Proportional weight for skill learning objectives (0.0 = pure LM)

    # Integration Strategy for Skill Output into the Language Model
    "context_window": 4,                   # Lookahead window size for determining insertion position of skill output
    "skill_integration_strategy": "hint",  # Options: 'fusion' | 'annotation' | 'hint'

    # Inference Behavior
    "use_cache": False              # Use KV caching to accelerate autoregressive generation
}

# Fusion Settings (used when skill_integration_strategy == 'fusion')
FUSION = {
      # Skill-LM Adapter Configuration
    "adapter": {
        "min_proj_dim": 32,         # Minimum hidden dimension size in adapter layers
        "proj_scale": 2,            # Expansion factor for intermediate adapter dimensions
        "proj_dropout": 0.1,        # Dropout rate applied after adapter projections
        "norm_position": "post"     # Position of LayerNorm: 'pre' (before) or 'post' (after) activation
    }
}

# Annotation Generation Settings (used when skill_integration_strategy == 'annotation')
ANNOTATION = {
    "words": 8,                     # Number of distinct word types allowed per annotation
    "max_length": 2,                # Max token length per annotation
    "max_annotations": 4,           # Max number of annotations per response (-1 for unlimited)
    "min_interval": 128             # Minimum token distance between annotations
}

# Hint Generation Settings (used when skill_integration_strategy == 'hint')
HINT = {
    "category": "minimal",          # Options: 'minimal' | 'standard' | 'enhanced' | 'advanced'
    "max_hints": 16,                # Max number of hints per response (-1 for unlimited)
    "min_interval": 32              # Minimum token distance between hints
}

# Checkpointing Configuration
CKPT = {
    # Enables gradient checkpointing to reduce GPU memory usage
    "gradient": {
        "lm": False,                # Gradient checkpointing for the Language Model
        "skill": False              # Skill Memory
    }
}

# Optimization Configuration
OPTIMIZER = {
    # Preference optimization method
    "preference": "SimPO",          # Options: 'SimPO' | 'NCA'
    
    # Gradient optimizer settings
    "gradient": {
        "lr": 1e-4,                 # Base learning rate
        "betas": (0.9, 0.98),       # Betas for AdamW optimizer
        "weight_decay": 0.01        # Regularization parameter to prevent overfitting
    }
}

# Data Processing Configuration
LOADER = {
    "truncation_mode": "keep_end"   # Truncation strategy; Options: 'keep_end' | 'keep_start'
}

# Numerical Precision Setting
PRECISION = "bf16-mixed"            # Floating-point precision; Options: '16-mixed' | 'bf16-mixed'

# Logging Configuration
LOG = True
