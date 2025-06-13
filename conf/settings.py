MODEL = {
    # Language Model Configuration
    "lm": {
        "path": "Qwen/Qwen2.5-0.5B",       # Pretrained language model identifier
        "temperature": 0.7                 # Sampling temperature for generation
    },

    # Skill Memory Configuration
    "skill": {
        # Memory Architecture Parameters (MAC - Memory as Context)
        "mac_persistent_mem_tokens": 64,   # Tokens reserved for long-lasting contextual memory
        "mac_longterm_mem_tokens": 64,     # Tokens allocated for extended memory retention
        "mac_depth": 1,                    # Number of stacked MAC blocks
        "mac_segment_len": 256,            # Segment length processed per MAC block
        "mac_neural_memory_qkv_receives_diff_views": False,  # If True, Q/K/V projections come from different layers/views
        "mac_neural_mem_weight_residual": False,  # Adds residual connections in neural memory weight updates

        # Loss Balancing Coefficients
        "mi_coef": 1.0,             # Weight for mutual information maximization
        "entropy_coef": 0.5,        # Encourages exploration via policy entropy regularization
        "adv_coef": 0.5,            # Scales adversarial loss component
        "kl_coef": 0.05,            # Controls KL divergence penalty for prior-policy alignment
        "forward_coef": 0.1,        # Weight for forward prediction consistency loss

        # Action Space Configuration
        "action_dim": 64,           # Dimensionality of action embeddings
    },

    # Skill-LM Adapter Configuration
    "adapter": {
        "min_proj_dim": 32,         # Minimum hidden dimension size in adapter layers
        "proj_scale": 2,            # Expansion factor for intermediate adapter dimensions
        "proj_dropout": 0.1,        # Dropout rate applied after adapter projections
        "norm_position": "post"     # Position of LayerNorm: 'pre' (before) or 'post' (after) activation
    },
    
    # Training Objective Weights
    "lm_coef": 1,                   # Proportional weight for language modeling loss
    "skill_coef": 0.05,             # Proportional weight for skill learning objectives (0.0 = pure LM)

    # Integration Strategy for Skill Output into the Language Model
    "skill_integration_strategy": "annotation",  # Options: ['fusion' | 'annotation']

    # Inference Behavior
    "use_cache": False              # Use KV caching to accelerate autoregressive generation
}

# Annotation Generation Settings (used when skill_integration_strategy == 'annotation')
ANNOTATION = {
    "words": 5,                     # Number of distinct word types allowed per annotation
    "max_length": 3,                # Max token length per annotation
    "temperature": 0.7,             # Sampling temperature for annotations
    "max_annotations": 2,           # Max number of annotations per response (-1 for unlimited)
    "trigger_sharpness": 0.5        # Controls the sharpness of the sampling distribution when deciding whether to trigger an annotation (lower = more deterministic, higher = more exploratory)
}

# Checkpointing Configuration
CKPT = {
    # Enables gradient checkpointing to reduce GPU memory usage
    "gradient": {
        "lm": False,                # LLM
        "skill": {                  # Skill Memory
            "mac": False,           #   MAC layers
            "policy": False,        #   Policy network
            "prior": False,         #   Skill prior model
            "discriminators": False #   MI discriminators
        }
    }
}

# Optimization Strategy
OPTIMIZER = {
    "preference": "SimPO"           # Preference optimization method; Options: SimPO (default) | NCA
}

# Data Processing Pipeline
LOADER = {
    "max_length": 512,              # Maximum token length for input + output sequences
    "max_prompt_length": 128,       # Max tokens allowed in prompt before truncation
    "truncation_mode": "keep_end"   # Truncation strategy: 'keep_end' (preferred) or 'keep_start'
}

# Numerical Precision Setting
PRECISION = "bf16-mixed"            # Floating-point precision; Options: 16-mixed | bf16-mixed

# Logging Configuration (enables detailed training logs)
LOG = False
