MODEL = {
    # Language Model Configuration
    "lm": {
        "name": "Qwen2.5-0.5B",
        "path": "Qwen/Qwen2.5-0.5B",
        "checkpoint": True  # Gradient checkpointing to reduce memory usage
    },

    # Skill Memory Configuration
    "skill": {
        # Memory Architecture
        "mac_persistent_mem_tokens": 128,  # Number of tokens allocated for persistent memory in MAC (Memory as a Context) architecture.
        "mac_longterm_mem_tokens": 128,  # Number of tokens allocated for long-term memory in MAC architecture.
        "mac_depth": 4,  # Depth of the MAC architecture.
        "mac_segment_len": 512,  # Length of segments processed by the MAC architecture.
        "mac_neural_memory_qkv_receives_diff_views": False,  # Allow Q/K/V to come from different views/layers.
        "mac_neural_mem_weight_residual": False,  # Add residual connections between memory weight updates.

        # Loss Balancing
        "mi_coeff": 1.0,  # Mutual information weight
        "entropy_coeff": 0.5,  # Exploration bonus
        "adv_coeff": 0.5,  # Adversarial loss scale
        "kl_coeff": 0.05,  # KL divergence penalty
        "forward_coeff": 0.1,  # Forward loss coefficient

        # Dimensionality
        "action_dim": 8192,
        
        # Memory Optimization
        "checkpoint": {  # Gradient checkpointing
            "mac": True,  # MAC layers
            "policy": True,  # Policy network
            "prior": True,  # Skill prior 
            "discriminators": True  # MI discriminators
        }
    },

    # Skill-LM Adapter
    "adapter" : {
        "min_proj_dim": 32,  # Min hidden dimension
        "proj_scale": 2,  # Multiplier for initial expansion
        "proj_dropout": 0.1,  # Dropout rate (regularization)
        "norm_position": "post"  # LayerNorm order: ('post'|'pre')
    },
    
    # Training Objectives
    "lm_coef": 0.95,  # Controls LM loss contribution
    "skill_coef": 0.05,  # Balances skill learning (0.0 = pure LM)

    # Inference Settings
    "use_cache": False  # Enables caching for faster generation
}

# Optimization Strategy
OPTIMIZER = {
    "preference": "SimPO"  # Options: SimPO (default) | NCA
}

# Data Processing
LOADER = {
    "max_length": 512,  # Max input token limit
    "max_prompt_length": 256,  # Prompt truncation threshold
    "truncation_mode": "keep_end"  # Input trimming: keep_end/start
}

# Hardware Setup
ACCELERATOR = "auto"  # Options: auto|cpu|gpu|mps 

# Numerical Precision
PRECISION = "bf16-mixed"  # Options: 32-true|16-mixed|bf16-mixed
