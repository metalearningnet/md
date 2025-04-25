MODEL = {
    # Language Model Configuration
    "lm": {
        "name": "Qwen2.5-0.5B",
        "site": "Qwen/Qwen2.5-0.5B",
        "checkpoint": True  # Gradient checkpointing to reduce memory usage
    },

    # Skill Memory Configuration
    "skill": {
        # ===== Memory Configuration =====
        "mac_persistent_mem_tokens": 2,  # Number of tokens allocated for persistent memory in MAC (Memory as a Context) architecture.
        "mac_longterm_mem_tokens": 2,  # Number of tokens allocated for long-term memory in MAC architecture.
        "mac_depth": 1,  # Depth of the MAC architecture.
        "mac_segment_len": 2,  # Length of segments processed by the MAC architecture.
        "mac_neural_memory_qkv_receives_diff_views": False,  # Allow Q/K/V to come from different views/layers.
        "mac_neural_mem_weight_residual": False,  # Add residual connections between memory weight updates.

        # ===== Loss Coefficients =====
        "mi_coeff": 1.0,  # Coefficient for mutual information loss to encourage diverse memory usage.
        "entropy_coeff": 0.1,  # Coefficient for entropy regularization to encourage exploration.
        "adv_coeff": 0.5,  # Coefficient for adversarial learning.
        "kl_coeff": 0.01,  # Coefficient for KL divergence penalty to stabilize policy updates.
        
        # ===== Gradient Checkpointing =====
        "checkpoint": {  # Memory-for-compute tradeoff settings
            "mac": True,  # Checkpoint MAC layers
            "policy": True,  # Checkpoint policy network
            "prior": True,  # Checkpoint skill prior network
            "discriminators": True  # Checkpoint mutual information discriminators
        }
    },

    # Loss Balancing Coefficients
    "lm_coef": 0.7,  # Language model loss weight
    "skill_coef": 0.3,  # Skill memory loss weight

    "use_cache": False
}

# Optimizer Configuration
OPTIMIZER = {
    "preference": "SimPO"
}

# Data Loader Configuration
LOADER = {
    "max_length": 32,  # Maximum sequence length for input data.
    "max_prompt_length": 16,  # Maximum length allowed for the prompt.
    "truncation_mode": "keep_end"  # Truncation strategy: 'keep_end' retains the end, 'keep_start' keeps the beginning.
}

# Accelerator Configuration
ACCELERATOR = "auto"  # Options: "cpu", "gpu", "tpu", or "mps".

# Precision Configuration
PRECISION = "bf16-mixed"  # Options: "16-mixed", "32-true".
