MODEL = {
    "name": "Qwen2.5-0.5B-Instruct",
    "site": "Qwen/Qwen2.5-0.5B-Instruct"
}

SKILL_MEMORY = {
    "action_dim": 2,
    "state_embed_dim": 2,
    "hidden_dim": 2,
    "mac_persistent_mem_tokens": 2,
    "mac_longterm_mem_tokens": 2,
    "mac_depth": 1,
    "mac_segment_len": 2,
    "mi_coeff": 1.0,
    "entropy_coeff": 0.1,
    "adv_coeff": 0.5,
    "kl_coeff": 0.01
}

LOADER = {
    "max_length": 4,
    "state_window": 2
}

EPOCHS = 1
BATCH_SIZE = 1
