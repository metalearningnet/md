import torch.nn as nn
from utils import info
from titans import MemoryAsContextTransformer

class Frontend(nn.Module):
    def __init__(self,
                 mem_type: str = 'mac',
                 state_dim: int = 256,
                 num_tokens: int = 1024,
                 update_memory: bool = True,
                 mem_config: dict = dict()):
        super().__init__()

        info(f"Frontend (mem_type: {mem_type}, state_dim: {state_dim})")
        
        self.state_dim = state_dim

        if 'mac' == mem_type:
            mac_kwargs = mem_config
            mac_depth = mac_kwargs.get('depth', 1)
            mac_segment_len = mac_kwargs.get('segment_len', 32)
            mac_use_flex_attn = mac_kwargs.get('use_flex_attn', False)
            mac_longterm_mem_tokens = mac_kwargs.get('longterm_mem_tokens', 32)
            mac_persistent_mem_tokens = mac_kwargs.get('persistent_mem_tokens', 16)
            mac_sliding_window_attn = mac_kwargs.get('sliding_window_attn', False)
            mac_neural_mem_heads = mac_kwargs.get('neural_mem_heads', 4)
            mac_neural_mem_head_dim = mac_kwargs.get('neural_mem_head_dim', 64)
            mac_neural_mem_batch_size = mac_kwargs.get('neural_mem_batch_size')
            mac_neural_mem_momentum = mac_kwargs.get('neural_mem_momentum', True)
            mac_neural_mem_momentum_order = mac_kwargs.get('neural_mem_momentum_order', 1)
            mac_neural_mem_qk_rmsnorm = mac_kwargs.get('neural_mem_qk_rmsnorm', True)
            mac_manual_per_sample_grads = mac_kwargs.get('manual_per_sample_grads', False)
            mac_neural_mem_weight_residual = mac_kwargs.get('neural_mem_weight_residual', True)
            mac_neural_mem_attn_pool_chunks = mac_kwargs.get('neural_mem_attn_pool_chunks', True)
            mac_neural_mem_use_accelerated_scan = mac_kwargs.get('neural_mem_use_accelerated_scan', False)
            mac_neural_mem_step_transform_max_lr = mac_kwargs.get('neural_mem_step_transform_max_lr', 1e-1)
            mac_neural_mem_qkv_receives_diff_views = mac_kwargs.get('neural_mem_qkv_receives_diff_views', True)
            mac_neural_mem_spectral_norm_surprises = mac_kwargs.get('neural_mem_spectral_norm_surprises', True)
            mac_neural_mem_per_head_learned_parameters = mac_kwargs.get('neural_mem_per_head_learned_parameters', False)
            mac_neural_mem_per_parameter_lr_modulation = mac_kwargs.get('neural_mem_per_parameter_lr_modulation', True)
            
            self.mem = MemoryAsContextTransformer(
                dim=state_dim,
                token_emb=None,
                depth=mac_depth,
                num_tokens=num_tokens,
                segment_len=mac_segment_len,
                use_flex_attn=mac_use_flex_attn,
                sliding_window_attn=mac_sliding_window_attn,
                num_persist_mem_tokens=mac_persistent_mem_tokens,
                num_longterm_mem_tokens=mac_longterm_mem_tokens,
                neural_mem_batch_size=mac_neural_mem_batch_size,
                neural_mem_weight_residual=mac_neural_mem_weight_residual,
                neural_mem_qkv_receives_diff_views=mac_neural_mem_qkv_receives_diff_views,
                neural_mem_kwargs = dict(
                    heads=mac_neural_mem_heads,
                    update_memory=update_memory,
                    dim_head=mac_neural_mem_head_dim,
                    momentum=mac_neural_mem_momentum,
                    qk_rmsnorm=mac_neural_mem_qk_rmsnorm,
                    momentum_order=mac_neural_mem_momentum_order,
                    attn_pool_chunks=mac_neural_mem_attn_pool_chunks,
                    manual_per_sample_grads=mac_manual_per_sample_grads,
                    use_accelerated_scan=mac_neural_mem_use_accelerated_scan,
                    spectral_norm_surprises=mac_neural_mem_spectral_norm_surprises,
                    default_step_transform_max_lr=mac_neural_mem_step_transform_max_lr,
                    per_head_learned_parameters=mac_neural_mem_per_head_learned_parameters,
                    per_parameter_lr_modulation=mac_neural_mem_per_parameter_lr_modulation
                )
            )
        else:
            raise ValueError(f"No memory configuration")

    def forward(self, states):
        return self.mem(states)
