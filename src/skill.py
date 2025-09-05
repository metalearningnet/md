import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import info, get_device
from torch.utils.checkpoint import checkpoint
from torch.distributions import Normal, kl_divergence
from titans import MemoryAsContextTransformer, NeuralMemory

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, memory):
        x = torch.cat([state, memory], dim=-1)
        batch, seq_len, _ = x.shape
        x = x.reshape(-1, x.size(-1))

        h = self.trunk(x)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)

        mean = mean.view(batch, seq_len, -1)
        log_std = log_std.view(batch, seq_len, -1)

        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

class SkillMemory(nn.Module):
    def __init__(self,
                 mem_type: str = 'mac',
                 state_dim: int = 256,
                 num_tokens: int = 1024,
                 action_dim: int = 64,
                 hidden_dim: int = 128,
                 mi_coef: float = 0.5,
                 kl_coef: float = 0.01,
                 adv_coef: float = 0.1,
                 entropy_coef: float = 0.3,
                 checkpoint: bool = False,
                 update_memory: bool = True,
                 mem_config: dict = dict()):
        super().__init__()

        info(f"Skill memory (mem: {mem_type}, state_dim: {state_dim}, action_dim: {action_dim}, hidden_dim: {hidden_dim})")
        
        self.mem_type = mem_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.checkpoint = checkpoint

        self.device = get_device()
        
        if 'mac' == mem_type:
            mac_depth = mem_config.get('depth', 1)
            mac_segment_len = mem_config.get('segment_len', 32)
            mac_use_flex_attn = mem_config.get('use_flex_attn', False)
            mac_longterm_mem_tokens = mem_config.get('longterm_mem_tokens', 32)
            mac_sliding_window_attn = mem_config.get('sliding_window_attn', False)
            mac_persistent_mem_tokens = mem_config.get('persistent_mem_tokens', 16)
            mac_manual_per_sample_grads = mem_config.get('manual_per_sample_grads', False)
            
            mac_neural_mem_heads = mem_config.get('neural_mem_heads', 4)
            mac_neural_mem_head_dim = mem_config.get('neural_mem_head_dim', 64)
            mac_neural_mem_batch_size = mem_config.get('neural_mem_batch_size')
            mac_neural_mem_momentum = mem_config.get('neural_mem_momentum', True)
            mac_neural_mem_qk_rmsnorm = mem_config.get('neural_mem_qk_rmsnorm', True)
            mac_neural_mem_momentum_order = mem_config.get('neural_mem_momentum_order', 1)
            mac_neural_mem_weight_residual = mem_config.get('neural_mem_weight_residual', True)
            mac_neural_mem_attn_pool_chunks = mem_config.get('neural_mem_attn_pool_chunks', True)
            mac_neural_mem_gated_transition = mem_config.get('neural_mem_gated_transition', False)
            mac_neural_mem_use_accelerated_scan = mem_config.get('neural_mem_use_accelerated_scan', False)
            mac_neural_mem_step_transform_max_lr = mem_config.get('neural_mem_step_transform_max_lr', 1e-1)
            mac_neural_mem_qkv_receives_diff_views = mem_config.get('neural_mem_qkv_receives_diff_views', True)
            mac_neural_mem_spectral_norm_surprises = mem_config.get('neural_mem_spectral_norm_surprises', True)
            mac_neural_mem_model_norm_add_residual = mem_config.get('neural_mem_model_norm_add_residual', True)
            mac_neural_mem_per_head_learned_parameters = mem_config.get('neural_mem_per_head_learned_parameters', False)
            mac_neural_mem_per_parameter_lr_modulation = mem_config.get('neural_mem_per_parameter_lr_modulation', True)
            
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
                    gated_transition=mac_neural_mem_gated_transition,
                    attn_pool_chunks=mac_neural_mem_attn_pool_chunks,
                    manual_per_sample_grads=mac_manual_per_sample_grads,
                    use_accelerated_scan=mac_neural_mem_use_accelerated_scan,
                    step_transform_max_lr=mac_neural_mem_step_transform_max_lr,
                    spectral_norm_surprises=mac_neural_mem_spectral_norm_surprises,
                    mem_model_norm_add_residual=mac_neural_mem_model_norm_add_residual,
                    per_head_learned_parameters=mac_neural_mem_per_head_learned_parameters,
                    per_parameter_lr_modulation=mac_neural_mem_per_parameter_lr_modulation
                )
            )
        elif 'mal' == mem_type:
            heads = mem_config.get('heads', 4)
            dim_head = mem_config.get('dim_head', 64)
            chunk_size = mem_config.get('chunk_size', 32)
            momentum_order = mem_config.get('momentum_order', 1)
            step_transform_max_lr = mem_config.get('step_transform_max_lr', 0.1)

            momentum = mem_config.get('momentum', True)
            qk_rmsnorm = mem_config.get('qk_rmsnorm', True)
            attn_pool_chunks = mem_config.get('attn_pool_chunks', True)
            gated_transition = mem_config.get('gated_transition', False)
            use_accelerated_scan = mem_config.get('use_accelerated_scan', False)
            accept_weight_residual = mem_config.get('accept_weight_residual', False)
            spectral_norm_surprises = mem_config.get('spectral_norm_surprises', True)
            qkv_receives_diff_views = mem_config.get('qkv_receives_diff_views', False)
            manual_per_sample_grads = mem_config.get('manual_per_sample_grads', False)
            mem_model_norm_add_residual = mem_config.get('mem_model_norm_add_residual', True)
            per_parameter_lr_modulation = mem_config.get('per_parameter_lr_modulation', True)
            per_head_learned_parameters = mem_config.get('per_head_learned_parameters', True)

            self.mem = NeuralMemory(
                dim=state_dim,
                heads=heads,
                dim_head=dim_head,
                chunk_size=chunk_size,
                momentum_order=momentum_order,
                step_transform_max_lr=step_transform_max_lr,
                momentum=momentum,
                qk_rmsnorm=qk_rmsnorm,
                update_memory=update_memory,
                attn_pool_chunks=attn_pool_chunks,
                gated_transition=gated_transition,
                use_accelerated_scan=use_accelerated_scan,
                accept_weight_residual=accept_weight_residual,
                spectral_norm_surprises=spectral_norm_surprises,
                qkv_receives_diff_views=qkv_receives_diff_views,
                manual_per_sample_grads=manual_per_sample_grads,
                mem_model_norm_add_residual=mem_model_norm_add_residual,
                per_parameter_lr_modulation=per_parameter_lr_modulation,
                per_head_learned_parameters=per_head_learned_parameters
            )
        else:
            raise ValueError(f"No memory configuration")
        
        self.mem_output_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.memory_var = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.uniform_(self.memory_var, -1, 1)

        self.policy = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        )

        # Skill-Conditioned Prior
        self.prior_net = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.prior_mean = nn.Linear(hidden_dim, hidden_dim)
        self.prior_std = nn.Linear(hidden_dim, hidden_dim)
        
        # Mutual Information Discriminator (I(S;M))
        self.disc_gru = nn.GRU(state_dim + hidden_dim, hidden_dim, batch_first=True)
        self.disc_linear = nn.Linear(hidden_dim, 1)

        # Forward Model (I(S;M))
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Adversarial Discriminator (I(A;M|S))
        self.mmi_discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        self.mi_coef = mi_coef
        self.kl_coef = kl_coef
        self.adv_coef = adv_coef
        self.entropy_coef = entropy_coef
        self.calc_kl = kl_coef != 0.0
    
    def get_action_logits(self, states, m):
        mean, log_std = self.policy(states, m)
        std = log_std.exp()
        dist = Normal(mean, std)
        action_logits = dist.rsample()
        entropy = dist.entropy().mean()
        return action_logits, entropy

    def get_prior(self, m):
        batch, seq_len, hidden_dim = m.shape
        fixed_prior_mean = torch.zeros(1, hidden_dim, device=m.device)
        fixed_prior_std = torch.ones(1, hidden_dim, device=m.device)
        fixed_prior_mean = fixed_prior_mean.expand(batch, 1, -1)
        fixed_prior_std = fixed_prior_std.expand(batch, 1, -1)
        if seq_len == 1:
            return fixed_prior_mean, fixed_prior_std
        prior_out, _ = self.prior_net(m[:, :-1])
        prior_mean = self.prior_mean(prior_out)
        prior_std = F.softplus(self.prior_std(prior_out)) + 1e-4
        return prior_mean, prior_std
    
    def get_mem(self, states):
        if states.size(1) == 0:
            raise AssertionError("seq_len > 0")
        mem_output = self.mem(states)
        if self.mem_type == 'mal':
            mem_output = mem_output[0]
        return self.mem_output_proj(mem_output), mem_output
    
    def forward(self, states):
        if self.checkpoint:
            m, mem_output = checkpoint(self.get_mem, states, use_reentrant=True)
            if self.calc_kl:
                prior_mean, prior_std = checkpoint(self.get_prior, m, use_reentrant=True)
        else:
            m, mem_output = self.get_mem(states)
            if self.calc_kl:
                prior_mean, prior_std = self.get_prior(m)
        
        if self.calc_kl:
            prior_dist = Normal(prior_mean, prior_std)
            mem_std = F.softplus(self.memory_var)
            mem_std = torch.clamp(mem_std, min=0.1, max=2.0)
            if m.size(1) > 1:
                mem_dist = Normal(m[:, 1:], mem_std.expand_as(m[:, 1:]))
            else:
                mem_dist = Normal(m, mem_std.expand_as(m))
        else:
            mem_dist = None
            prior_dist = None
        
        action_logits, entropy = self.get_action_logits(states, m)

        return {
            'm': m,
            'states': states,
            'entropy': entropy,
            'mem_dist': mem_dist,
            'prior_dist': prior_dist,
            'mem_output': mem_output,
            'action_logits': action_logits
        }
    
    def compute_losses(self, outputs):
        m = outputs['m']
        states = outputs['states']
        entropy = outputs['entropy']
        mem_dist = outputs['mem_dist']
        mem_output = outputs['mem_output']
        prior_dist = outputs['prior_dist']
        action_logits = outputs['action_logits']
        
        # I(S; M)
        pos_pairs = torch.cat([states, m], dim=-1)
        neg_m = m[torch.randperm(m.size(0))]
        neg_pairs = torch.cat([states, neg_m], dim=-1)

        # Discriminator scores
        pos_scores = self.disc_linear(self.disc_gru(pos_pairs)[0][:, -1])
        neg_scores = self.disc_linear(self.disc_gru(neg_pairs)[0][:, -1])

        # InfoNCE loss
        mi_disc_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # Policy loss (maximize I(S; M))
        mi_policy_loss = -pos_scores.mean()

        # Forward model loss
        assert states.shape[1] >= 1 and m.shape[1] >= 1, "Sequence length must be > 0"
        forward_input = torch.cat([states[:, :-1], m[:, :-1]], dim=-1)
        pred_states = self.forward_model(forward_input)
        forward_loss = F.mse_loss(pred_states, states[:, 1:])

        mi_loss = mi_disc_loss + mi_policy_loss + forward_loss

        # I(A;M|S)
        real_input = torch.cat([states, action_logits], dim=-1)
        real_input = GradientReversal.apply(real_input)
        pred_m = self.mmi_discriminator(real_input)
        adv_loss = F.mse_loss(pred_m, mem_output.detach())

        # KL Regularization
        if self.calc_kl:
            kl_loss = kl_divergence(mem_dist, prior_dist)
            kl_loss = torch.clamp(kl_loss, max=1e3)
            kl_loss = kl_loss.mean()
        else:
            kl_loss = 0.0

        total_loss = (
            self.mi_coef * mi_loss +
            self.entropy_coef * -entropy +
            self.adv_coef * adv_loss +
            self.kl_coef * kl_loss
        )
        
        return {
            'total_loss': total_loss
        }
