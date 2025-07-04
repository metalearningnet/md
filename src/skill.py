import math
import torch
import torch.nn as nn
from utils import info
import torch.nn.functional as F
from titans import MemoryAsContextTransformer
from torch.utils.checkpoint import checkpoint
from torch.distributions import Normal, kl_divergence

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
        self.input_layer = nn.Linear(state_dim + hidden_dim, hidden_dim * 4)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim * 4, hidden_dim * 4) for _ in range(2)
        ])
        self.output_layer = nn.Linear(hidden_dim * 4, action_dim)
        self.ln = nn.LayerNorm(hidden_dim * 4)
        self.act = nn.SiLU()

        nn.init.orthogonal_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(self.act(self.ln(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.act = nn.SiLU()
        self.shortcut = nn.Identity() if in_features == out_features else \
                       nn.Linear(in_features, out_features)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.ln1(self.linear1(x)))
        x = self.ln2(self.linear2(x))
        return self.act(x + residual)

class SkillMemory(nn.Module):
    def __init__(self,
                 state_dim: int = 1152,
                 num_tokens: int = 262152,
                 action_dim: int = 128,
                 hidden_dim: int = 1152,
                 mac_persistent_mem_tokens: int = 16,
                 mac_longterm_mem_tokens: int = 48,
                 mac_depth: int = 2,
                 mac_segment_len: int = 32,
                 mac_neural_memory_qkv_receives_diff_views: bool = False,
                 mac_neural_mem_weight_residual: bool = False,
                 mi_coef: float = 0.8,
                 entropy_coef: float = 0.3,
                 adv_coef: float = 0.3,
                 kl_coef: float = 0.05,
                 forward_coef: float = 0.2,
                 checkpoint: bool = False):
        
        super().__init__()
        info(f"Skill memory (state_size: {state_dim} action_size: {action_dim})")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.checkpoint = checkpoint
        
        self.mac = MemoryAsContextTransformer(
            num_tokens=num_tokens,
            dim=hidden_dim,
            depth=mac_depth,
            segment_len=mac_segment_len,
            num_persist_mem_tokens=mac_persistent_mem_tokens,
            num_longterm_mem_tokens=mac_longterm_mem_tokens,
            neural_memory_qkv_receives_diff_views=mac_neural_memory_qkv_receives_diff_views,
            neural_mem_weight_residual=mac_neural_mem_weight_residual,
            token_emb=None
        )
        self.mac_output_proj = nn.Linear(num_tokens, hidden_dim)
        self.memory_var = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.constant_(self.memory_var, math.log(0.5))

        self.policy = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        )
        if isinstance(self.policy.output_layer, nn.Linear):
            self.policy.output_layer.weight.sparse_grad = True
        else:
            self.policy.output_layer[-1].weight.sparse_grad = True

        # Skill-Conditioned Prior
        self.prior_net = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.prior_mean = nn.Linear(hidden_dim, hidden_dim)
        self.prior_std = nn.Linear(hidden_dim, hidden_dim)
        
        # Mutual Information Discriminator (I(S;M))
        self.disc_gru = nn.GRU(state_dim + hidden_dim, hidden_dim, batch_first=True)
        self.disc_linear = nn.Linear(hidden_dim, 1)

        # Forward Model (for I(S;M))
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Adversarial Discriminator (I(A;M|S))
        self.mmi_discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.rev = GradientReversal.apply

        self.mi_coef = mi_coef
        self.entropy_coef = entropy_coef
        self.adv_coef = adv_coef
        self.kl_coef = kl_coef
        self.forward_coef = forward_coef
        self.kl_epsilon = 1e-8
    
    def get_action_logits(self, states, m):
        policy_input = torch.cat([states, m], dim=-1)
        
        if self.checkpoint:
            def run_policy(policy_input):
                return torch.clamp(self.policy(policy_input), min=-50, max=50)
            action_logits = checkpoint(run_policy, policy_input)
        else:
            action_logits = torch.clamp(self.policy(policy_input), min=-50, max=50)
            
        return action_logits, m

    def forward(self, states):
        if self.checkpoint:
            def run_mac(states):
                mac_output = self.mac(states)
                return self.mac_output_proj(mac_output)
            m = checkpoint(run_mac, states)
        else:
            mac_output = self.mac(states)
            m = self.mac_output_proj(mac_output)

        if self.checkpoint:
            def run_prior(m):
                prior_out, _ = self.prior_net(m)
                prior_mean = self.prior_mean(prior_out)
                prior_std = F.softplus(self.prior_std(prior_out)) + 1e-4
                return prior_mean, prior_std
            prior_mean, prior_std = checkpoint(run_prior, m)
        else:
            prior_out, _ = self.prior_net(m)
            prior_mean = self.prior_mean(prior_out)
            prior_std = F.softplus(self.prior_std(prior_out)) + 1e-4

        # Memory Distribution
        mem_std = F.softplus(self.memory_var) + 1e-4
        mem_std = torch.clamp(mem_std, min=0.01, max=1.0)
        mem_dist = Normal(m, mem_std.expand_as(m))
        
        action_logits, m = self.get_action_logits(states, m)
        return {
            'm': m,
            'states': states,
            'mem_dist': mem_dist,
            'action_logits': action_logits,
            'prior_dist': Normal(prior_mean, prior_std)
        }

    def compute_losses(self, outputs):
        action_logits = outputs['action_logits']
        prior_dist = outputs['prior_dist']
        mem_dist = outputs['mem_dist']
        states = outputs['states']
        m = outputs['m']
        
        # I(S;M) - Discriminator Loss
        pos_pairs = torch.cat([states, m], dim=-1)
        neg_m = m[torch.randperm(m.size(0))]
        neg_pairs = torch.cat([states, neg_m], dim=-1)

        if self.checkpoint:
            def run_discriminator(pairs):
                gru_out, _ = self.disc_gru(pairs)
                return self.disc_linear(gru_out[:, -1])
            pos_scores = checkpoint(run_discriminator, pos_pairs)
            neg_scores = checkpoint(run_discriminator, neg_pairs)
        else:
            gru_out, _ = self.disc_gru(pos_pairs)
            pos_scores = self.disc_linear(gru_out[:, -1])
            gru_out, _ = self.disc_gru(neg_pairs)
            neg_scores = self.disc_linear(gru_out[:, -1])
        
        # Discriminator loss
        mi_disc_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        ) + F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        
        # Policy MI loss
        policy_scores = GradientReversal.apply(pos_scores)
        mi_policy_loss = F.binary_cross_entropy_with_logits(
            policy_scores, torch.ones_like(policy_scores)
        )

        # Combined MI loss components
        mi_loss = mi_disc_loss + mi_policy_loss

        # I(S;M) - Forward Model Loss
        forward_input = torch.cat([states[:, :-1], m[:, :-1]], dim=-1)
        pred_states = self.forward_model(forward_input)
        forward_loss = F.mse_loss(pred_states, states[:, 1:])

        # Action Entropy
        log_probs = F.log_softmax(action_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(log_probs * probs).sum(dim=-1).mean()

        # Adversarial Loss I(A;M|S)
        real_input = torch.cat([
            states, 
            action_logits.detach(),
            m.detach()
        ], dim=-1)
        
        if self.checkpoint:
            def run_mmi_discriminator(inputs):
                return self.mmi_discriminator(inputs)
            real_logits = checkpoint(run_mmi_discriminator, real_input)
        else:
            real_logits = self.mmi_discriminator(real_input)
        
        real_logits_rev = GradientReversal.apply(real_logits)
        adv_loss = F.binary_cross_entropy_with_logits(
            real_logits_rev, 
            torch.ones_like(real_logits_rev)
        )

        # KL Regularization
        kl_loss = kl_divergence(mem_dist, prior_dist)
        kl_loss = torch.clamp(kl_loss, min=-self.kl_epsilon, max=1e3)
        kl_loss = kl_loss.mean()

        total_loss = (
            self.mi_coef * mi_loss +
            self.forward_coef * forward_loss +
            self.entropy_coef * -entropy +
            self.adv_coef * adv_loss +
            self.kl_coef * kl_loss
        )
        
        return {
            'total_loss': total_loss,
            'loss_components': {
                'mi_loss': mi_loss.item(),
                'forward_loss': forward_loss.item(),
                'entropy': entropy.item(),
                'adv_loss': adv_loss.item(),
                'kl_loss': kl_loss.item()
            }
        }
