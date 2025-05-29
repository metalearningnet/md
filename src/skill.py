import math
import torch
import torch.nn as nn
from utils import info
import torch.nn.functional as F
from titans import MemoryAsContextTransformer
from torch.utils.checkpoint import checkpoint
from torch.distributions import Normal, kl_divergence, Categorical

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
                 state_dim: int = 892,
                 num_tokens: int = 151936,
                 action_dim: int = 1024,
                 hidden_dim: int = 892,
                 mac_persistent_mem_tokens: int = 128,
                 mac_longterm_mem_tokens: int = 128,
                 mac_depth: int = 4,
                 mac_segment_len: int = 512,
                 mac_neural_memory_qkv_receives_diff_views: bool = False,
                 mac_neural_mem_weight_residual: bool = False,
                 mi_coeff: float = 1.0,
                 entropy_coeff: float = 0.5,
                 adv_coeff: float = 0.5,
                 kl_coeff: float = 0.05,
                 forward_coeff: float = 0.1,
                 checkpoint: dict = None):
        
        super().__init__()
        info(f"Skill memory (state_dim: {state_dim} action_dim: {action_dim} hidden_dim: {hidden_dim})")

        ckpt_config = checkpoint or {}
        self.checkpoint_mac = ckpt_config.get('mac', False)
        self.checkpoint_policy = ckpt_config.get('policy', False)
        self.checkpoint_prior = ckpt_config.get('prior', False)
        self.checkpoint_discriminators = ckpt_config.get('discriminators', False)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Memory-Augmented Context Transformer
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
        self.kl_epsilon = 1e-8
        self.mac_output_proj = nn.Linear(num_tokens, hidden_dim)
        self.memory_var = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.constant_(self.memory_var, math.log(0.5))

        # Policy Network
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
            nn.Linear(hidden_dim, state_dim)  # Predict next state
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

        # Loss Coefficients
        self.mi_coeff = mi_coeff
        self.entropy_coeff = entropy_coeff
        self.adv_coeff = adv_coeff
        self.kl_coeff = kl_coeff
        self.forward_coeff = forward_coeff
    
    def get_action_logits(self, states, m):
        policy_input = torch.cat([states, m], dim=-1)
        
        if self.checkpoint_policy:
            def run_policy(policy_input):
                return torch.clamp(self.policy(policy_input), min=-50, max=50)
            action_logits = checkpoint(run_policy, policy_input)
        else:
            action_logits = torch.clamp(self.policy(policy_input), min=-50, max=50)
            
        return action_logits, m

    def forward(self, states):
        # Process Through MAC
        if self.checkpoint_mac:
            def run_mac(states):
                mac_output = self.mac(states)
                return self.mac_output_proj(mac_output)
            m = checkpoint(run_mac, states)
        else:
            mac_output = self.mac(states)
            m = self.mac_output_proj(mac_output)

        # Skill-Conditioned Prior
        if self.checkpoint_prior:
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

    def compute_losses(self, states):
        outputs = self.forward(states)
        action_logits = outputs['action_logits']
        prior_dist = outputs['prior_dist']
        mem_dist = outputs['mem_dist']
        states = outputs['states']
        m = outputs['m']
        
        # I(S;M) - Discriminator Loss
        pos_pairs = torch.cat([states, m], dim=-1)
        neg_m = m[torch.randperm(m.size(0))]
        neg_pairs = torch.cat([states, neg_m], dim=-1)

        if self.checkpoint_discriminators:
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
        
        mi_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                  F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

        # I(S;M) - Forward Model Loss
        # Predict next state using current state and memory
        forward_input = torch.cat([states[:, :-1], m[:, :-1]], dim=-1)
        pred_states = self.forward_model(forward_input)
        forward_loss = F.mse_loss(pred_states, states[:, 1:])

         # Combined MI loss
        mi_total_loss = mi_loss + forward_loss

        # Action Entropy
        entropy = Categorical(logits=action_logits).entropy().mean()
        
        # Adversarial Loss (I(A;M|S))
        real_input = torch.cat([states, action_logits, m.detach()], dim=-1)
        fake_input = torch.cat([states, action_logits.detach(), m], dim=-1)
        if self.checkpoint_discriminators:
            def run_mmi_discriminator(inputs):
                return self.mmi_discriminator(inputs)
            real_logits = checkpoint(run_mmi_discriminator, real_input)
            fake_logits = checkpoint(run_mmi_discriminator, fake_input)
        else:
            real_logits = self.mmi_discriminator(real_input)
            fake_logits = self.mmi_discriminator(fake_input)
        
        adv_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + \
                   F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))

        # KL Regularization
        kl_loss = kl_divergence(mem_dist, prior_dist)
        kl_loss = torch.clamp(kl_loss, min=-self.kl_epsilon, max=1e3)
        kl_loss = kl_loss.mean()

        total_loss = (
            self.mi_coeff * mi_total_loss +
            self.forward_coeff * forward_loss +
            self.entropy_coeff * -entropy +
            self.adv_coeff * adv_loss +
            self.kl_coeff * kl_loss
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

    def generate(self, states):
        with torch.no_grad():
            # MAC Processing
            mac_output = self.mac(states)
            m = self.mac_output_proj(mac_output)
        
            # Action Generation
            action_logits, _ = self.get_action_logits(states, m)
            return action_logits