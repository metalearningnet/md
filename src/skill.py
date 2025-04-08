import torch
import torch.nn as nn
from utils import info
import torch.nn.functional as F
from titans import MemoryAsContextTransformer
from torch.distributions import Normal, kl_divergence, Categorical

class SkillMemory(nn.Module):
    def __init__(self,
                 action_dim: int = 64,
                 state_embed_dim: int = 128,
                 hidden_dim: int = 256,
                 mac_persistent_mem_tokens: int = 64,
                 mac_longterm_mem_tokens: int = 64,
                 mac_depth: int = 4,
                 mac_segment_len: int = 32,
                 mi_coeff: float = 1.0,
                 entropy_coeff: float = 0.1,
                 adv_coeff: float = 0.5,
                 kl_coeff: float = 0.01):
        super().__init__()
        info(f'Skill memory (action_dim: {action_dim} state_embed_dim: {state_embed_dim} hidden_dim: {hidden_dim})')
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # ========== Embedding Layers ==========
        self.state_embedding = nn.Linear(state_embed_dim, hidden_dim)

        # ========== MAC Preprocessor ==========
        self.token_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # ========== Memory-Augmented Context Transformer ==========
        self.mac = MemoryAsContextTransformer(
            num_tokens=1,
            dim=hidden_dim,
            depth=mac_depth,
            segment_len=mac_segment_len,
            num_persist_mem_tokens=mac_persistent_mem_tokens,
            num_longterm_mem_tokens=mac_longterm_mem_tokens,
            neural_memory_segment_len=mac_segment_len,
            token_emb=None
        )
        self.mac_output_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Fixes dimension collapse
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # ========== Memory System ==========
        self.memory_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.memory_var = nn.Parameter(torch.zeros(1, hidden_dim))

        # ========== Policy Network ==========
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # ========== Skill-Conditioned Prior ==========
        self.prior_net = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.prior_mean = nn.Linear(hidden_dim, hidden_dim)
        self.prior_std = nn.Linear(hidden_dim, hidden_dim)

        # ========== MD Objective Components ==========
        self.disc_gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.disc_linear = nn.Linear(hidden_dim, 1)

        self.mmi_discriminator = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # ========== Loss Coefficients ==========
        self.mi_coeff = mi_coeff
        self.entropy_coeff = entropy_coeff
        self.adv_coeff = adv_coeff
        self.kl_coeff = kl_coeff

    def forward(self, states):
        batch_size, seq_len, _ = states.shape
        
        # ===== 1. Embed Inputs =====
        state_embed = self.state_embedding(states)  # (B, T, H)

        # ===== 2. Prepare MAC Input =====
        processed = self.token_processor(state_embed)  # (B, T, H)
        
        # ===== 3. Process Through MAC =====
        # Create valid token indices
        dummy_tokens = torch.zeros(
            batch_size, seq_len,
            dtype=torch.long,
            device=states.device
        )
        
        # Pass processed features via memory_input
        mac_output, _ = self.mac(
            dummy_tokens,  # Valid integer indices
            memory_input=processed,  # Bypass token_emb
            return_cache=False
        )
        mac_output = self.mac_output_proj(mac_output.unsqueeze(-1))  # [B, T, 1] => [B, T, H]
        m_seq = self.memory_proj(mac_output)  # (B, T, H)

        # ===== 4. Skill-Conditioned Prior =====
        prior_input = m_seq[:, -1, 0, :]
        prior_out, _ = self.prior_net(prior_input)
        prior_mean = self.prior_mean(prior_out)
        prior_std = F.softplus(self.prior_std(prior_out)) + 1e-4

        # ===== 5. Memory Distribution =====
        mem_std = F.softplus(self.memory_var) + 1e-4
        mem_dist = Normal(m_seq, mem_std.expand_as(m_seq))
        
        return {
            "mem_dist": mem_dist,
            "prior_dist": Normal(prior_mean, prior_std),
            "state_embed": state_embed,
            "m_seq": m_seq
        }

    def compute_losses(self, batch):
        states, actions = batch
        outputs = self.forward(states)
        
        # ===== 1. I(S;M) =====
        m_seq_aligned = outputs["m_seq"].squeeze(2)
        pos_pairs = torch.cat([outputs["state_embed"], m_seq_aligned], dim=-1)
        neg_outputs = self.forward(states)
        neg_m_seq_aligned = neg_outputs["m_seq"].squeeze(2)
        neg_pairs = torch.cat([outputs["state_embed"], neg_m_seq_aligned], dim=-1)
        
        pos_scores = self.disc_linear(self.disc_gru(pos_pairs)[0][:, -1])
        neg_scores = self.disc_linear(self.disc_gru(neg_pairs)[0][:, -1])
        mi_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                  F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

        # ===== 2. Action Entropy =====
        action_logits = self.policy(torch.cat([outputs["state_embed"], m_seq_aligned], dim=-1))
        entropy = Categorical(logits=action_logits).entropy().mean()

        # ===== 3. Adversarial Loss =====
        actions_expanded = actions.unsqueeze(-1).expand(-1, -1, self.action_dim)
        real_logits = self.mmi_discriminator(torch.cat([actions_expanded, m_seq_aligned.detach()], dim=-1))
        fake_logits = self.mmi_discriminator(torch.cat([actions_expanded.detach(), m_seq_aligned], dim=-1))
        adv_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + \
                   F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))

        # ===== 4. KL Regularization =====
        kl_loss = kl_divergence(outputs["mem_dist"], outputs["prior_dist"]).mean()

        # ===== Total Loss =====
        total_loss = (
            self.mi_coeff * mi_loss +
            self.entropy_coeff * -entropy +
            self.adv_coeff * adv_loss +
            self.kl_coeff * kl_loss
        )
        
        return {
            "total_loss": total_loss,
            "loss_components": {
                "mi_loss": mi_loss.item(),
                "entropy": entropy.item(),
                "adv_loss": adv_loss.item(),
                "kl_loss": kl_loss.item()
            }
        }

    def generate(self, states):
        """Generate action logits for given states"""
        with torch.no_grad():
            # ===== 1. Input Handling =====
            device = next(self.parameters()).device

            # Ensure proper tensor dimensions
            original_dims = states.dim()
            if original_dims == 1:  # Single sample (state_dim,)
                states = states.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
                seq_len = 1
            elif original_dims == 2:  # Batch without sequence [B, state_dim]
                states = states.unsqueeze(1)  # [B, 1, state_dim]
                seq_len = 1
            else:  # Full sequence [B, T, state_dim]
                seq_len = states.size(1)

            # Convert to proper types
            states = states.float().to(device)

            # ===== 2. Embedding Processing =====
            # State embedding (always 3D: [B, T, H])
            state_embed = self.state_embedding(states)

            # ===== 3. MAC Processing =====
            # Prepare valid token indices
            batch_size = states.size(0)
            dummy_tokens = torch.zeros(
                batch_size, seq_len,
                dtype=torch.long,
                device=device
            )
            
            # Create combined input and process
            processed = self.token_processor(state_embed)
            
            # Process through MAC
            mac_output, _ = self.mac(dummy_tokens, memory_input=processed)
            mac_output = self.mac_output_proj(mac_output.unsqueeze(-1))
            assert mac_output.shape[-1] == self.hidden_dim, f"MAC output dim {mac_output.shape[-1]} != hidden_dim {self.hidden_dim}"

            # ===== 4. Memory Projection =====
            m = self.memory_proj(mac_output).squeeze(2)

            # ===== 5. Action Generation =====
            policy_input = torch.cat([
                state_embed,
                m
            ], dim=-1)
            
            action_logits = self.policy(policy_input)
            if original_dims == 1:
                action_logits = action_logits.squeeze(0)
                m = m.squeeze(0)

            return action_logits, m