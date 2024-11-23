# decision_transformer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # Register buffer to ensure mask is not updated during training
        self.register_buffer('mask', mask)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape  # Batch size, sequence length, h_dim
        N, D = self.n_heads, C // self.n_heads  # Number of heads and head dimension

        # Rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)  # (B, N, T, D)

        # Compute attention weights (B, N, T, T)
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        # Create causal mask
        causal_mask = self.mask[..., :T, :T]  # (1, 1, T, T)

        # If attention_mask is provided, incorporate it
        if attention_mask is not None:
            # Expand attention_mask to (B, 1, 1, 3*T)
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 3*T)
            # Combine masks: only attend to positions where both masks are True
            combined_mask = causal_mask & extended_mask  # (B, 1, T, T)
        else:
            combined_mask = causal_mask  # (1, 1, T, T)

        # Apply mask: set to -inf where mask is False
        weights = weights.masked_fill(combined_mask == 0, float('-inf'))

        # Normalize weights
        normalized_weights = F.softmax(weights, dim=-1)  # (B, N, T, T)
        normalized_weights = self.att_drop(normalized_weights)

        # Compute attention output
        attention = normalized_weights @ v  # (B, N, T, D)
        attention = self.att_drop(attention)

        # Concatenate heads and project
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)  # (B, T, h_dim)
        out = self.proj_drop(self.proj_net(attention))  # (B, T, h_dim)

        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x, attention_mask=None):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x, attention_mask=attention_mask)  # Residual connection
        x = self.ln1(x)
        x = x + self.mlp(x)  # Residual connection
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### Transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.ModuleList(blocks)  # Changed from nn.Sequential to nn.ModuleList

        ### Projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # Discrete actions
        self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        use_action_tanh = False  # False for discrete actions

        ### Prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(h_dim, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, timesteps, states, actions, returns_to_go, attention_mask=None):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)  # Shape: (B, T, h_dim)

        # Time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings  # Shape: (B, T, h_dim)
        action_embeddings = self.embed_action(actions) + time_embeddings  # Shape: (B, T, h_dim)
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings  # Shape: (B, T, h_dim)

        # Stack embeddings: (B, 3, T, h_dim) -> (B, 3*T, h_dim)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)  # Shape: (B, 3*T, h_dim)

        # Adjust attention mask for interleaved sequence
        if attention_mask is not None:
            # Repeat the attention mask for each interleaved token
            attention_mask = attention_mask.repeat_interleave(3, dim=1)  # Shape: (B, 3*T)
        else:
            attention_mask = torch.ones(B, 3 * T, dtype=torch.long, device=states.device)

        # Apply transformer blocks with adjusted attention_mask
        for block in self.transformer:
            h = block(h, attention_mask=attention_mask)

        # Reshape to original format: (B, 3, T, h_dim)
        h = h.reshape(B, 3, T, self.h_dim).permute(0, 1, 2, 3)

        # Get predictions
        return_preds = self.predict_rtg(h[:, 0])     # Shape: (B, T, 1)
        state_preds = self.predict_state(h[:, 1])    # Shape: (B, T, state_dim)
        action_preds = self.predict_action(h[:, 2])  # Shape: (B, T, act_dim)
        
        return action_preds