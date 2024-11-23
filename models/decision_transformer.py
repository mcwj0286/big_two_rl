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
        causal_mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # Register buffer for the causal mask
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape  # Batch size, sequence length, hidden dimension

        N, D = self.n_heads, C // self.n_heads  # Number of heads, dimension per head

        # Reshape q, k, v for multi-head attention
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)  # Shape: (B, N, T, D)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # Compute raw attention scores
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # Shape: (B, N, T, T)

        # Prepare causal mask
        causal_mask = self.causal_mask[:, :, :T, :T]  # Adjust to current sequence length

        # Combine masks if attention_mask is provided
        if attention_mask is not None:
            # attention_mask shape: (B, T)
            attention_mask = attention_mask[:, None, None, :]  # Shape: (B, 1, 1, T)
            combined_mask = causal_mask * attention_mask
        else:
            combined_mask = causal_mask

        # Apply the combined mask
        weights = weights.masked_fill(combined_mask == 0, float('-inf'))

        # Normalize and apply attention
        normalized_weights = F.softmax(weights, dim=-1)
        attention = self.att_drop(normalized_weights @ v)  # Shape: (B, N, T, D)

        # Reshape back to (B, T, C)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))  # Shape: (B, T, C)
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
        # Apply attention with residual connection
        x = x + self.attention(x, attention_mask=attention_mask)
        x = self.ln1(x)
        # Apply MLP with residual connection
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len

        # Transformer blocks
        input_seq_len = 3 * context_len  # Since we interleave returns, states, actions
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # Embedding layers
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_return = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)

        # Action embedding for discrete actions
        self.embed_action = nn.Embedding(act_dim, h_dim)
        use_action_tanh = False  # Set to True for continuous actions

        # Prediction heads
        self.predict_return = nn.Linear(h_dim, 1)
        self.predict_state = nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(h_dim, act_dim),
            *([nn.Tanh()] if use_action_tanh else [])
        )

    def forward(self, timesteps, states, actions, returns_to_go, attention_mask=None):
        """
        Args:
            timesteps (torch.LongTensor): Shape (B, T)
            states (torch.FloatTensor): Shape (B, T, state_dim)
            actions (torch.LongTensor): Shape (B, T)
            returns_to_go (torch.FloatTensor): Shape (B, T, 1)
            attention_mask (torch.LongTensor, optional): Shape (B, T), 1 for valid tokens, 0 for padding
        """
        B, T = timesteps.shape

        # Time embeddings
        time_embeddings = self.embed_timestep(timesteps)  # Shape: (B, T, h_dim)

        # Input embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        return_embeddings = self.embed_return(returns_to_go) + time_embeddings

        # Stack embeddings: (B, 3*T, h_dim)
        h = torch.stack((return_embeddings, state_embeddings, action_embeddings), dim=2)
        h = h.reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)

        # Adjust attention mask for interleaved sequence
        if attention_mask is not None:
            # Expand and repeat mask to match the interleaved sequence
            attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, 3)
            attention_mask = attention_mask.reshape(B, 3 * T)  # Shape: (B, 3*T)
        else:
            attention_mask = torch.ones(B, 3 * T, dtype=torch.long, device=states.device)

        # Apply transformer blocks with attention mask
        for block in self.transformer:
            h = block(h, attention_mask=attention_mask)

        # Reshape to original format
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # Predictions
        return_pred = self.predict_return(h[:, 2])  # Shape: (B, T, 1)
        state_pred = self.predict_state(h[:, 2])    # Shape: (B, T, state_dim)
        action_pred = self.predict_action(h[:, 1])  # Shape: (B, T, act_dim)

        return action_pred  # Return predicted actions