import numpy as np
import torch
import torch.nn as nn
import sys
import transformers
sys.path.append('/home/johnmok/Documents/GitHub/big_two_rl')
from models.trajectory_gpt2 import GPT2Model


class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])
    
    
class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim=412,
            act_dim=1695,
            hidden_size=512,
            max_length=None,
            max_ep_len=90,
            seq_len=30,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_ctx=max_ep_len,
            n_layer=6,
            n_head=8,
            n_inner=4 * hidden_size,  # Add this line
            resid_pdrop=0.1,         # Add dropout
            layer_norm_epsilon=1e-5,  # Add layer norm epsilon
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(seq_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        # self.embed_action = torch.nn.Embedding(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions,  returns_to_go, timesteps, attention_mask=None):
        """
        Forward pass for the Decision Transformer model.
        Args:
            states (torch.Tensor): Tensor of shape (batch_size, seq_length, state_dim) representing the states.
            actions (torch.Tensor): Tensor of shape (batch_size, seq_length, action_dim) representing the actions.
            returns_to_go (torch.Tensor): Tensor of shape (batch_size, seq_length, 1) representing the returns to go.
            timesteps (torch.Tensor): Tensor of shape (batch_size, seq_length) representing the timesteps.
            attention_mask (torch.Tensor, optional): Tensor of shape (batch_size, seq_length) representing the attention mask. 
                                                        If None, a mask of ones will be used. Default is None.
        Returns:
            tuple: A tuple containing:
                - state_preds (torch.Tensor): Tensor of shape (batch_size, seq_length, state_dim) representing the predicted states.
                - action_preds (torch.Tensor): Tensor of shape (batch_size, seq_length, action_dim) representing the predicted actions.
                - return_preds (torch.Tensor): Tensor of shape (batch_size, seq_length, 1) representing the predicted returns.
        """

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long,device=states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask
            # attention_mask=stacked_attention_mask.half() if stacked_attention_mask.dtype == torch.float32 else stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions,returns_to_go, timesteps, **kwargs):
      
        """
        Generate an action prediction based on the given states, actions, returns to go, and timesteps.
        Args:
            states (torch.Tensor): Tensor of shape (sequence_length, state_dim) representing the states.
            actions (torch.Tensor): Tensor of shape (sequence_length, act_dim) representing the actions.
            returns_to_go (torch.Tensor): Tensor of shape (sequence_length, 1) representing the returns to go.
            timesteps (torch.Tensor): Tensor of shape (sequence_length,) representing the timesteps.
            **kwargs: Additional arguments for the forward method.
        Returns:
            torch.Tensor: The predicted action for the last timestep in the sequence.
        Functionality:
            This function reshapes the input tensors to ensure they have the correct dimensions and pads them if necessary
            to match the maximum sequence length. It then calls the forward method to generate action predictions and 
            returns the predicted action for the last timestep in the sequence.
        """
        
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
    
# if __name__ == "__main__":
#     state_dim = 10
#     act_dim = 5
#     hidden_size = 128
#     max_length = 20

#     model = DecisionTransformer(state_dim, act_dim, 768, max_length=max_length)

#     # Create example tensors
#     batch_size = 2
#     seq_length = 15
#     attention_mask = torch.zeros(batch_size, seq_length)
#     states = torch.randn(batch_size, seq_length, state_dim)
#     actions = torch.randint(0, act_dim, (batch_size, seq_length))
#     rewards = torch.randn(batch_size, seq_length)
#     returns_to_go = torch.randn(batch_size, seq_length, 1)
#     timesteps = torch.randint(0, max_length, (batch_size, seq_length))

#     print("Input States Shape:", states.shape)
#     print("Input Actions Shape:", actions.shape)
#     print("Input Rewards Shape:", rewards.shape)
#     print("Input Returns to Go Shape:", returns_to_go.shape)
#     print("Input Timesteps Shape:", timesteps.shape)
#     print("Attention Mask Shape:", attention_mask)
#     # Test the forward pass
#     state_preds, action_preds, return_preds = model(states, actions, returns_to_go, timesteps,attention_mask)

#     print("State Predictions:", state_preds.shape)
#     print("Action Predictions:", action_preds.shape)
#     print("Return Predictions:", return_preds.shape)

#     # Test the get_action method
#     single_state = torch.randn(state_dim)
#     single_action = torch.randn(act_dim)
#     single_return_to_go = torch.randn(1)
#     single_timestep = torch.randint(0, max_length, (1,))

#     action = model.get_action(
#         single_state.unsqueeze(0),
#         single_action.unsqueeze(0),
#         single_return_to_go.unsqueeze(0),
#         single_timestep.unsqueeze(0)
#     )

#     print("Predicted Action:", action)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")