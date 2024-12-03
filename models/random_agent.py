import torch
import numpy as np

class RandomAgent:
    def __init__(self, state_dim, act_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = device

    def act(self, states, available_actions):
        # Remove batch dimension if present
        if isinstance(available_actions, torch.Tensor):
            available_actions = available_actions.cpu().numpy()
        
        if len(available_actions.shape) == 3:
            available_actions = available_actions[0][0]  # Remove batch dimension
        action_indices = np.arange(1695)
        action_indices = action_indices + available_actions
        # print("Action indices:", action_indices.shape)
        # print("available_actions indices:", available_actions.shape)
        # Get valid action indices where value is 0
        valid_actions = np.where(action_indices >=0)[1]
        print("Available action choices:", valid_actions)
        
        # Randomly select one valid action
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = 1694  # Default action if no valid actions available
            print('error')
        
        # Return action in the same format as other agents
        # action, log_prob, entropy
        return [action], None, None

