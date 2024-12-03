# ppo_gameplay_dataset.py

import h5py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F  # Import for one-hot encoding

class PPOGameplayDataset(Dataset):
    def __init__(self, hdf5_path):
        """
        Initializes the PPOGameplayDataset.

        Args:
            hdf5_path (str): Path to the HDF5 file containing the trajectories.
        """
        self.hdf5_path = hdf5_path
        self.file = h5py.File(self.hdf5_path, 'r')
        self.sequences_group = self.file['sequences']
        self.sequence_names = list(self.sequences_group.keys())

        # Compute min and max rewards for normalization
        all_rewards = []
        for seq_name in self.sequence_names:
            seq_group = self.sequences_group[seq_name]
            rewards = seq_group['rewards'][:]
            all_rewards.extend(rewards)
        self.min_reward = min(all_rewards)
        self.max_reward = max(all_rewards)

    def __len__(self):
        """
        Returns the total number of sequences in the dataset.
        """
        return len(self.sequence_names)

    def __getitem__(self, idx):
        """
        Retrieves the sequence at the specified index.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - 'states' (torch.FloatTensor): Tensor of shape (sequence_length, state_dim)
                - 'actions' (torch.LongTensor): Tensor of shape (sequence_length,)
                - 'rewards' (torch.FloatTensor): Normalized tensor of shape (sequence_length,)
                - 'timesteps' (torch.LongTensor): Tensor of shape (sequence_length,)
                - 'game_id' (torch.LongTensor): Tensor containing the game ID
                - 'player_id' (torch.LongTensor): Tensor containing the player ID
        """
        seq_name = self.sequence_names[idx]
        seq_group = self.sequences_group[seq_name]

        # Retrieve data from the sequence group
        states = seq_group['states'][:]         # Shape: (N, state_dim)
        actions = seq_group['actions'][:]       # Shape: (N,)
        rewards = seq_group['rewards'][:]       # Shape: (N,)
        timesteps = seq_group['timesteps'][:]   # Shape: (N,)
        game_id = seq_group.attrs['game_id']    # Scalar
        player_id = seq_group.attrs['player_id']# Scalar

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        timesteps = torch.tensor(timesteps, dtype=torch.long)
        game_id = torch.tensor(game_id, dtype=torch.long)
        player_id = torch.tensor(player_id, dtype=torch.long)

        # Make a copy of actions and remove the last element
        actions_shifted = actions[:-1]
        # One-hot encode the shifted actions
        actions_shifted_onehot = F.one_hot(actions_shifted, num_classes=1695).float()
        # Create a zero vector and prepend it
        zero_vector = torch.zeros(1, 1695, dtype=torch.float32)
        actions_shifted_onehot = torch.cat([zero_vector, actions_shifted_onehot], dim=0)

        # Normalize rewards to range [-1, 1]
        rewards = 2 * (rewards - self.min_reward) / (self.max_reward - self.min_reward) - 1
        # rewards[actions == 1694] -= 0.1  # Apply penalty for action 1694

        return {
            'states': states,
            'actions': actions,
            'actions_one_hot': actions_shifted_onehot,
            'rewards': rewards,
            'timesteps': timesteps,
            'game_id': game_id,
            'player_id': player_id
        }

    def close(self):
        """
        Closes the HDF5 file.
        """
        if self.file:
            self.file.close()

def collate_fn(batch, fixed_seq_length=30, act_dim=1695):
    """
    Collate function for DataLoader that pads sequences to fixed length.
    
    Args:
        batch: List of dictionaries containing sequence data
        fixed_seq_length: Length to pad/truncate sequences to
        act_dim: Dimension of action space
    """
    # Get batch size
    batch_size = len(batch)
    
    # Initialize lists to store tensors
    states_list = []
    actions_list = []
    actions_one_hot_list = []
    rewards_list = []
    timesteps_list = []
    attention_masks = []
    
    # Process each sequence in the batch
    for seq in batch:
        seq_len = len(seq['states'])
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(fixed_seq_length, dtype=torch.float32)
        if seq_len < fixed_seq_length:
            attention_mask[seq_len:] = 0
        elif seq_len > fixed_seq_length:
            attention_mask = attention_mask[:fixed_seq_length]
            
        # Generate local timesteps starting from 0
        local_timesteps = torch.arange(min(seq_len, fixed_seq_length), dtype=torch.long)
        if seq_len < fixed_seq_length:
            local_timesteps = F.pad(local_timesteps, (0, fixed_seq_length - seq_len))
            
        # Pad or truncate sequences
        states = F.pad(seq['states'][:fixed_seq_length], 
                      (0, 0, 0, max(0, fixed_seq_length - seq_len)))
        actions = F.pad(seq['actions'][:fixed_seq_length], 
                       (0, max(0, fixed_seq_length - seq_len)))
        actions_one_hot = F.pad(seq['actions_one_hot'][:fixed_seq_length], 
                               (0, 0, 0, max(0, fixed_seq_length - seq_len)))
        rewards = F.pad(seq['rewards'][:fixed_seq_length], 
                       (0, max(0, fixed_seq_length - seq_len)))
        
        # Append to lists
        states_list.append(states)
        actions_list.append(actions)
        actions_one_hot_list.append(actions_one_hot)
        rewards_list.append(rewards)
        timesteps_list.append(local_timesteps)  # Use local_timesteps instead of seq['timesteps']
        attention_masks.append(attention_mask)
    
    # Stack all tensors
    return {
        'states': torch.stack(states_list),
        'actions': torch.stack(actions_list),
        'actions_one_hot': torch.stack(actions_one_hot_list),
        'rewards': torch.stack(rewards_list),
        'timesteps': torch.stack(timesteps_list),
        'attention_mask': torch.stack(attention_masks)
    }

# if __name__ == "__main__":
#     # Example usage of PPOGameplayDataset and collate_fn
#     hdf5_path = 'output/pytorch_ppo_trajectories.hdf5'  # Update with the actual path to your HDF5 file
#     dataset = PPOGameplayDataset(hdf5_path)

#     # Create a DataLoader with the custom collate function

#     dataloader = DataLoader(
#         dataset,
#         batch_size=2,  # Example batch size
#         collate_fn=lambda batch: collate_fn(batch, fixed_seq_length=30, act_dim=1695)
#     )

#     # Iterate through the DataLoader and print the output of collate_fn
#     for batch in dataloader:
#         print("States shape:", batch['states'].shape)
#         print("Actions shape:", batch['actions'].shape)
#         print("Actions one-hot shape:", batch['actions_one_hot'].shape)
#         print("Rewards shape:", batch['rewards'].shape)
#         print("Timesteps shape:", batch['timesteps'].shape)
#         print("Attention mask shape:", batch['attention_mask'].shape)
#         break  # Only process the first batch for demonstration
#         break  # Only process the first batch for demonstration

#     # Close the dataset
#     dataset.close()