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

        # Normalize rewards to range [-1, 1]
        rewards = 2 * (rewards - self.min_reward) / (self.max_reward - self.min_reward) - 1
        # rewards[actions == 1694] -= 0.1  # Apply penalty for action 1694

        return {
            'states': states,
            'actions': actions,
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
    Custom collate function to pad/truncate sequences, generate attention masks,
    and convert actions to one-hot encoding without using a padding value in act_dim.

    Args:
        batch (list): List of dictionaries returned by PPOGameplayDataset.__getitem__().
        fixed_seq_length (int): Desired fixed sequence length.
        act_dim (int): Dimension of the action space for one-hot encoding.

    Returns:
        dict: Batched and padded tensors along with the attention mask and one-hot encoded actions.
    """
    padded_states = []
    padded_actions = []
    padded_rewards = []
    padded_timesteps = []
    attention_masks = []
    game_ids = []
    player_ids = []

    for sample in batch:
        seq_len = sample['states'].size(0)

        # Truncate sequences longer than fixed_seq_length
        if seq_len > fixed_seq_length:
            states = sample['states'][:fixed_seq_length]
            actions = sample['actions'][:fixed_seq_length]
            rewards = sample['rewards'][:fixed_seq_length]
            timesteps = sample['timesteps'][:fixed_seq_length]
            attn_mask = torch.ones(fixed_seq_length, dtype=torch.long)
        else:
            pad_size = fixed_seq_length - seq_len

            # Pad states with zeros
            states = torch.cat([
                sample['states'],
                torch.zeros(pad_size, sample['states'].size(1), dtype=torch.float32)
            ], dim=0)

            # Pad actions with -1 (since action indices are >= 0)
            actions = torch.cat([
                sample['actions'],
                torch.full((pad_size,), -1, dtype=torch.long)
            ], dim=0)

            # Pad rewards with zeros
            rewards = torch.cat([
                sample['rewards'],
                torch.zeros(pad_size, dtype=torch.float32)
            ], dim=0)

            # Pad timesteps with zeros
            timesteps = torch.cat([
                sample['timesteps'],
                torch.zeros(pad_size, dtype=torch.long)
            ], dim=0)

            # Create attention mask: 1 for valid tokens, 0 for padding
            attn_mask = torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(pad_size, dtype=torch.long)
            ], dim=0)

        padded_states.append(states)
        padded_actions.append(actions)
        padded_rewards.append(rewards)
        padded_timesteps.append(timesteps)
        attention_masks.append(attn_mask)
        game_ids.append(sample['game_id'])
        player_ids.append(sample['player_id'])

    # Stack all padded tensors
    batch_states = torch.stack(padded_states)           # Shape: (B, fixed_seq_length, state_dim)
    batch_actions = torch.stack(padded_actions)         # Shape: (B, fixed_seq_length)
    batch_rewards = torch.stack(padded_rewards)         # Shape: (B, fixed_seq_length)
    batch_timesteps = torch.stack(padded_timesteps)     # Shape: (B, fixed_seq_length)
    batch_attention_mask = torch.stack(attention_masks) # Shape: (B, fixed_seq_length)
    batch_game_ids = torch.stack(game_ids)              # Shape: (B,)
    batch_player_ids = torch.stack(player_ids)          # Shape: (B,)

    # Create one-hot encoded actions
    # Initialize tensor with zeros
    batch_actions_one_hot = torch.zeros(
        batch_actions.size(0),  # batch size
        batch_actions.size(1),  # sequence length
        act_dim,                # action dimension
        dtype=torch.float32
    )

    # Create a mask for valid actions (actions != -1)
    valid_actions_mask = batch_actions != -1  # Shape: (B, T)

    # Get indices of valid actions
    indices = valid_actions_mask.nonzero(as_tuple=False)  # Shape: (num_valid_actions, 2)

    # Get valid action indices
    valid_action_indices = batch_actions[valid_actions_mask].long()  # Shape: (num_valid_actions,)

    # Scatter ones into the one-hot tensor at valid positions
    batch_actions_one_hot[indices[:, 0], indices[:, 1], valid_action_indices] = 1.0

    return {
        'states': batch_states,                       # Shape: (B, fixed_seq_length, state_dim)
        'actions': batch_actions,                     # Shape: (B, fixed_seq_length)
        'actions_one_hot': batch_actions_one_hot,     # Shape: (B, fixed_seq_length, act_dim)
        'rewards': batch_rewards,                     # Shape: (B, fixed_seq_length)
        'timesteps': batch_timesteps,                 # Shape: (B, fixed_seq_length)
        'attention_mask': batch_attention_mask,       # Shape: (B, fixed_seq_length)
        'game_id': batch_game_ids,                    # Shape: (B,)
        'player_id': batch_player_ids                 # Shape: (B,)
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
#         print(batch)
#         break  # Only process the first batch for demonstration

#     # Close the dataset
#     dataset.close()