# ppo_gameplay_dataset.py

import h5py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
                - 'rewards' (torch.FloatTensor): Tensor of shape (sequence_length,)
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
    and convert actions to one-hot encoding.

    Args:
        batch (list): List of dictionaries returned by PPOGameplayDataset.__getitem__().
        fixed_seq_length (int): Desired fixed sequence length. Must be a multiple of 3.
        act_dim (int): Dimension of the action space for one-hot encoding.

    Returns:
        dict: Batched and padded tensors along with the attention mask.
    """
    if fixed_seq_length % 3 != 0:
        raise ValueError("fixed_seq_length must be a multiple of 3 to align with interleaved tokens.")

    padded_states = []
    padded_actions = []
    padded_rewards = []
    padded_timesteps = []
    game_ids = []
    player_ids = []
    attention_masks = []

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
            # Calculate padding size
            pad_size = fixed_seq_length - seq_len

            # Pad states with zeros
            states = torch.cat([
                sample['states'],
                torch.zeros(pad_size, sample['states'].size(1), dtype=torch.float32)
            ], dim=0)

            # Use act_dim as padding index
            padding_value = act_dim  # Valid index for padding

            # Pad actions with the padding index
            actions = torch.cat([
                sample['actions'],
                torch.full((pad_size,), padding_value, dtype=torch.long)
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
    batch_game_ids = torch.stack(game_ids)             # Shape: (B,)
    batch_player_ids = torch.stack(player_ids)         # Shape: (B,)

    return {
        'states': batch_states,                 # (B, fixed_seq_length, state_dim)
        'actions': batch_actions,               # (B, fixed_seq_length)
        'rewards': batch_rewards,               # (B, fixed_seq_length)
        'timesteps': batch_timesteps,           # (B, fixed_seq_length)
        'attention_mask': batch_attention_mask, # (B, fixed_seq_length)
        'game_id': batch_game_ids,              # (B,)
        'player_id': batch_player_ids           # (B,)
    }

# if __name__ == "__main__":
#     # Example usage
#     dataset = PPOGameplayDataset(hdf5_path='output/pytorch_ppo_trajectories.hdf5')
#     print(f"Total sequences: {len(dataset)}")

#     # Retrieve a sample
#     sample = dataset[9]
#     print(sample['states'].shape)
#     print(sample['actions'])
#     print(sample['rewards'])
#     print(sample['timesteps'].shape)
#     print(sample['game_id'])
#     print(sample['player_id'])

# #     # Close the dataset when done
# #     dataset.close()