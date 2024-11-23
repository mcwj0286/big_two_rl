# ppo_gameplay_dataset.py

import h5py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.

    Args:
        batch (list): List of dictionaries with sequence data.

    Returns:
        dict: Batched and padded tensors.
    """
    states = [item['states'] for item in batch]
    actions = [item['actions'] for item in batch]
    rewards = [item['rewards'] for item in batch]
    timesteps = [item['timesteps'] for item in batch]
    game_ids = [item['game_id'] for item in batch]
    player_ids = [item['player_id'] for item in batch]

    # Pad sequences
    states_padded = pad_sequence(states, batch_first=True, padding_value=0)
    actions_padded = pad_sequence(actions, batch_first=True, padding_value=-1)  # Assuming -1 is the padding index
    rewards_padded = pad_sequence(rewards, batch_first=True, padding_value=0)
    timesteps_padded = pad_sequence(timesteps, batch_first=True, padding_value=0)

    # Create attention masks
    attention_masks = (actions_padded != -1).long()  # Assuming actions_padded are padded with -1

    return {
        'states': states_padded,
        'actions': actions_padded,
        'rewards': rewards_padded,
        'timesteps': timesteps_padded,
        'game_id': torch.stack(game_ids),
        'player_id': torch.stack(player_ids),
        'attention_mask': attention_masks
    }
if __name__ == "__main__":
    # Example usage
    dataset = PPOGameplayDataset(hdf5_path='trajectories.hdf5')
    print(f"Total sequences: {len(dataset)}")

    # Retrieve a sample
    sample = dataset[9]
    print(sample['states'].shape)
    print(sample['actions'])
    print(sample['rewards'])
    print(sample['timesteps'].shape)
    print(sample['game_id'])
    print(sample['player_id'])

    # Close the dataset when done
    dataset.close()

