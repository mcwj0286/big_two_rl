import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F  # Import for one-hot encoding

class replaybuffer(Dataset):
    def __init__(self, data):
        """
        Initializes the replaybuffer with in-memory data.

        Args:
            data (list): List of trajectory dictionaries.
        """
        self.data = data

    def __len__(self):
        """Returns the total number of trajectories in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves the trajectory at the specified index."""
        traj = self.data[idx]

        reward = torch.tensor(traj['rewards'], dtype=torch.float32)
        action = torch.tensor(traj['actions'], dtype=torch.long)
        # Normalize rewards
        reward = reward/ 10.0

        #reward shaping , add penality to pass action
        reward[action == 1694] -= 0.3
        # add reward for action index >= 13 (not single card)
        reward[action >=14] += 0.1
        return {
            'states': torch.tensor(traj['states'], dtype=torch.float32),
            'actions': torch.tensor(traj['actions'], dtype=torch.long),
            'rewards': torch.tensor(reward, dtype=torch.float32),
            'timesteps': torch.tensor(traj['timesteps'], dtype=torch.long),
            'actions_one_hot' : F.one_hot(action, num_classes=1695).float()
        }

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
    returns_to_go_list = []  # New list for returns-to-go
    
    # Process each sequence in the batch
    for seq in batch:
        seq_len = len(seq['states'])
        
        # Calculate returns-to-go
        rewards = seq['rewards'][:fixed_seq_length]
        returns_to_go = torch.zeros_like(rewards)
        curr_return = 0
        for t in reversed(range(len(rewards))):
            curr_return = rewards[t] + curr_return
            returns_to_go[t] = curr_return
            
        # Pad returns-to-go if necessary
        if seq_len < fixed_seq_length:
            returns_to_go = F.pad(returns_to_go, (0, fixed_seq_length - seq_len))
        
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
        timesteps_list.append(local_timesteps)
        attention_masks.append(attention_mask)
        returns_to_go_list.append(returns_to_go)  # Add returns-to-go to list
    
    # Stack all tensors
    return {
        'states': torch.stack(states_list),
        'actions': torch.stack(actions_list),
        'actions_one_hot': torch.stack(actions_one_hot_list),
        'rewards': torch.stack(rewards_list),
        'timesteps': torch.stack(timesteps_list),
        'attention_mask': torch.stack(attention_masks),
        'returns_to_go': torch.stack(returns_to_go_list)  # Include returns-to-go in output
    }

# Example usage and testing of collate_fn
# if __name__ == "__main__":
#     # Create sample batch of 2 trajectories
#     sample_batch = [
#         {
#             'states': torch.randn(25, 416),  # Shorter sequence
#             'actions': torch.randint(0, 1695, (25,)),
#             'rewards': torch.rand(25),
#             'timesteps': torch.arange(25),
#             'actions_one_hot': F.one_hot(torch.randint(0, 1695, (25,)), num_classes=1695).float()
#         },
#         {
#             'states': torch.randn(35, 416),  # Longer sequence
#             'actions': torch.randint(0, 1695, (35,)),
#             'rewards': torch.rand(35),
#             'timesteps': torch.arange(35),
#             'actions_one_hot': F.one_hot(torch.randint(0, 1695, (35,)), num_classes=1695).float()
#         }
#     ]

#     # Test collate function
#     batch = collate_fn(sample_batch)
    
#     # Print shapes of output tensors
#     print("Batch shapes:")
#     for key, value in batch.items():
#         print(f"{key}: {value.shape}")
#         if key =='rewards':
#             print(value)
#         if key == 'returns_to_go':
#             print(value)