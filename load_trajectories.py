# analyze_trajectories.py
#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_trajectories(hdf5_path='output/pytorch_ppo_trajectories.hdf5', act_dim=1695):
    """Analyze game trajectories stored in the updated HDF5 structure."""
    
    with h5py.File(hdf5_path, 'r') as h5file:
        sequences_group = h5file['sequences']
        player_ids = [1, 2, 3, 4]
        total_action_counts = np.zeros(act_dim, dtype=np.int64)
        player_action_counts = {pid: np.zeros(act_dim, dtype=np.int64) for pid in player_ids}
        
        for seq_name in sequences_group:
            seq = sequences_group[seq_name]
            actions = seq['actions'][:]
            player_id = seq.attrs['player_id']
            
            if player_id in player_ids:
                player_action_counts[player_id] += np.bincount(actions, minlength=act_dim)
                total_action_counts += np.bincount(actions, minlength=act_dim)
        
        for player_id in player_ids:
            actions = player_action_counts[player_id]
            total = actions.sum()
            print(f"\nPlayer {player_id} - Total Actions: {total}")
            top_actions = np.argsort(-actions)[:5]
            print(f"Most Common Actions for Player {player_id}:")
            for a in top_actions:
                count = actions[a]
                percentage = (count / total) * 100
                print(f"Action {a}: {count} times ({percentage:.2f}%)")
        
        # Overall action distribution
        total_actions = total_action_counts.sum()
        print("\nOverall Action Distribution:")
        top_actions = np.argsort(-total_action_counts)[:100]
        for a in top_actions:
            count = total_action_counts[a]
            percentage = (count / total_actions) * 100
            print(f"Action {a}: {count} times ({percentage:.2f}%)")
        
        # Plot the action distribution
        plt.figure(figsize=(12,6))
        plt.bar(np.arange(act_dim), total_action_counts)
        plt.title('Overall Action Distribution')
        plt.xlabel('Action ID')
        plt.ylabel('Frequency')
        plt.show()

if __name__ == "__main__":
    analyze_trajectories(hdf5_path='output/pytorch_ppo_trajectories.hdf5', act_dim=1695)
#%%
import h5py
import os 
import numpy as np
import matplotlib.pyplot as plt
def split_hdf5_file(original_file_path, output_dir, num_splits=6):
    """
    Splits a large HDF5 file into smaller files.

    Parameters:
    - original_file_path (str): Path to the original large HDF5 file.
    - output_dir (str): Directory where smaller HDF5 files will be saved.
    - num_splits (int): Number of smaller files to create.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the original HDF5 file in read mode
    with h5py.File(original_file_path, 'r') as orig_file:
        player_ids = list(orig_file.keys())  # ['player_1', 'player_2', ...]
        
        # Determine the number of samples for each player
        player_lengths = {}
        for player in player_ids:
            group = orig_file[player]
            # Assuming all datasets have the same length
            player_lengths[player] = len(group['states'])
        
        # Calculate split indices for each player
        split_indices = {}
        for player in player_ids:
            total = player_lengths[player]
            indices = np.linspace(0, total, num_splits + 1, dtype=int)
            split_indices[player] = indices
        
        # Create each split file
        for split_num in range(num_splits):
            split_file_name = f"trajectories_part_{split_num + 1}.hdf5"
            split_file_path = os.path.join(output_dir, split_file_name)
            
            with h5py.File(split_file_path, 'w') as split_file:
                for player in player_ids:
                    orig_group = orig_file[player]
                    split_group = split_file.create_group(player)
                    
                    start = split_indices[player][split_num]
                    end = split_indices[player][split_num + 1]
                    
                    for dataset_name in orig_group.keys():
                        data = orig_group[dataset_name][start:end]
                        # Create extendable datasets if needed
                        maxshape = list(data.shape)
                        maxshape[0] = None  # Allow unlimited rows
                        split_group.create_dataset(
                            dataset_name,
                            data=data,
                            maxshape=tuple(maxshape),
                            chunks=True  # Enable chunking for larger datasets
                        )
            
            print(f"Created {split_file_name} with data samples {split_num + 1}/{num_splits}.")

if __name__ == "__main__":
    # Path to the original large HDF5 file
    original_hdf5_path = 'output/trajectories.hdf5'
    
    # Directory to save the smaller HDF5 files
    output_directory = 'output/split_trajectories'
    
    # Number of splits desired
    number_of_splits = 6
    
    # Call the splitting function
    split_hdf5_file(original_hdf5_path, output_directory, number_of_splits)
    
    print("Splitting completed successfully.")
# %%
import h5py
import os

def verify_split_files(split_dir, num_splits=6):
    for split_num in range(1, num_splits + 1):
        split_file = os.path.join(split_dir, f"trajectories_part_{split_num}.hdf5")
        print(f"\nVerifying {split_file}:")
        with h5py.File(split_file, 'r') as f:
            for player in f.keys():
                group = f[player]
                states = group['states']
                actions = group['actions']
                returns = group['rewards']
                timesteps = group['timesteps']
                print(f"  {player}: states={states.shape}, actions={actions.shape}, returns={returns.shape}, timesteps={timesteps.shape}")

if __name__ == "__main__":
    split_directory = 'output/split_trajectories'
    verify_split_files(split_directory, num_splits=6)
# %%
# ...existing code...
import h5py
def validate_hdf5_file(hdf5_path='output/trajectories.hdf5'):
    """Read and validate the HDF5 file structure and data integrity."""
    with h5py.File(hdf5_path, 'r') as f:
        required_groups = ['sequences']
        required_datasets = {
            'sequences': ['states', 'actions', 'rewards', 'timesteps', 'game_ids', 'player_ids']
        }
        
        # Check for required groups
        for group in required_groups:
            if group not in f:
                print(f"Missing required group: {group}")
                return False
            else:
                print(f"Group '{group}' found.")
        
        # Check for required datasets in each group
        for group, datasets in required_datasets.items():
            grp = f[group]
            for dataset in datasets:
                if dataset not in grp:
                    print(f"Missing required dataset: {dataset} in group {group}")
                    return False
                else:
                    print(f"Dataset '{dataset}' found in group '{group}'.")
        
        # Additional integrity checks can be added here
        # For example, check dataset lengths are consistent
        sequences_group = f['sequences']
        lengths = [len(sequences_group[ds]) for ds in required_datasets['sequences']]
        if len(set(lengths)) != 1:
            print("Inconsistent dataset lengths in 'sequences' group.")
            return False
        else:
            print("All datasets in 'sequences' group have consistent lengths.")
        
        print("HDF5 file validation passed.")
        return True

if __name__ == "__main__":
    hdf5_path = 'trajectories.hdf5'
    if validate_hdf5_file(hdf5_path):
        print("Validation successful.")
    else:
        print("Validation failed.")

    with h5py.File(hdf5_path, 'r') as f:
        sequences_group = f['sequences']
        sequence_1_data = {dataset: sequences_group[dataset][10000] for dataset in sequences_group.keys()}
        print("Sequence 1 Data:")
        for key, value in sequence_1_data.items():
            print(f"{key}: {value.shape}")
    
        num_sequences = len(sequences_group['states'])
        print(f"Number of sequences in the file: {num_sequences}")
# %%
