# analyze_trajectories.py
#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_trajectories(hdf5_path='output/trajectories.hdf5', act_dim=1695):
    # Open the HDF5 file in read mode
    with h5py.File(hdf5_path, 'r') as h5file:
        player_ids = [1, 2, 3, 4]
        total_action_counts = np.zeros(act_dim, dtype=np.int64)
        for player_id in player_ids:
            group_name = f'player_{player_id}'
            if group_name in h5file:
                group = h5file[group_name]
                if 'actions' in group:
                    actions = group['actions'][:]
                    action_counts = np.bincount(actions, minlength=act_dim)
                    total_actions = len(actions)
                    print(f"\nPlayer {player_id} - Total Actions: {total_actions}")
                    print(f"Most Common Actions for Player {player_id}:")
                    top_actions = np.argsort(-action_counts)[:5]
                    for a in top_actions:
                        count = action_counts[a]
                        percentage = (count / total_actions) * 100
                        print(f"Action {a}: {count} times ({percentage:.2f}%)")
                    # Add to total action counts
                    total_action_counts += action_counts
                else:
                    print(f"No actions data for player {player_id}")
            else:
                print(f"No data for player {player_id}")
        # Overall action distribution
        total_actions = total_action_counts.sum()
        print("\nOverall Action Distribution:")
        top_actions = np.argsort(-total_action_counts)[:10]
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
    analyze_trajectories(hdf5_path='output/trajectories.hdf5', act_dim=1695)