# ppo_gameplay_collect.py

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from models.decision_transformer import DecisionTransformer
import random
from game.big2Game import big2Game  # Assuming big2Game is the correct class
import game.enumerateOptions as enumerateOptions
from PPONetwork import PPONetwork  # Import the PPONetwork class
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import h5py  # Import h5py for HDF5 handling

class PPO_gameplay_collect:
    def __init__(
        self,
        state_dim=412,
        act_dim=1695,
        n_blocks=6,
        h_dim=1024,
        context_len=20,  # Retained for potential future use
        n_heads=8,
        drop_p=0.1,
        n_games=32,
        batch_size=64,
        learning_rate=1e-4,
        max_ep_len=200,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        hdf5_path='trajectories.hdf5'  # Path to HDF5 file
    ):
        self.best_loss = float('inf')
        self.best_model_path = 'best_dt_model.pt'

        self.device = device
        self.context_len = context_len  # Not used for fixed-length splitting anymore
        self.n_games = n_games
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.player_ids = [1, 2, 3, 4]
        # Initialize Decision Transformer
        self.dt = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=h_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=drop_p
        ).to(device)

        # Initialize PPO Agent
        self.sess = tf.Session()
        self.ppo_agent = PPONetwork(self.sess, state_dim, act_dim, 'ppo_agent')
        self.ppo_agent.load_model(self.ppo_agent, 'modelParameters136500')
        self.sess.run(tf.global_variables_initializer())

        # Initialize HDF5 file
        self.hdf5_file = h5py.File(hdf5_path, 'a')  # Append mode
        self.sequences_group = self.hdf5_file.require_group('sequences')

        # Initialize a counter for sequences
        if 'sequence_counter' not in self.sequences_group.attrs:
            self.sequences_group.attrs['sequence_counter'] = 0

        # Optionally, you can store metadata about the dataset
        if 'metadata' not in self.hdf5_file.attrs:
            self.hdf5_file.attrs['metadata'] = 'PPO Gameplay Trajectories'

    def collect_trajectories(self, n_games):
        """Collect game trajectories and store each player's full gameplay as a sequence"""

        print(f"Starting collection of {n_games} games.")
        for game_num in range(n_games):
            # Initialize new game
            game = big2Game()
            game.reset()
            game_done = False
            timestep = 0

            # Initialize trajectories for each player
            current_trajectories = {
                1: defaultdict(list),
                2: defaultdict(list),
                3: defaultdict(list),
                4: defaultdict(list)
            }

            while not game_done and timestep < self.max_ep_len:
                # Get current player, state, and available actions
                current_player, curr_state, curr_avail_actions = game.getCurrentState()
                state = curr_state[0]  # Remove batch dimension
                available_actions = curr_avail_actions[0]  # Remove batch dimension

                # Use PPO agent to select action
                action, _, _ = self.ppo_agent.step([state], [available_actions])
                action = action[0]

                # Record current state, action, and timestep
                current_trajectories[current_player]['states'].append(state)
                current_trajectories[current_player]['actions'].append(action)
                current_trajectories[current_player]['rewards'].append(0)  # Reward will be assigned later
                current_trajectories[current_player]['timesteps'].append(timestep)

                # Step the environment
                reward, game_done, info = game.step(action)

                # Assign rewards if the game is done
                if game_done:
                    for player_id in self.player_ids:
                        if len(current_trajectories[player_id]['rewards']) > 0:
                            current_trajectories[player_id]['rewards'][-1] = reward[player_id-1]

                # Update timestep
                timestep += 1

            # After the game is done, process and save trajectories
            for player_id in self.player_ids:
                traj = current_trajectories[player_id]
                if len(traj['states']) > 0:
                    # Save the full trajectory as a single sequence
                    self.save_sequence(traj, game_num, player_id)

        print("Finished collecting trajectories.")
        # Close the HDF5 file after collection
        self.hdf5_file.close()

    def save_sequence(self, traj, game_id, player_id):
        """Save a single player's full gameplay trajectory as a sequence"""

        # Retrieve and update the sequence counter
        seq_counter = self.sequences_group.attrs['sequence_counter']
        seq_name = f'sequence_{seq_counter}'

        # Create a new subgroup for the sequence
        seq_group = self.sequences_group.create_group(seq_name)

        # Convert lists to numpy arrays
        states = np.array(traj['states'], dtype='float32')
        actions = np.array(traj['actions'], dtype='int32')
        rewards = np.array(traj['rewards'], dtype='float32')
        timesteps = np.array(traj['timesteps'], dtype='int32')

        # Save datasets within the sequence group
        seq_group.create_dataset('states', data=states)
        seq_group.create_dataset('actions', data=actions)
        seq_group.create_dataset('rewards', data=rewards)
        seq_group.create_dataset('timesteps', data=timesteps)

        # Save metadata
        seq_group.attrs['game_id'] = game_id
        seq_group.attrs['player_id'] = player_id

        # Increment the sequence counter
        self.sequences_group.attrs['sequence_counter'] = seq_counter + 1

    def load_best_model(self):
        if os.path.exists(self.best_model_path):
            self.dt.load_state_dict(torch.load(self.best_model_path))
            self.dt.to(self.device)
            print("Loaded best model from checkpoint.")
        else:
            print("No checkpoint found.")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    trainer = PPO_gameplay_collect()

    trainer.collect_trajectories(n_games=2500)  # Adjust the number of games as needed
    # Removed training invocation
    # trainer.train(n_epochs=1500, games_per_epoch=100)