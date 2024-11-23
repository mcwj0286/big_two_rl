import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from models.decision_transformer import DecisionTransformer
import random
from game.big2Game import big2Game  # Assuming big2Game is the correct class
# If vectorizedBig2Games is properly implemented, you can use it instead
import game.enumerateOptions as enumerateOptions
from PPONetwork import PPONetwork  # Import the PPONetwork class
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import h5py  # Import h5py for HDF5 handling
class Big2DecisionTransformerTrainer:
    def __init__(
        self,
        state_dim=412,
        act_dim=1695,
        n_blocks=6,
        h_dim=1024,
        context_len=20,
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
        self.context_len = context_len
        self.n_games = n_games
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len

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

        self.optimizer = torch.optim.AdamW(
            self.dt.parameters(),
            lr=learning_rate,
            eps=1e-5  # Increase epsilon for stability
        )

        # Separate buffers for each player
        self.trajectory_buffer = {
            1: [],
            2: [],
            3: [],
            4: []
        }

        self.player_ids = [1, 2, 3, 4]

        self.sess = tf.Session()
        self.ppo_agent = PPONetwork(self.sess, state_dim, act_dim, 'ppo_agent')
        self.ppo_agent.load_model(self.ppo_agent,'modelParameters136500')
        self.sess.run(tf.global_variables_initializer())
                # Initialize HDF5 file
        self.hdf5_file = h5py.File(hdf5_path, 'a')  # Append mode
        for player_id in self.player_ids:
            group = self.hdf5_file.require_group(f'player_{player_id}')
            # Create extendable datasets
            if 'states' not in group:
                group.create_dataset('states', shape=(0, state_dim), maxshape=(None, state_dim), dtype='f')
            if 'actions' not in group:
                group.create_dataset('actions', shape=(0,), maxshape=(None,), dtype='i')
            if 'rewards' not in group:
                group.create_dataset('rewards', shape=(0,), maxshape=(None,), dtype='f')
            if 'timesteps' not in group:
                group.create_dataset('timesteps', shape=(0,), maxshape=(None,), dtype='i')


    def collect_trajectories(self, n_games):
        """Collect game trajectories for all players"""

        for _ in range(n_games):
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
                # Get current player, state and available actions
                current_player, curr_state, curr_avail_actions = game.getCurrentState()
                state = curr_state[0]  # Remove batch dimension
                available_actions = curr_avail_actions[0]  # Remove batch dimension

                # Use PPO agent to select action
                action, _, _ = self.ppo_agent.step([state], [available_actions])
                action = action[0]

                # Record current state and action
                current_trajectories[current_player]['states'].append(state)
                current_trajectories[current_player]['actions'].append(action)
                current_trajectories[current_player]['rewards'].append(0)  # Reward will be assigned later
                current_trajectories[current_player]['timesteps'].append(timestep)
                # print(f"Player {current_player} - Action: {action}")
                # Step the environment
                reward, game_done, info = game.step(action)

                # Update timestep
                timestep += 1

                # If game is done, assign rewards
                if game_done:
                    # print(f"Game done. Reward: {reward}, Info: {info}")
                    for player_id in self.player_ids:
                        if len(current_trajectories[player_id]['rewards']) > 0:
                            current_trajectories[player_id]['rewards'][-1] = reward[player_id-1]
                    
                    # Append trajectories to buffer
                    for player_id in self.player_ids:
                        traj = current_trajectories[player_id]
                        if len(traj['states']) > 0:
                            self.trajectory_buffer[player_id].append(dict(traj))
                            
                            # Save to HDF5
                            group = self.hdf5_file[f'player_{player_id}']
                            states_ds = group['states']
                            actions_ds = group['actions']
                            rewards_ds = group['rewards']
                            timesteps_ds = group['timesteps']
                            
                            # Current size
                            current_size = states_ds.shape[0]
                            new_size = current_size + len(traj['states'])
                            
                            # Resize datasets
                            states_ds.resize((new_size, self.dt.state_dim))
                            actions_ds.resize((new_size,))
                            rewards_ds.resize((new_size,))
                            timesteps_ds.resize((new_size,))

                            # Write data
                            states_ds[current_size:new_size, :] = traj['states']
                            actions_ds[current_size:new_size] = traj['actions']
                            rewards_ds[current_size:new_size] = traj['rewards']
                            timesteps_ds[current_size:new_size] = traj['timesteps']
                

    def prepare_batch(self, player):
        """Prepare batch of data for specific player"""
        player_trajectories = self.trajectory_buffer[player]
        if len(player_trajectories) < self.batch_size:
            return None

        # Randomly sample trajectories
        batch_trajectories = np.random.choice(
            player_trajectories,
            size=self.batch_size,
            replace=True
        )

        # Initialize batch tensors
        states = torch.zeros(
            (self.batch_size, self.context_len, self.dt.state_dim),
            dtype=torch.float32,
            device=self.device
        )
        actions = torch.zeros(
            (self.batch_size, self.context_len),
            dtype=torch.long,
            device=self.device
        )
        returns_to_go = torch.zeros(
            (self.batch_size, self.context_len, 1),
            dtype=torch.float32,
            device=self.device
        )
        timesteps = torch.zeros(
            (self.batch_size, self.context_len),
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.zeros(
            (self.batch_size, self.context_len),
            dtype=torch.bool,
            device=self.device
        )

        for i, traj in enumerate(batch_trajectories):
            traj_len = len(traj['states'])
            # Random start point in trajectory to sample context_len steps
            if traj_len >= self.context_len:
                start_idx = np.random.randint(0, traj_len - self.context_len + 1)
                end_idx = start_idx + self.context_len
            else:
                start_idx = 0
                end_idx = traj_len

            # Fetch the slices
            traj_states = traj['states'][start_idx:end_idx]
            traj_actions = traj['actions'][start_idx:end_idx]
            traj_rewards = traj['rewards'][start_idx:end_idx]
            traj_timesteps = traj['timesteps'][start_idx:end_idx]

            # Convert to tensors
            traj_states = torch.tensor(np.array(traj_states), dtype=torch.float32, device=self.device)
            traj_actions = torch.tensor(np.array(traj_actions), dtype=torch.long, device=self.device)
            traj_rewards = torch.tensor(np.array(traj_rewards), dtype=torch.float32, device=self.device)
            traj_timesteps = torch.tensor(np.array(traj_timesteps), dtype=torch.long, device=self.device)

            # Calculate returns to go
            returns_to_go_ = traj_rewards.flip(0).cumsum(0).flip(0).unsqueeze(-1)

            # Place into batch tensors
            seq_len = traj_states.shape[0]
            states[i, :seq_len] = traj_states
            actions[i, :seq_len] = traj_actions
            returns_to_go[i, :seq_len] = returns_to_go_
            timesteps[i, :seq_len] = traj_timesteps

            attention_mask[i, :seq_len] = 1  # Mark valid steps
            # print(f"States shape: {states.shape}, Actions shape: {actions.shape}, Returns shape: {returns_to_go.shape}, Timesteps shape: {timesteps.shape}, Mask shape: {attention_mask.shape}")
            # print(f"returns_to_go: {returns_to_go}")
        return timesteps, states, actions, returns_to_go, attention_mask

    def train_step(self, player):
        """Perform a training step for a specific player"""
        batch = self.prepare_batch(player)
        if batch is None:
            print("Not enough data to form a batch.")
            return None  # Not enough data to form a batch

        timesteps, states, actions, returns_to_go, attention_mask = batch
        self.dt.train()
        self.optimizer.zero_grad()

        # Forward pass through the model
        action_preds = self.dt(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
        )

        # Add debug prints
        # print(f"Action preds shape: {action_preds.shape}")
        # print(f"Actions shape: {actions.shape}")
        
        # Shift predictions and targets
        action_preds = action_preds[:, :-1]  # Remove last prediction
        action_target = actions[:, 1:]  # Remove first action
        attention_mask = attention_mask[:, 1:]
        
        # Reshape tensors
        action_preds = action_preds.reshape(-1, self.dt.act_dim)
        action_target = action_target.reshape(-1)
        attention_mask = attention_mask.reshape(-1)

        # Filter padding
        valid_indices = attention_mask.bool()
        action_preds = action_preds[valid_indices]
        action_target = action_target[valid_indices]

        # Verify shapes before loss calculation
        if action_preds.size(0) == 0 or action_target.size(0) == 0:
            return float('inf')

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 1.0)

        # Compute loss with safety checks
        try:
            loss = F.cross_entropy(action_preds, action_target)
            if torch.isfinite(loss):
                loss.backward()
                self.optimizer.step()
                return loss.item()
            else:
                return float('inf')
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            return float('inf')
        

    def save_checkpoint(self, current_loss):
        if current_loss < self.best_loss:
            torch.save(self.dt.state_dict(), self.best_model_path)
            self.best_loss = current_loss
            print(f"Saved new best model with loss {current_loss:.4f}")

    def train(self, n_epochs=100, games_per_epoch=100):
        for epoch in range(n_epochs):
            # Collect new trajectories
            self.collect_trajectories(games_per_epoch)

            epoch_losses = {player_id: [] for player_id in self.player_ids}

            # For each player, perform training steps
            for player_id in self.player_ids:
                num_batches = len(self.trajectory_buffer[player_id]) // self.batch_size

                for _ in range(num_batches):
                    loss = self.train_step(player_id)
                    if loss is not None:
                        epoch_losses[player_id].append(loss)

            # Calculate average losses
            avg_losses = {}
            for player_id in self.player_ids:
                if epoch_losses[player_id]:
                    avg_loss = sum(epoch_losses[player_id]) / len(epoch_losses[player_id])
                    avg_losses[player_id] = avg_loss
                else:
                    avg_losses[player_id] = float('inf')

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{n_epochs}")
            for player_id in self.player_ids:
                print(f"Player {player_id} - Loss: {avg_losses[player_id]:.4f}")

            # Save checkpoint
            avg_loss = sum(avg_losses.values()) / len(avg_losses)
            self.save_checkpoint(avg_loss)

            # Optionally clear buffers to save memory
            self.trajectory_buffer = {player_id: [] for player_id in self.player_ids}

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
    trainer = Big2DecisionTransformerTrainer()
    trainer.train(n_epochs=1500, games_per_epoch=100)