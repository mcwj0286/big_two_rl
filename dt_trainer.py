# dt_train_with_validation.py

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import random
from models.decision_transformer import DecisionTransformer
from PPONetwork import PPONetwork
from game.big2Game import big2Game
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import h5py

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
        batch_size=64,
        learning_rate=1e-4,
        max_ep_len=200,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        hdf5_path='trajectories.hdf5',
        best_model_path='best_dt_model.pt',
        validation_games=50
    ):
        self.device = device
        self.context_len = context_len
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.validation_games = validation_games

        self.best_avg_reward = -float('inf')
        self.best_win_rate = -float('inf')
        self.best_model_path = best_model_path

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
            eps=1e-5
        )

        # Initialize PPO Agent for validation
        self.sess = tf.Session()
        self.ppo_agent = PPONetwork(self.sess, state_dim, act_dim, 'ppo_agent')
        self.ppo_agent.load_model(self.ppo_agent, 'modelParameters136500')
        self.sess.run(tf.global_variables_initializer())

        # Load data from HDF5
        self.hdf5_path = hdf5_path
        self.player_ids = [1, 2, 3, 4]
        self.data = self.load_data()

    def load_data(self):
        data = {player_id: {'states': [], 'actions': [], 'returns': [], 'timesteps': []} for player_id in self.player_ids}

        with h5py.File(self.hdf5_path, 'r') as h5file:
            for player_id in self.player_ids:
                group_name = f'player_{player_id}'
                if group_name in h5file:
                    group = h5file[group_name]
                    states = group['states'][:]
                    actions = group['actions'][:]
                    rewards = group['rewards'][:]
                    timesteps = group['timesteps'][:]

                    # Calculate returns to go
                    returns = np.zeros_like(rewards)
                    ep_indices = np.where(rewards != 0)[0]
                    start_idx = 0
                    for end_idx in ep_indices:
                        ep_rewards = rewards[start_idx:end_idx+1]
                        ep_returns = np.flip(np.cumsum(np.flip(ep_rewards)))
                        returns[start_idx:end_idx+1] = ep_returns
                        start_idx = end_idx + 1

                    data[player_id]['states'] = states
                    data[player_id]['actions'] = actions
                    data[player_id]['returns'] = returns
                    data[player_id]['timesteps'] = timesteps
                else:
                    print(f"No data for player {player_id} in HDF5 file.")

        return data

    def prepare_batches(self, player):
        states = self.data[player]['states']
        actions = self.data[player]['actions']
        returns = self.data[player]['returns']
        timesteps = self.data[player]['timesteps']

        dataset_size = len(states)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        batches = []
        for start_idx in range(0, dataset_size - self.context_len, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            batch_states = []
            batch_actions = []
            batch_returns = []
            batch_timesteps = []
            batch_attention_mask = []

            for idx in batch_indices:
                if idx + self.context_len <= dataset_size:
                    s = states[idx:idx + self.context_len]
                    a = actions[idx:idx + self.context_len]
                    r = returns[idx:idx + self.context_len]
                    t = timesteps[idx:idx + self.context_len]

                    batch_states.append(s)
                    batch_actions.append(a)
                    batch_returns.append(r[:, None])
                    batch_timesteps.append(t)
                    batch_attention_mask.append(np.ones(self.context_len))

            if batch_states:
                batch_states = torch.tensor(batch_states, dtype=torch.float32, device=self.device)
                batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
                batch_returns = torch.tensor(batch_returns, dtype=torch.float32, device=self.device)
                batch_timesteps = torch.tensor(batch_timesteps, dtype=torch.long, device=self.device)
                batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.bool, device=self.device)

                batches.append((batch_timesteps, batch_states, batch_actions, batch_returns, batch_attention_mask))

        return batches

    def train_one_epoch(self):
        epoch_losses = {player_id: [] for player_id in self.player_ids}
        for player_id in self.player_ids:
            batches = self.prepare_batches(player_id)
            for batch in batches:
                loss = self.train_step(batch)
                if loss is not None:
                    epoch_losses[player_id].append(loss)

        avg_losses = {}
        for player_id in self.player_ids:
            if epoch_losses[player_id]:
                avg_loss = sum(epoch_losses[player_id]) / len(epoch_losses[player_id])
                avg_losses[player_id] = avg_loss
            else:
                avg_losses[player_id] = float('inf')

        return avg_losses

    def train_step(self, batch):
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

        # Shift predictions and targets
        action_preds = action_preds[:, :-1].contiguous()
        action_target = actions[:, 1:].contiguous()
        attention_mask = attention_mask[:, 1:].contiguous()

        # Reshape tensors
        action_preds = action_preds.reshape(-1, self.dt.act_dim)
        action_target = action_target.reshape(-1)
        attention_mask = attention_mask.reshape(-1)

        # Filter padding
        valid_indices = attention_mask.bool()
        action_preds = action_preds[valid_indices]
        action_target = action_target[valid_indices]

        if action_preds.size(0) == 0 or action_target.size(0) == 0:
            return None

        # Compute loss
        loss = F.cross_entropy(action_preds, action_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def validate(self):
        evaluator = ModelEvaluator(
            dt_model=self.dt,
            ppo_agent=self.ppo_agent,
            device=self.device
        )
        avg_reward_dt, avg_reward_ppo, dt_win_rate = evaluator.evaluate_game(num_games=self.validation_games)
        return avg_reward_dt, avg_reward_ppo, dt_win_rate

    def save_checkpoint(self, avg_reward_dt, dt_win_rate):
        improvement = False
        if avg_reward_dt > self.best_avg_reward and dt_win_rate > self.best_win_rate:
            self.best_avg_reward = avg_reward_dt
            self.best_win_rate = dt_win_rate
            torch.save(self.dt.state_dict(), self.best_model_path)
            print(f"Saved new best model with average reward {avg_reward_dt:.4f} and win rate {dt_win_rate:.2f}%")
            improvement = True
        return improvement

    def train(self, n_epochs=100):
        for epoch in range(n_epochs):
            avg_losses = self.train_one_epoch()
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            for player_id in self.player_ids:
                print(f"Player {player_id} - Loss: {avg_losses[player_id]:.4f}")

            # Validate the model
            print("Validating the model...")
            avg_reward_dt, avg_reward_ppo, dt_win_rate = self.validate()
            print(f"Validation Results - DT Avg Reward: {avg_reward_dt:.4f}, PPO Avg Reward: {avg_reward_ppo:.4f}, DT Win Rate: {dt_win_rate:.2f}%")

            # Save the model if both win rate and rewards improved
            self.save_checkpoint(avg_reward_dt, dt_win_rate)

        print("Training complete.")

    def __del__(self):
        if hasattr(self, 'sess') and self.sess:
            self.sess.close()
        print("Closed TensorFlow session.")

class ModelEvaluator:
    def __init__(
        self,
        dt_model,
        ppo_agent,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        state_dim=412,
        act_dim=1695
    ):
        self.device = device
        self.dt = dt_model.to(self.device)
        self.dt.eval()
        self.ppo_agent = ppo_agent
        self.state_dim = state_dim
        self.act_dim = act_dim

    def evaluate_game(self, num_games=100):
        dt_wins = 0
        ppo_wins = 0
        total_rewards_dt = 0
        total_rewards_ppo = 0

        for game_num in range(1, num_games + 1):
            game = big2Game()
            game.reset()
            game_done = False
            timestep = 0
            target_reward = 13

            while not game_done:
                current_player, curr_state, curr_avail_actions = game.getCurrentState()
                state = curr_state[0]
                available_actions = curr_avail_actions[0]

                if current_player in [1, 3]:  # Decision Transformer players
                    with torch.no_grad():
                        # Prepare inputs for the Decision Transformer
                        timesteps = torch.tensor([timestep], dtype=torch.long, device=self.device).unsqueeze(0)
                        states = torch.tensor([state], dtype=torch.float32, device=self.device).unsqueeze(0)
                        actions = torch.zeros((1), dtype=torch.long, device=self.device)
                        returns_to_go = torch.tensor([target_reward], dtype=torch.float32, device=self.device).unsqueeze(0)

                        action_logits = self.dt.forward(
                            timesteps=timesteps,
                            states=states,
                            actions=actions,
                            returns_to_go=returns_to_go,
                        )
                        action_probs = torch.softmax(action_logits, dim=-1).squeeze(0)

                        # Mask unavailable actions
                        available_actions_mask = torch.zeros(self.act_dim, device=self.device)
                        available_actions_indices = torch.tensor(available_actions, dtype=torch.long, device=self.device)
                        available_actions_mask[available_actions_indices] = 1
                        masked_action_probs = action_probs * available_actions_mask

                        if masked_action_probs.sum().item() == 0:
                            action = 1694  # Pass action ID
                        else:
                            masked_action_probs = masked_action_probs / masked_action_probs.sum()
                            action = torch.multinomial(masked_action_probs, num_samples=1).item()
                else:  # PPO Agent players
                    action, _, _ = self.ppo_agent.step([state], [available_actions])
                    action = action[0]

                # Step the environment
                reward, game_done, info = game.step(action)

                # Update rewards
                if game_done:
                    total_rewards_dt += reward[0] + reward[2]  # Players 1 and 3
                    total_rewards_ppo += reward[1] + reward[3]  # Players 2 and 4

                    # Determine the winner
                    reward_list = reward.tolist()
                    max_reward = max(reward_list)
                    if reward_list.count(max_reward) == 1:
                        winner_index = reward_list.index(max_reward)
                        if winner_index in [0, 2]:  # Decision Transformer players
                            dt_wins += 1
                        else:
                            ppo_wins += 1

                timestep += 1

        avg_reward_dt = total_rewards_dt / num_games
        avg_reward_ppo = total_rewards_ppo / num_games
        dt_win_rate = (dt_wins / num_games) * 100

        return avg_reward_dt, avg_reward_ppo, dt_win_rate

if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = Big2DecisionTransformerTrainer(
        state_dim=412,
        act_dim=1695,
        hdf5_path='trajectories.hdf5',
        best_model_path='best_dt_model.pt',
        validation_games=50
    )
    trainer.train(n_epochs=100)