# evaluate_model.py

import os
import numpy as np
import torch
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import h5py
# from PPONetwork import PPONetwork
from models.PPONetwork_pytorch import PPONetwork
# from models.decision_transformer_original import DecisionTransformer
from models.decision_transformer import DecisionTransformer
from models.random_agent import RandomAgent  # Add this import

from game.big2Game import big2Game  # Ensure this is the correct import path

class ModelEvaluator:
    def __init__(
        self,
        state_dim=412,
        act_dim=1695,
        best_model_path='output/official_decision_transformer.pt',
        ppo_model_path='output/modelParameters_best.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model=None,
        mode='eval',
        # player_types=['dt', 'random', 'dt', 'random']  # New parameter for player types
        player_types=['dt', 'ppo', 'dt', 'ppo']  # New parameter for player types
    ):
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.mode = mode
        
        # Validate player types
        valid_types = {'dt', 'ppo', 'random'}
        if len(player_types) != 4 or not all(pt.lower() in valid_types for pt in player_types):
            raise ValueError("player_types must be a list of 4 items, each being 'dt', 'ppo', or 'random'")
        
        self.player_types = [pt.lower() for pt in player_types]
        
        # Initialize agents based on unique types needed
        self.agents = {}
        
        # Initialize DT if needed
        if 'dt' in self.player_types:
            if model is not None:
                self.agents['dt'] = model
                self.agents['dt'].eval()
            else:
                self.agents['dt'] = DecisionTransformer(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    hidden_size=512,
                    max_ep_len=90,
                    seq_len=30,
                    max_length=None,
                    action_tanh=False
                ).to(device)
                
                if os.path.exists(best_model_path):
                    self.agents['dt'].load_state_dict(torch.load(best_model_path, map_location=self.device))
                    self.agents['dt'].eval()
                    if self.mode == 'eval':
                        print(f"Loaded Decision Transformer weights from {best_model_path}")
                else:
                    raise FileNotFoundError(f"Decision Transformer model file not found: {best_model_path}")
        
        # Initialize PPO if needed
        if 'ppo' in self.player_types:
            self.agents['ppo'] = PPONetwork(self.state_dim, self.act_dim).to(self.device)
            if os.path.exists(ppo_model_path):
                self.agents['ppo'].load_state_dict(torch.load(ppo_model_path, map_location=self.device))
                self.agents['ppo'].eval()
                if self.mode == 'eval':
                    print(f"Loaded PPO Agent weights from {ppo_model_path}")
            else:
                raise FileNotFoundError(f"PPO Agent model file not found: {ppo_model_path}")
        
        # Initialize Random agent if needed
        if 'random' in self.player_types:
            self.agents['random'] = RandomAgent(state_dim, act_dim, device)

        # Initialize buffers for DT agents
        self.state_buffer = [[] for _ in range(4)]
        self.action_buffer = [[] for _ in range(4)]
        self.timestep_buffer = [[] for _ in range(4)]
        self.reward_buffer = [[] for _ in range(4)]
        self.local_timesteps = [0, 0, 0, 0]

    def evaluate_game(self, num_games=1000):
        win_counts = {'dt': 0, 'ppo': 0, 'random': 0}
        total_rewards = {'dt': 0, 'ppo': 0, 'random': 0}
        
        for game_num in range(1, num_games + 1):
            game = big2Game()
            game.reset()
            game_done = False
            self.local_timesteps = [0, 0, 0, 0]
            target_reward = 1

            while not game_done:
                current_player, curr_state, curr_avail_actions = game.getCurrentState()
                player_idx = current_player - 1
                agent_type = self.player_types[player_idx]
                
                state = curr_state[0]
                available_actions = curr_avail_actions[0]
                
                # Handle action selection based on agent type
                if agent_type == 'dt':
                    # DT specific logic
                    with torch.no_grad():
                        state = torch.tensor([state], dtype=torch.float32, device=self.device)
                        local_t = self.local_timesteps[player_idx]
                        timestep = torch.tensor([local_t], dtype=torch.long, device=self.device)
                        return_to_go = torch.tensor([target_reward], dtype=torch.float32, device=self.device)

                        # Update buffers and get action using existing DT logic
                        # ... (keep existing DT logic here)
                        self.state_buffer[player_idx].append(state.squeeze(0))
                        self.timestep_buffer[player_idx].append(timestep)
                        self.reward_buffer[player_idx].append(return_to_go)

                        seq_length = len(self.state_buffer[player_idx])
                        batch_states = torch.stack(self.state_buffer[player_idx], dim=0).unsqueeze(0)
                        batch_rewards = torch.stack(self.reward_buffer[player_idx], dim=0).unsqueeze(0)
                        batch_timesteps = torch.tensor(self.timestep_buffer[player_idx], dtype=torch.long, device=self.device).unsqueeze(0)

                        action_one_hot = torch.zeros(1, 1, self.act_dim, device=self.device)
                        self.action_buffer[player_idx].append(action_one_hot)
                        batch_action_one_hot = torch.stack(self.action_buffer[player_idx], dim=1)

                        action_logits = self.agents['dt'].get_action(
                            timesteps=batch_timesteps,
                            states=batch_states,
                            actions=batch_action_one_hot,
                            returns_to_go=batch_rewards,
                        )

                        available_actions_tensor = torch.tensor(available_actions, dtype=torch.float32, device=self.device)
                        masked_logits = action_logits + available_actions_tensor
                        action = torch.argmax(masked_logits).item()

                        action_one_hot = torch.zeros(1,1,self.act_dim, device=self.device)
                        action_one_hot[0][0][action] = 1
                        self.action_buffer[player_idx][-1] = action_one_hot

                        self.local_timesteps[player_idx] += 1

                        print(f"DT Action: {action}")
                else:
                    # PPO or Random agent logic
                    action, _, _ = self.agents[agent_type].act(curr_state, curr_avail_actions)
                    action = action[0]
                    print(f"{agent_type.upper()} Action: {action}")
                # Step the environment
                reward, game_done, info = game.step(action)

                if game_done:
                    # Update rewards and wins for each agent type
                    for idx, player_type in enumerate(self.player_types):
                        total_rewards[player_type] += reward[idx]
                    
                    # Determine winner
                    winner_idx = reward.tolist().index(max(reward))
                    winner_type = self.player_types[winner_idx]
                    win_counts[winner_type] += 1

                    # Clear buffers
                    self.state_buffer = [[] for _ in range(4)]
                    self.action_buffer = [[] for _ in range(4)]
                    self.timestep_buffer = [[] for _ in range(4)]
                    self.reward_buffer = [[] for _ in range(4)]
                    self.local_timesteps = [0, 0, 0, 0]

            if game_num % 100 == 0 and self.mode == 'eval':
                print(f"Completed {game_num}/{num_games} games.")

        # Calculate and print statistics
        if self.mode == 'eval':
            print("\nEvaluation Results:")
            print(f"Total Games: {num_games}")
            # Get unique agent types that are actually playing
            active_agents = set(self.player_types)
            for agent_type in active_agents:
                win_rate = (win_counts[agent_type] / num_games) * 100
                avg_reward = total_rewards[agent_type] / num_games
                print(f"{agent_type.upper()} Wins: {win_counts[agent_type]} ({win_rate:.2f}%)")
                print(f"{agent_type.upper()} Average Reward: {avg_reward:.2f}")

        if self.mode == 'eval':
            print("Evaluation complete.")
        else:
            return win_counts['dt'], total_rewards['dt']/ num_games

if __name__ == "__main__":
    evaluator = ModelEvaluator(
        state_dim=412,
        act_dim=1695,
        # best_model_path='output/decision_transformer.pt',
        # best_model_path='output/dt_20winrate.pt',
        ppo_model_path='output/modelParameters_best.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mode='eval',
       
    )
    evaluator.evaluate_game(num_games=100)