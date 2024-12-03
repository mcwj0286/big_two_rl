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

from game.big2Game import big2Game  # Ensure this is the correct import path

class ModelEvaluator:
    def __init__(
        self,
        state_dim=412,
        act_dim=1695,
        best_model_path='output/official_decision_transformer_1.pt',
        ppo_model_path='modelParameters136500',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model = None,
        mode = 'eval'  # Add mode parameter
    ):
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.mode = mode  # or val

        if model is not None:
            self.dt = model
            self.dt.eval()
        else:
            # Initialize Decision Transformer
            self.dt = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                hidden_size=512,
                max_ep_len=90,
                seq_len=30,  
                max_length=None,
                action_tanh = False
            ).to(device)
            

        
            # Load Decision Transformer weights
            if os.path.exists(best_model_path):
                self.dt.load_state_dict(torch.load(best_model_path, map_location=self.device))
                self.dt.eval()
                if self.mode == 'eval':
                    print(f"Loaded Decision Transformer weights from {best_model_path}")
            else:
                raise FileNotFoundError(f"Decision Transformer model file not found: {best_model_path}")


        self.state_buffer = [[] for _ in range(4)]
        self.action_buffer = [[] for _ in range(4)]
        self.timestep_buffer = [[] for _ in range(4)]
        self.reward_buffer = [[] for _ in range(4)]
        
    
        self.ppo_agent = PPONetwork(self.state_dim, self.act_dim).to(self.device)

        if os.path.exists(ppo_model_path):
            self.ppo_agent.load_state_dict(torch.load(ppo_model_path, map_location=self.device))
            self.ppo_agent.to(self.device)
            self.ppo_agent.eval()
            if self.mode == 'eval':
                print(f"Loaded PPO Agent weights from {ppo_model_path}")
        else:
            raise FileNotFoundError(f"PPO Agent model file not found: {ppo_model_path}")

        # Add local timestep counters for each player
        self.local_timesteps = [0, 0, 0, 0]  # One counter for each player

    def evaluate_game(self, num_games=1000):
        """Evaluate the Decision Transformer against the PPO Agent."""
        dt_wins = 0
        ppo_wins = 0
        draws = 0
        total_rewards_dt = 0
        total_rewards_ppo = 0

        for game_num in range(1, num_games + 1):
            game = big2Game()
            game.reset()
            game_done = False
            self.local_timesteps = [0, 0, 0, 0]  # Reset local timesteps at start of each game
            target_reward = 1 # random set number of target reward ,TODO : adjust it to the real target reward 
            while not game_done:
                current_player, curr_state, curr_avail_actions = game.getCurrentState() # (n_game) , (n_game,1,412) , (n_game,1,1695)
                # print(f"Game {game_num} | Timestep {timestep} | Player {current_player}")
                state = curr_state[0]  # Remove batch dimension
                available_actions = curr_avail_actions[0]  # Remove batch dimension
                if current_player in [1, 3]:  # Decision Transformer players
                    with torch.no_grad():
                        # if self.mode == 'eval':
                        #     print(f"Decision Transformer player {current_player}")
                        # Convert state and prepare initial tensors
                        
                        state = torch.tensor([state], dtype=torch.float32, device=self.device)# Shape: (1, 412)
                        
                        local_t = self.local_timesteps[current_player - 1]# Shape: (1,) 
                        timestep = torch.tensor([local_t], dtype=torch.long, device=self.device)
                        
                        return_to_go = torch.tensor([target_reward], dtype=torch.float32, device=self.device)# Shape: (1,) 

                        # First add current state and use the previous action (or zero if first move)
                        self.state_buffer[current_player - 1].append(state.squeeze(0))
                        self.timestep_buffer[current_player - 1].append(timestep)
                        self.reward_buffer[current_player - 1].append(return_to_go)

                        # Now prepare the input based on updated buffer
                        seq_length = len(self.state_buffer[current_player - 1])
                        # Shape: (1, seq_length, 412) - batch_size=1, sequence_length, state_dim
                        batch_states = torch.stack(self.state_buffer[current_player - 1], dim=0).unsqueeze(0)
                        # Shape: (1, seq_length) - batch_size=1, sequence_length
                        batch_rewards = torch.stack(self.reward_buffer[current_player - 1], dim=0).unsqueeze(0)
                        # Shape: (1, seq_length) - batch_size=1, sequence_length
                        batch_timesteps = torch.tensor(self.timestep_buffer[current_player - 1], dtype=torch.long, device=self.device).unsqueeze(0)

                        
                        # Prepare one-hot encoded action
                        action_one_hot = torch.zeros(1, 1, self.act_dim, device=self.device)
                        self.action_buffer[current_player - 1].append(action_one_hot)
                        
                        batch_action_one_hot = torch.stack(self.action_buffer[current_player - 1], dim=1)
                        # print(f"States shape: {batch_states.shape}")  # (1, seq_length, 412)
                        # print(f"Actions shape: {batch_action_one_hot.shape}")  # (1, seq_length, 1695)
                        # print(f"Returns shape: {batch_rewards}")  # (1, seq_length)
                        # print(f"Timesteps shape: {batch_timesteps}")  # (1, seq_length)
                        # Get action from model
                        action_logits = self.dt.get_action(
                            timesteps=batch_timesteps,
                            states=batch_states,
                            actions=batch_action_one_hot,
                            returns_to_go=batch_rewards,
                        )

                        # Apply action mask and get action
                        available_actions_tensor = torch.tensor(available_actions, dtype=torch.float32, device=self.device)
                        masked_logits = action_logits + available_actions_tensor
                        action = torch.argmax(masked_logits).item()
                        
                        # Convert action to one-hot
                        action_one_hot = torch.zeros(1,1,self.act_dim, device=self.device)
                        action_one_hot[0][0][action] = 1
                        self.action_buffer[current_player - 1][-1] = action_one_hot

                        # Increment the local timestep counter
                        self.local_timesteps[current_player - 1] += 1
                        # if self.mode == 'eval':
                        #     print(f"Decision Transformer action: {action}")
                else:  # PPO Agent players
                    # curr_state = torch.tensor(curr_state, dtype=torch.float32, device=self.device)
                    # curr_avail_actions = torch.tensor(curr_avail_actions, dtype=torch.float32, device=self.device)
                    action, _, _ = self.ppo_agent.act(curr_state, curr_avail_actions)
                    action = action[0]
                    # print(f'ppo agent action: {action}')    

                # Step the environment
                reward, game_done, info = game.step(action)

                # Update rewards
                if game_done:
                    total_rewards_dt += reward[0] + reward[2]  # Players 1 and 3
                    total_rewards_ppo += reward[1] + reward[3]  # Players 2 and 4

                    # Determine the only winner
                    reward_list = reward.tolist()  # Convert numpy array to list
                    max_reward = max(reward_list)
                    if reward_list.count(max_reward) == 1:  # Ensure there's only one winner
                        winner_index = reward_list.index(max_reward)
                        if winner_index in [0, 2]:  # Decision Transformer players
                            dt_wins += 1
                        else:  # PPO Agent players
                            ppo_wins += 1
                    else:
                        print(f'error in reward: {reward_list}')

                    #clear buffer
                    self.state_buffer = [[] for _ in range(4)]
                    self.action_buffer = [[] for _ in range(4)]
                    self.timestep_buffer = [[] for _ in range(4)]
                    self.reward_buffer = [[] for _ in range(4)]
                    self.local_timesteps = [0, 0, 0, 0]

                    

                # game_timestep += 1

            if game_num % 100 == 0 and self.mode == 'eval':
                print(f"Completed {game_num}/{num_games} games.")

        # Calculate statistics
        dt_win_rate = (dt_wins / num_games) * 100
        ppo_win_rate = (ppo_wins / num_games) * 100
        draw_rate = (draws / num_games) * 100
        avg_reward_dt = total_rewards_dt / num_games
        avg_reward_ppo = total_rewards_ppo / num_games

        if self.mode == 'eval':
            print("\nEvaluation Results:")
            print(f"Total Games: {num_games}")
            print(f"Decision Transformer Wins: {dt_wins} ({dt_win_rate:.2f}%)")
            print(f"PPO Agent Wins: {ppo_wins} ({ppo_win_rate:.2f}%)")
            print(f"Draws: {draws} ({draw_rate:.2f}%)")
            print(f"Average Reward - Decision Transformer: {avg_reward_dt:.2f}")
            print(f"Average Reward - PPO Agent: {avg_reward_ppo:.2f}")
        return dt_win_rate, avg_reward_dt

    # def __del__(self):
    #     # Close TensorFlow session
    #     if hasattr(self, 'sess') and self.sess:
    #         self.sess.close()
    #     print("Closed TensorFlow session.")

if __name__ == "__main__":
    evaluator = ModelEvaluator(
        state_dim=412,
        act_dim=1695,
        # best_model_path='output/decision_transformer.pt',
        # best_model_path='output/dt_20winrate.pt',
        ppo_model_path='output/modelParameters_best.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mode='eval'  # Add mode parameter
    )
    evaluator.evaluate_game(num_games=10)