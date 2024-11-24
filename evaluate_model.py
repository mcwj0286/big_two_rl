
# evaluate_model.py

import os
import numpy as np
import torch
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import h5py
# from PPONetwork import PPONetwork
from models.PPONetwork_pytorch import PPONetwork
from models.decision_transformer_original import DecisionTransformer
from game.big2Game import big2Game  # Ensure this is the correct import path

class ModelEvaluator:
    def __init__(
        self,
        state_dim=412,
        act_dim=1695,
        best_model_path='./output/decision_transformer.pt',
        ppo_model_path='modelParameters136500',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model = None
    ):
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim

        if model is not None:
            self.dt = model
            self.dt.eval()
        else:
            # Initialize Decision Transformer
            self.dt = DecisionTransformer(
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                n_blocks=6,
                h_dim=1024,
                context_len=30,
                n_heads=8,
                drop_p=0.1,
                max_timestep=1000
            ).to(self.device)
            # Load Decision Transformer weights
            if os.path.exists(best_model_path):
                self.dt.load_state_dict(torch.load(best_model_path, map_location=self.device))
                self.dt.eval()
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
            print(f"Loaded PPO Agent weights from {ppo_model_path}")
        else:
            raise FileNotFoundError(f"PPO Agent model file not found: {ppo_model_path}")

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
            timestep = 0
            target_reward = 13 # random set number of target reward ,TODO : adjust it to the real target reward 
            while not game_done:
                current_player, curr_state, curr_avail_actions = game.getCurrentState() # (n_game) , (n_game,1,412) , (n_game,1,1695)
                # print(f"Game {game_num} | Timestep {timestep} | Player {current_player}")
                state = curr_state[0]  # Remove batch dimension
                available_actions = curr_avail_actions[0]  # Remove batch dimension
                if current_player in [1, 3]:  # Decision Transformer players
                    with torch.no_grad():
                        # print('decision transformer turn')
                        state= torch.tensor([state], dtype=torch.float32, device=self.device) # (1,412)
                        timestep = torch.tensor([timestep], dtype=torch.long, device=self.device)
                        return_to_go = torch.tensor([target_reward], dtype=torch.float32, device=self.device)
                        if len(self.state_buffer[current_player - 1]) <1:
                            seq_length = 1
                            batch_states = state.unsqueeze(0)
                            batch_actions = torch.zeros((1, seq_length, self.act_dim), dtype=torch.float32, device=self.device)
                            batch_rewards = return_to_go.unsqueeze(0).unsqueeze(0)
                            batch_timesteps = timestep.unsqueeze(0)
                            # batch_attention_mask = torch.ones((1, seq_length), dtype=torch.bool, device=self.device)

             
                        else:
                            seq_length = len(self.state_buffer[current_player - 1])
    
                            # Stack states and actions without extra dimensions
                            batch_states = torch.stack(self.state_buffer[current_player - 1], dim=0).unsqueeze(0)      # Shape: [1, N, 412]
                            batch_actions = torch.stack(self.action_buffer[current_player - 1], dim=0).unsqueeze(0)    # Shape: [1, N, 1695]
                            batch_rewards = torch.stack(self.reward_buffer[current_player - 1], dim=0).unsqueeze(0)    # Shape: [1, N]
                            # Convert timesteps and rewards lists to tensors
                            # batch_rewards = torch.tensor(self.reward_buffer[current_player - 1], dtype=torch.float32, device=self.device).unsqueeze(0)    # Shape: [1, N]
                           
                            batch_timesteps = torch.tensor(self.timestep_buffer[current_player - 1], dtype=torch.long, device=self.device).unsqueeze(0)    # Shape: [1, N]

                        timesteps = batch_timesteps
                        states = batch_states
                        actions = batch_actions
                        returns_to_go = batch_rewards
                            # attention_mask = batch_attention_mask
                        
                        # print(f"Input shapes - Timesteps: {timesteps.shape}, States: {states.shape}, Actions: {actions.shape}, Returns to go: {returns_to_go.shape}")
                        action_probs = self.dt.forward(
                            timesteps=timesteps,
                            states=states,
                            actions=actions,
                            returns_to_go=returns_to_go,
                            # attention_mask=attention_mask
                        ) # Shape: [1, N, 1695]
                        
                        action_token = action_probs[0][-1]
                        available_actions_tensor = torch.tensor(available_actions, dtype=torch.float32, device=self.device)
                        # available_actions_tensor = available_actions_tensor.unsqueeze(0).unsqueeze(0)

                        action_probs = action_token + available_actions_tensor
                        action_probs = torch.softmax(action_probs, dim=-1)
                       
                        # prevent always choose pass (comment it if no always pass problem)
                        # top_probs, top_indices = torch.topk(action_probs, 2, dim=-1)

                        # print(f'Top 2 action probabilities: {top_probs.tolist()}')
                        # print(f'Top 2 action indices: {top_indices.tolist()}')
                        max_prob , action_idx = torch.max(action_probs, dim=0)
             
                        if action_idx.item()== 1694:
                            action_probs[1694] = 0.001



                        # predict action
                        max_prob , action = torch.max(action_probs, dim=-1)

                        # action = torch.argmax(action_probs, dim=-1).item()
                        # print(f'decision transformer action: {action.item()}, max_prob: {max_prob.item()}')

                        # # One-hot encode the action (optional)
                        action_onehot = torch.zeros(self.act_dim, dtype=torch.float32, device=self.device)
                        action_onehot[action] = 1.0
                        # Append to buffers
                        self.state_buffer[current_player - 1].append(state.squeeze(0))         # Shape: [412]
                        self.action_buffer[current_player - 1].append(action_onehot)           # Shape: [1695]
                        # self.action_buffer[current_player - 1].append(action_probs.squeeze(0).squeeze(0))           # Shape: [1695]
                        self.timestep_buffer[current_player - 1].append(timestep.item())       # Scalar
                        self.reward_buffer[current_player - 1].append(return_to_go)     # [1,1]

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

                timestep += 1

            if game_num % 100 == 0:
                print(f"Completed {game_num}/{num_games} games.")

        # Calculate statistics
        dt_win_rate = (dt_wins / num_games) * 100
        ppo_win_rate = (ppo_wins / num_games) * 100
        draw_rate = (draws / num_games) * 100
        avg_reward_dt = total_rewards_dt / num_games
        avg_reward_ppo = total_rewards_ppo / num_games

        # print("\nEvaluation Results:")
        # print(f"Total Games: {num_games}")
        # print(f"Decision Transformer Wins: {dt_wins} ({dt_win_rate:.2f}%)")
        # print(f"PPO Agent Wins: {ppo_wins} ({ppo_win_rate:.2f}%)")
        # print(f"Draws: {draws} ({draw_rate:.2f}%)")
        # print(f"Average Reward - Decision Transformer: {avg_reward_dt:.2f}")
        # print(f"Average Reward - PPO Agent: {avg_reward_ppo:.2f}")
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
        best_model_path='output/decision_transformer.pt',
        ppo_model_path='output/modelParameters_best.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    evaluator.evaluate_game(num_games=5)