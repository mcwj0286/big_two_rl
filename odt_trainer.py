import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import defaultdict
from models.decision_transformer import DecisionTransformer
from game.big2Game import big2Game
from torch.utils.data import DataLoader
# from ppo_gameplay_dataset import PPOGameplayDataset, collate_fn
from replaybuffer import replaybuffer , collate_fn
from evaluate_model import ModelEvaluator  # Add this import at the top
import wandb
from dotenv import load_dotenv
load_dotenv()
def setup_wandb():
    """Setup wandb with API key and handle login"""
    try:
        # Try to get API key from environment variable
        api_key = os.getenv('WANDB_API_KEY')
        if (api_key is None):
            # If not found in env, look for it in the config file
            wandb_dir = os.path.expanduser("~/.wandb")
            api_key_file = os.path.join(wandb_dir, "api_key")
            if os.path.exists(api_key_file):
                with open(api_key_file, "r") as f:
                    api_key = f.read().strip()
        
        if (api_key is None):
            # If still not found, prompt user
            print("WandB API key not found. Please enter your API key:")
            api_key = input().strip()
            
        # Login with the API key
        wandb.login(key=api_key)
        print("Successfully logged in to Weights & Biases!")
        
    except Exception as e:
        print(f"Failed to login to WandB: {e}")
        print("Continuing without WandB logging...")
        return False
    
    return True

class ODTTrainer:
    def __init__(
        self,
        state_dim=412,
        act_dim=1695,
        hidden_size=512,
        context_len=30,
        n_heads=8,
        n_layers=6,
        learning_rate=1e-4,
        batch_size=64,
        n_self_play_games=10,
        max_epochs=1,
        save_winners_only=True,  # New parameter
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_save_path='output/odt_model.pt',
        eval_games=100,  # Add this parameter
        project_name="big2-dt",  # Add wandb project name
        exp_name="dt-experiment",  # Add experiment name
        temperature=2.0  # Start with a higher temperature
    ):
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.n_self_play_games = n_self_play_games
        self.max_epochs = max_epochs
        self.model_save_path = model_save_path
        self.save_winners_only = save_winners_only
        self.eval_games = eval_games
        self.best_win_rate = 0.0  # Add this to track best model
        self.best_reward = -1000.0  # Add this to track best model
        self.initial_temperature = temperature
        self.temperature = temperature
        
        # Initialize the Decision Transformer model
        self.model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            max_ep_len=90,
            seq_len=context_len,
            n_blocks=n_layers,
            n_heads=n_heads,
        ).to(device)
        self.evaluator = ModelEvaluator(
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                best_model_path=self.model_save_path,
                model=self.model,  # Pass the current model directly
                mode='val', 
                player_types=['dt', 'random', 'dt', 'random']  # Evaluate against random agents
            )
        # Initialize optimizer and loss function
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Initialize wandb
        self.run = wandb.init(
            project=project_name,
            name=exp_name,
            config={
                "state_dim": state_dim,
                "act_dim": act_dim,
                "hidden_size": hidden_size,
                "context_len": context_len,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_self_play_games": n_self_play_games,
                "max_epochs": max_epochs,
                "save_winners_only": save_winners_only,
            }
        )
        
        # Log model architecture as a table
        wandb.config.update({"model_type": "DecisionTransformer"})

    def self_play(self):
        """Collect trajectories through self-play using the current model."""
        # Initialize a list to store trajectories
        trajectories = []

        for game_num in range(self.n_self_play_games):
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

            # Buffers for model input per player
            state_buffers = [[] for _ in range(4)]
            action_buffers = [[] for _ in range(4)]
            reward_buffers = [[] for _ in range(4)]
            timestep_buffers = [[] for _ in range(4)]
            local_timesteps = [0, 0, 0, 0]

            while not game_done and timestep < 200:
                current_player, curr_state, curr_avail_actions = game.getCurrentState()
                state = curr_state[0]
                available_actions = curr_avail_actions[0]
                player_idx = current_player - 1

                with torch.no_grad():
                    # Prepare tensors
                    state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
                    available_actions_tensor = torch.tensor(available_actions, dtype=torch.float32, device=self.device)
                    timestep_tensor = torch.tensor([local_timesteps[player_idx]], dtype=torch.long, device=self.device)
                    reward_tensor = torch.tensor([2.0], dtype=torch.float32, device=self.device)
                 
                    # Update buffers
                    state_buffers[player_idx].append(state_tensor.squeeze(0))
                    reward_buffers[player_idx].append(reward_tensor)
                    timestep_buffers[player_idx].append(timestep_tensor)
                    action_buffers[player_idx].append(torch.zeros(1, 1, self.act_dim, device=self.device))
                    # Prepare model inputs
                    batch_states = torch.stack(state_buffers[player_idx], dim=0).unsqueeze(0)
                    batch_actions = torch.stack(action_buffers[player_idx], dim=1) 
                    batch_rewards = torch.stack(reward_buffers[player_idx], dim=0).unsqueeze(0)
                    batch_timesteps = torch.stack(timestep_buffers[player_idx], dim=0).unsqueeze(0)

                    # Get action from the model with current temperature
                    action_logits = self.model.get_action(
                        states=batch_states,
                        actions=batch_actions,
                        returns_to_go=batch_rewards,
                        timesteps=batch_timesteps,
                        temperature=self.temperature
                    )

                    # Mask invalid actions
                    masked_logits = action_logits +available_actions_tensor

                    # stochastic policy
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    action = dist.sample().item()
                    # action = torch.argmax(masked_logits).item()

                    # Print available actions for debugging
                    # action_indices = np.arange(1695)
                    # action_indices = action_indices + curr_avail_actions
                    # # print("Action indices:", action_indices.shape)
                    # # print("available_actions indices:", available_actions.shape)
                    # # Get valid action indices where value is 0
                    # valid_actions = np.where(action_indices >=0)[1]
                    # print("Available action choices:", valid_actions)
        
                    # print(f"Player {current_player} Action: {action}")
                    # Convert action to one-hot
                    action_one_hot = torch.zeros(1, 1, self.act_dim, device=self.device)
                    action_one_hot[0, 0, action] = 1
                    action_buffers[player_idx][-1]=action_one_hot

                    # Record trajectory
                    current_trajectories[current_player]['states'].append(state)
                    current_trajectories[current_player]['actions'].append(action)
                    current_trajectories[current_player]['rewards'].append(0)  # Placeholder
                    current_trajectories[current_player]['timesteps'].append(timestep)

                    # Step the environment
                    reward, game_done, info = game.step(action)

                    # Update timestep
                    timestep += 1
                    local_timesteps[player_idx] += 1

                    # Assign rewards if game is done
                    if game_done:
                        # print(f"Game {game_num} done. Reward: {reward}")
                        for pid in [1, 2, 3, 4]:
                            if len(current_trajectories[pid]['rewards']) > 0:
                                current_trajectories[pid]['rewards'][-1] = reward[pid - 1]
                            # Clear buffers for next round
                            state_buffers[pid-1] = []
                            action_buffers[pid-1] = []
                            reward_buffers[pid-1] = []
                            timestep_buffers[pid-1] = []
                            local_timesteps[pid-1] = 0

            # Modified trajectory collection logic
            if game_done:
                if self.save_winners_only:
                    # Find the winning player (player with highest reward)
                    winner_id = max(range(4), key=lambda i: reward[i]) + 1
                    # print(f"Game {game_num} done. Winner: Player {winner_id}")
                    # Only save the winner's trajectory
                    if len(current_trajectories[winner_id]['states']) > 0:
                        trajectories.append(current_trajectories[winner_id])
                else:
                    # Save all trajectories
                    for pid in [1, 2, 3, 4]:
                        if len(current_trajectories[pid]['states']) > 0:
                            trajectories.append(current_trajectories[pid])

        # Pass the collected trajectories directly to the train method
        self.train(trajectories)

    def train(self, trajectories):
        """Train the model using the collected trajectories."""
        # Prepare dataset from in-memory trajectories
        dataset = replaybuffer(data=trajectories)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=1
        )

        self.model.train()
        for epoch in range(self.max_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                actions_one_hot = batch['actions_one_hot'].to(self.device)
                # rewards = batch['rewards'].to(self.device)
                timesteps = batch['timesteps'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                returns_to_go = batch['returns_to_go'].to(self.device)

                # Forward pass
                state_preds, action_preds, return_preds = self.model(
                    states=states,
                    actions=actions_one_hot,
                    returns_to_go=returns_to_go.unsqueeze(-1),
                    timesteps=timesteps,
                    attention_mask=attention_mask
                )

                # Compute loss
                # Shift tokens for next token prediction
                action_target = actions.clone()
                action_preds = action_preds.reshape(-1, self.act_dim)
                action_target = action_target.reshape(-1)
                loss = self.criterion(action_preds, action_target)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Log batch loss
                # wandb.log({
                #     "batch_loss": loss.item(),
                #     "epoch": epoch,
                #     "batch": batch_idx
                # })
                
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            # Log epoch metrics
            wandb.log({
                "epoch_loss": avg_loss,
                "epoch": epoch,
            })
            print(f"Epoch [{epoch + 1}/{self.max_epochs}] Average Loss: {avg_loss:.4f}")

    def loop(self, num_iterations):
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")

            # Adjust temperature over iterations (exponential decay)
            decay_rate = 0.99  # Adjust decay rate as needed
            self.temperature = max(0.1, self.initial_temperature * (decay_rate ** iteration))

            # Optionally, use linear decay instead
            # min_temperature = 0.1
            # self.temperature = max(min_temperature, self.initial_temperature - (
            #     (self.initial_temperature - min_temperature) * iteration / num_iterations))

            # Log the current temperature
            wandb.log({"temperature": self.temperature, "iteration": iteration})

            # Self-play and training
            self.self_play()
            
            # Evaluate current model
            wins, avg_reward = self.evaluator.evaluate_game(num_games=self.eval_games)
            
            # Log validation metrics
            wandb.log({
                # "iteration": iteration,
                "validation/win_rate": wins,
                "validation/avg_reward": avg_reward,
                # "best_win_rate": self.best_win_rate,
                # "best_reward": self.best_reward
            })
            
            print(f"\nEvaluation Results:")
            print(f"Win Rate: {wins:.2f}%")
            print(f"Average Reward: {avg_reward:.4f}")
            
            # Save model if it's the best so far
            if wins > self.best_win_rate and avg_reward > self.best_reward:
                self.best_win_rate = wins
                best_model_path = self.model_save_path.replace('.pt', '_best.pt')
                torch.save(self.model.state_dict(), best_model_path)
                # Log best model with wandb
                # wandb.save(best_model_path)
                print(f"New best model saved with {wins:.2f} win rate and {avg_reward} reward ")
            
            # Always save latest model
            torch.save(self.model.state_dict(), self.model_save_path)
            # wandb.save(self.model_save_path)
            print(f"Latest model saved to {self.model_save_path}")
        
        # Finish the wandb run
        wandb.finish()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup wandb before creating trainer
    if setup_wandb():
        trainer = ODTTrainer(
            save_winners_only=True,
            project_name="big2-dt",
            exp_name=f"dt-exp-{wandb.util.generate_id()}"
        )
        trainer.loop(num_iterations=2000)
    else:
        print("Running without WandB logging")
        trainer = ODTTrainer(save_winners_only=True)
        trainer.loop(num_iterations=10000)
