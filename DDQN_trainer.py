import torch
import numpy as np
import random
from collections import deque
import json
import time
from game.big2Game import vectorizedBig2Games
from models.ddqn import QNetwork, Memory, update_parameters
import wandb
import os 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_wandb = True
# if use_wandb:
#     # Retrieve environment variables
#     wandb_api_key = os.getenv("WANDB_API_KEY")
#     wandb_project = os.getenv("WANDB_PROJECT")

#     wandb.login(key=wandb_api_key)

class Big2DDQNTrainer:
    def __init__(self, 
                 state_dim=412,
                 action_dim=1695,
                 hidden_dim=1024,
                 lr=2.5e-4,
                 gamma=0.995,
                 memory_size=50000,
                 batch_size=64,
                 target_update_freq=100,
                 eps_start=1.0,
                 eps_end=0.01,
                 eps_decay=0.9995,
                 n_games=8):
        
        # Initialize networks
        self.q_network = QNetwork(action_dim=action_dim, 
                                state_dim=state_dim, 
                                hidden_dim=hidden_dim).to(device)
        self.target_network = QNetwork(action_dim=action_dim, 
                                     state_dim=state_dim, 
                                     hidden_dim=hidden_dim).to(device)
        
        # Copy Q-network params to target network
        update_parameters(self.q_network, self.target_network)
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = Memory(memory_size)
        
        # Training parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        # Epsilon parameters for exploration
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        # Environment setup
        self.env = vectorizedBig2Games(n_games)
        self.n_games = n_games
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.games_completed = 0
        
        # Add tracking for best model
        self.best_average_reward = float('-inf')
        self.no_improvement_counter = 0

    def select_action(self, state, avail_actions):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.eps:
            # Random action from available actions
            valid_actions = np.where(avail_actions == 1)[0]
            return np.random.choice(valid_actions)
        
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.q_network(state)
            # Mask unavailable actions
            q_values = q_values + (avail_actions - 1) * 1e8
            return q_values.argmax().item()
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory.rewards) < self.batch_size:
            return None
            
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        
        # Get Q-values for current states
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Get Q-values for next states using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * max_next_q
        
        # Compute loss and update
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, total_steps, log_freq=1000, save_freq=10000):
        """Main training loop"""
        step_counter = 0
        episode_counter = 0
        
        # # Initialize wandb
        # wandb.init(project="big2-ddqn", config={
        #     "learning_rate": self.optimizer.param_groups[0]['lr'],
        #     "gamma": self.gamma,
        #     "batch_size": self.batch_size,
        #     "n_games": self.n_games
        # })
        
        while step_counter < total_steps:
            episode_counter += 1
            episode_rewards = np.zeros(self.n_games)
            episode_lengths = np.zeros(self.n_games)
            
            # Get initial states
            player_turns, states, avail_actions = self.env.getCurrStates()
            states = np.squeeze(states)
            avail_actions = np.squeeze(avail_actions)
            
            done = False
            while not done:
                step_counter += 1
                
                # Select and perform actions
                actions = []
                for i in range(self.n_games):
                    action = self.select_action(states[i], avail_actions[i])
                    actions.append(action)
                
                # Step environment
                rewards, dones, infos = self.env.step(actions)
                
                # Get next states
                next_player_turns, next_states, next_avail_actions = self.env.getCurrStates()
                next_states = np.squeeze(next_states)
                next_avail_actions = np.squeeze(next_avail_actions)
                
                # Store transitions in memory
                for i in range(self.n_games):
                    self.memory.update(states[i], actions[i], rewards[i], dones[i])
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                
                # Train model
                loss = self.train_step()
                if loss is not None:
                    self.losses.append(loss)
                
                # Update target network
                if step_counter % self.target_update_freq == 0:
                    update_parameters(self.q_network, self.target_network)
                
                # Update states
                states = next_states
                avail_actions = next_avail_actions
                
                # Check if all games are done
                done = all(dones)
                
                # Decay epsilon
                self.eps = max(self.eps_end, self.eps * self.eps_decay)
            
            # Log episode statistics
            self.episode_rewards.extend(episode_rewards)
            self.episode_lengths.extend(episode_lengths)
            self.games_completed += sum(dones)
            
            # Log to wandb
            if episode_counter % log_freq == 0:
                # wandb.log({
                #     "episode": episode_counter,
                #     "average_reward": np.mean(episode_rewards),
                #     "average_length": np.mean(episode_lengths),
                #     "epsilon": self.eps,
                #     "games_completed": self.games_completed,
                #     "loss": np.mean(self.losses[-100:]) if self.losses else 0
                # })
                
                print(f"Episode {episode_counter}")
                print(f"Average Reward: {np.mean(episode_rewards):.2f}")
                print(f"Average Length: {np.mean(episode_lengths):.2f}")
                print(f"Epsilon: {self.eps:.3f}")
                print(f"Games Completed: {self.games_completed}")
                print(f"Loss: {np.mean(self.losses[-100:]):.4f}" if self.losses else "Loss: N/A")
                print("-" * 50)
            
            # Save model
            if episode_counter % save_freq == 0:
                self.save_model(f"ddqn_model_ep_{episode_counter}.pt")

            if np.mean(episode_rewards) > self.best_average_reward:
                self.best_average_reward = np.mean(episode_rewards)
                self.no_improvement_counter = 0
                self.save_model("ddqn_model_best.pt")
            else:
                self.no_improvement_counter += 1
                print(f"No improvement in average reward for {self.no_improvement_counter} episodes.")
    
    def save_model(self, filename):
        """Save model state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'games_completed': self.games_completed,
            'epsilon': self.eps
        }, filename)
    
    def load_model(self, filename):
        """Load model state"""
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.losses = checkpoint['losses']
        self.games_completed = checkpoint['games_completed']
        self.eps = checkpoint['epsilon']

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize trainer
    trainer = Big2DDQNTrainer(
        state_dim=412,
        action_dim=1695,
        hidden_dim=512,
        lr=2.5e-4,
        gamma=0.995,
        memory_size=50000,
        batch_size=64,
        target_update_freq=100,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.9995,
        n_games=8
    )
    
    # Start training
    try:
        trainer.train(total_steps=1000000, log_freq=100, save_freq=1000)
    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
        trainer.save_model("ddqn_model_interrupted.pt")
    
    # Save final model
    trainer.save_model("ddqn_model_final.pt")