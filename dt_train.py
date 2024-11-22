import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decision_transformer import DecisionTransformer
from game.big2Game import vectorizedBig2Games
from collections import defaultdict
import os 
class Big2DecisionTransformerTrainer:
    def __init__(
        self,
        state_dim=412,
        act_dim=1695,
        n_blocks=6,
        h_dim=128,
        context_len=20,
        n_heads=8,
        drop_p=0.1,
        n_games=8,
        batch_size=64,
        learning_rate=1e-4,
        max_ep_len=200,
        device='cuda' if torch.cuda.is_available() else 'cpu'
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
            lr=learning_rate
        )

        self.env = vectorizedBig2Games(n_games)
        
        # Separate buffers for each player
        self.trajectory_buffer = {
            1: [],
            2: [],
            3: [],
            4: []
        }

    def collect_trajectories(self, n_games):
        """Collect game trajectories for all players"""
        for _ in range(n_games):
            # Initialize trajectory storage for each player
            current_trajectories = {
                1: defaultdict(list),
                2: defaultdict(list),
                3: defaultdict(list),
                4: defaultdict(list)
            }
            
            # Reset environment
            curr_gos, curr_states, curr_avail_acs = self.env.getCurrStates()
            timestep = 0
            game_done = False
            
            while not game_done and timestep < self.max_ep_len:
                # For each active player
                for player_idx in range(self.n_games):
                    player = curr_gos[player_idx]
                    
                    # Store state and available actions
                    current_trajectories[player]['states'].append(curr_states[player_idx])
                    current_trajectories[player]['available_actions'].append(curr_avail_acs[player_idx])
                    current_trajectories[player]['timesteps'].append(timestep)
                    
                    # Get action (random valid action during collection)
                    valid_actions = np.where(curr_avail_acs[player_idx] == 1)[0]
                    
                    # Check if there are valid actions
                    if len(valid_actions) == 0:
                        print(f"Warning: No valid actions for player {player} at timestep {timestep}")
                        # We should investigate why this happens
                        # For now, let's print debug information
                        print(f"Current game state: {curr_states[player_idx]}")
                        print(f"Available actions mask: {curr_avail_acs[player_idx]}")
                        # Choose pass action or default action
                        action = 1694  # Pass action index, verify this is correct for your game
                    else:
                        action = np.random.choice(valid_actions)
                    
                    # Take action
                    try:
                        rewards, dones, infos = self.env.step([action])
                    except Exception as e:
                        print(f"Error taking action {action}: {e}")
                        print(f"Player: {player}, Valid actions: {valid_actions}")
                        raise e
                    
                    # Store action and intermediate reward
                    current_trajectories[player]['actions'].append(action)
                    current_trajectories[player]['rewards'].append(0)  # Store intermediate reward
                    
                    game_done = any(dones)
                    if game_done:
                        # Update final rewards for all players
                        for p in range(1, 5):
                            if len(current_trajectories[p]['rewards']) > 0:
                                current_trajectories[p]['rewards'][-1] = rewards[player_idx][p-1]
                        break
                    
                    # Get next state
                    curr_gos, curr_states, curr_avail_acs = self.env.getCurrStates()
                
                timestep += 1
            
            # Add completed trajectories to buffer
            for player in range(1, 5):
                if len(current_trajectories[player]['states']) > 0:
                    self.trajectory_buffer[player].append(dict(current_trajectories[player]))

            # Print debug information for completed game
            print(f"Game completed at timestep {timestep}")
            print(f"Trajectory lengths: {[len(current_trajectories[p]['states']) for p in range(1, 5)]}")

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
            (self.batch_size, self.context_len, 412),
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
        attention_mask = torch.ones(
            (self.batch_size, self.context_len),
            dtype=torch.bool,
            device=self.device
        )

        for i, traj in enumerate(batch_trajectories):
            # Get trajectory length
            traj_length = len(traj['states'])
            
            # Calculate returns-to-go
            returns = torch.tensor(traj['rewards'], device=self.device)
            returns_to_go_all = torch.cumsum(returns.flip(0), 0).flip(0)
            
            # Handle trajectories shorter than context_len
            if traj_length >= self.context_len:
                start_idx = np.random.randint(0, traj_length - self.context_len + 1)
                end_idx = start_idx + self.context_len
                attention_mask[i] = 1
            else:
                start_idx = 0
                end_idx = traj_length
                attention_mask[i, traj_length:] = 0
                
            # Fill batch tensors
            states[i, :end_idx-start_idx] = torch.tensor(
                traj['states'][start_idx:end_idx],
                device=self.device
            )
            actions[i, :end_idx-start_idx] = torch.tensor(
                traj['actions'][start_idx:end_idx],
                device=self.device
            )
            returns_to_go[i, :end_idx-start_idx] = returns_to_go_all[start_idx:end_idx].unsqueeze(-1)
            timesteps[i, :end_idx-start_idx] = torch.tensor(
                traj['timesteps'][start_idx:end_idx],
                device=self.device
            )

        return timesteps, states, actions, returns_to_go, attention_mask

    def train_step(self, player):
        """Perform one training step for specific player"""
        self.dt.train()
        
        batch = self.prepare_batch(player)
        if batch is None:
            return None
            
        timesteps, states, actions, returns_to_go, attention_mask = batch
        
        # Forward pass
        state_preds, action_preds, return_preds = self.dt(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
        )

        # Calculate losses using attention mask
        action_loss = F.cross_entropy(
            action_preds.reshape(-1, 1695)[attention_mask.reshape(-1)],
            actions.reshape(-1)[attention_mask.reshape(-1)]
        )
        
        state_loss = F.mse_loss(
            state_preds[attention_mask],
            states[attention_mask]
        )
        
        return_loss = F.mse_loss(
            return_preds[attention_mask],
            returns_to_go[attention_mask]
        )
        
        loss = action_loss + 0.1 * state_loss + 0.1 * return_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dt.parameters(), 0.25)
        self.optimizer.step()

        return {
            'action_loss': action_loss.item(),
            'state_loss': state_loss.item(),
            'return_loss': return_loss.item(),
            'total_loss': loss.item()
        }

    def save_checkpoint(self, current_loss):
        """
        Save model checkpoint if there's an improvement
        """
        if current_loss < self.best_loss:
            print(f"Loss improved from {self.best_loss:.6f} to {current_loss:.6f}")
            print(f"Saving model to {self.best_model_path}")
            # Delete previous best model if it exists
            if os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            # Save new best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.dt.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': current_loss,
            }, self.best_model_path)
            self.best_loss = current_loss
            return True
        return False

    def train(self, n_epochs=100, games_per_epoch=100):
        """Main training loop"""
        print(f"Number of epochs: {n_epochs}")
        print(f"Games per epoch: {games_per_epoch}")
        
        # Validate environment
        curr_gos, curr_states, curr_avail_acs = self.env.getCurrStates()
        print(f"Initial state shapes:")
        print(f"Current players: {curr_gos.shape}")
        print(f"States: {curr_states.shape}")
        print(f"Available actions: {curr_avail_acs.shape}")
        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")
            
            # Collect new trajectories
            self.collect_trajectories(games_per_epoch)
            
            # Train on each player's data
            epoch_losses = defaultdict(list)
            total_epoch_loss = 0
            total_valid_players = 0

            for player in range(1, 5):
                player_losses = []
                for _ in range(games_per_epoch // 4):
                    losses = self.train_step(player)
                    if losses is not None:
                        player_losses.append(losses['total_loss'])
                
                if player_losses:  # If we have valid losses for this player
                    avg_player_loss = np.mean(player_losses)
                    epoch_losses[player] = avg_player_loss
                    total_epoch_loss += avg_player_loss
                    total_valid_players += 1
                    print(f"Player {player} Average Loss = {avg_player_loss:.6f}")

            # Only compute average loss if we have valid losses
            if total_valid_players > 0:
                average_epoch_loss = total_epoch_loss / total_valid_players
                print(f"Epoch {epoch} Average Loss Across Players = {average_epoch_loss:.6f}")
                
                # Try to save checkpoint
                improved = self.save_checkpoint(average_epoch_loss)
                if improved:
                    print(f"New best model saved at epoch {epoch}")

    def load_best_model(self):
        """
        Load the best model checkpoint
        """
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            self.dt.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_loss = checkpoint['loss']
            print(f"Loaded best model with loss {self.best_loss:.6f}")
            return True
        print("No checkpoint found")
        return False
    

if __name__ == "__main__":
    trainer = Big2DecisionTransformerTrainer()
    trainer.train(n_epochs=100, games_per_epoch=100)