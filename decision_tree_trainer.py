from game.big2Game import big2Game, vectorizedBig2Games
from models.decision_tree import Big2DecisionTree
from ppo_gameplay_dataset import PPOGameplayDataset
import numpy as np
import torch

class DecisionTreeTrainer:
    def __init__(self, num_games=10, num_parallel=4, hdf5_path=None):
        self.num_games = num_games
        self.env = vectorizedBig2Games(num_parallel)
        self.model = Big2DecisionTree()
        self.hdf5_path = hdf5_path
        
    def collect_ppo_data(self):
        """Collect state-action pairs from PPO gameplay dataset"""
        if not self.hdf5_path:
            raise ValueError("HDF5 path not provided for PPO gameplay data")
            
        dataset = PPOGameplayDataset(self.hdf5_path)
        states = []
        actions = []
        
        # Collect all state-action pairs from the dataset
        for i in range(len(dataset)):
            data = dataset[i]
            states.append(data['states'].numpy())
            actions.append(data['actions'].numpy())
            # if i >500:
            #     break
            
        # Concatenate all sequences
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        
        # Check which action indices are missing
        unique_actions = set(actions)
        missing_actions = set(range(1695)) - unique_actions
        
        # Add synthetic data for missing actions
        if missing_actions:
            print(f"Adding synthetic data for {len(missing_actions)} missing actions")
            zero_state = np.zeros(412)  # Zero state vector
            for action in missing_actions:
                states = np.vstack([states, zero_state])
                actions = np.append(actions, action)
        
        dataset.close()
        return states, actions

    def collect_game_data(self):
        """Collect state-action pairs from random gameplay"""
        states = []
        actions = []
        games_completed = 0
        
        while games_completed < self.num_games:
            player_ids, curr_states, avail_actions = self.env.getCurrStates()
            
            # Choose random actions from available actions
            game_actions = []
            for avail in avail_actions:
                valid_actions = np.where(avail[0] > -np.inf)[0]
                action = np.random.choice(valid_actions)
                game_actions.append(action)
            
            # Store state-action pairs
            for state, action in zip(curr_states, game_actions):
                states.append(state[0])
                actions.append(action)
                
            # Take actions in environment
            rewards, dones, infos = self.env.step(game_actions)
            
            # Count completed games
            games_completed += sum(dones)
            
        return np.array(states), np.array(actions)
    
    def train(self, save_path=None, use_ppo_data=True):
        """Train the decision tree model on collected game data"""
        # Collect training data
        print("Collecting game data...")
        if use_ppo_data:
            states, actions = self.collect_ppo_data()
        else:
            states, actions = self.collect_game_data()
        
        print(f"Training decision tree on {len(states)} state-action pairs...")
        self.model.train(states, actions)
        
        if save_path:
            self.model.save(save_path)
            
    def test_model(self, num_games=100):
        """Test the trained model's performance"""
        rewards_history = []
        games_completed = 0
        
        while games_completed < num_games:
            player_ids, curr_states, avail_actions = self.env.getCurrStates()
            
            # Get model predictions
            game_actions = []
            for state, avail in zip(curr_states, avail_actions):
                action = self.model.predict(state[0], avail[0])
                game_actions.append(action)
                
            # Take actions
            rewards, dones, infos = self.env.step(game_actions)
            
            # Record completed game rewards
            for done, reward in zip(dones, rewards):
                if done:
                    rewards_history.append(reward)
                    games_completed += 1
                    
        return np.mean(rewards_history, axis=0)

def main():
    # Create and train model using PPO gameplay data
    trainer = DecisionTreeTrainer(
        num_games=1000,
        hdf5_path='output/pytorch_ppo_trajectories.hdf5'
    )
    
    # Train using PPO data
    trainer.train(save_path="/output/big2_decision_tree.pkl", use_ppo_data=True)
    
    # Test model performance
    mean_rewards = trainer.test_model(num_games=100)
    print(f"Average rewards per player: {mean_rewards}")

if __name__ == "__main__":
    main()
