import numpy as np
import torch
from models.ddqn_mcts import DDQNwithMCTS
from game.big2Game import vectorizedBig2Games
import logging
import os
from datetime import datetime

# Training Parameters
NUM_EPISODES = 100000
NUM_PARALLEL_GAMES = 8
SAVE_INTERVAL = 1000
LOG_INTERVAL = 100

class StateEncoder:
    def __init__(self):
        self.state_size = 412
        self.action_size = 1695
    
    def decode_action(self, action_idx):
        # Convert action index to game action format
        from game.enumerateOptions import getOptionNC
        return getOptionNC(action_idx)

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        filename=f'{log_dir}/training_{timestamp}.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def train():
    setup_logging()
    
    # Initialize environment and agent
    env = vectorizedBig2Games(NUM_PARALLEL_GAMES)
    state_encoder = StateEncoder()
    agent = DDQNwithMCTS(state_encoder)
    
    # Training metrics
    episode_rewards = []
    best_reward = float('-inf')
    
    try:
        for episode in range(NUM_EPISODES):
            episode_reward = 0
            done = False
            
            # Get initial state
            player_ids, states, available_actions = env.getCurrStates()
            states = np.squeeze(states)
            available_actions = np.squeeze(available_actions)
            
            while not done:
                # Get actions for all parallel games
                actions = []
                for i in range(NUM_PARALLEL_GAMES):
                    state_dict = agent.create_state_dict(
                        (player_ids[i], states[i], available_actions[i]),
                        states[i],
                        available_actions[i]
                    )
                    legal_actions = np.where(available_actions[i] != float('-inf'))[0]
                    action = agent.select_action(state_dict, legal_actions, player_ids[i])
                    actions.append(action)
                
                # Step the environments
                rewards, dones, infos = env.step(actions)
                next_player_ids, next_states, next_available_actions = env.getCurrStates()
                
                # Store experiences and train
                for i in range(NUM_PARALLEL_GAMES):
                    if rewards[i] != 0:  # Only store meaningful transitions
                        agent.memory.push(
                            states[i],
                            actions[i],
                            rewards[i],
                            next_states[i],
                            dones[i]
                        )
                
                # Train if enough samples
                if len(agent.memory) >= BATCH_SIZE:
                    batch = agent.memory.sample(BATCH_SIZE)
                    loss = agent.train_step(batch)
                
                # Update states
                states = next_states
                available_actions = next_available_actions
                player_ids = next_player_ids
                
                # Check if any game is done
                done = any(dones)
                if done:
                    for info in infos:
                        if info is not None:
                            episode_reward = np.mean(info['rewards'])
            
            # Update target network and epsilon
            agent.update()
            
            # Logging
            episode_rewards.append(episode_reward)
            if episode % LOG_INTERVAL == 0:
                avg_reward = np.mean(episode_rewards[-LOG_INTERVAL:])
                logging.info(f"Episode {episode}/{NUM_EPISODES} - Avg Reward: {avg_reward:.2f}")
                print(f"Episode {episode}/{NUM_EPISODES} - Avg Reward: {avg_reward:.2f}")
            
            # Save model if improved
            if episode % SAVE_INTERVAL == 0:
                avg_reward = np.mean(episode_rewards[-SAVE_INTERVAL:])
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(agent.q_network.state_dict(), f'models/ddqn_mcts_best.pth')
                    logging.info(f"New best model saved with average reward: {avg_reward:.2f}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        env.close()
        # Save final model
        torch.save(agent.q_network.state_dict(), f'models/ddqn_mcts_final.pth')
        print("Training completed")

if __name__ == "__main__":
    train()
