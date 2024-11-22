## This code is for collecting gameplay for offline RL
import os 
import torch
class Big2GameDataCollector:
    def __init__(self, save_dir="gameplay_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Data storage
        self.trajectories = []
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'available_actions': [],
            'values': [],
            'player_turns': []
        }

    def add_transition(self, state, action, reward, next_state, done, 
                      available_actions, value, player_turn):
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['next_states'].append(next_state)
        self.current_trajectory['dones'].append(done)
        self.current_trajectory['available_actions'].append(available_actions)
        self.current_trajectory['values'].append(value)
        self.current_trajectory['player_turns'].append(player_turn)

    def end_game(self):
        # Store completed game
        self.trajectories.append(self.current_trajectory)
        # Reset for next game
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'available_actions': [],
            'values': [],
            'player_turns': []
        }

    def save_data(self, filename):
        data_path = os.path.join(self.save_dir, filename)
        with open(data_path, 'wb') as f:
            torch.save({
                'trajectories': self.trajectories
            }, f)