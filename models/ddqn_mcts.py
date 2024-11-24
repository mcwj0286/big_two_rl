import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import copy

# Constants
STATE_SIZE = 4 * 4 * 15  # 4 planes x 4 colors x 15 card types
ACTION_SIZE = 61
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
MEMORY_SIZE = 100000
MCTS_SIMULATIONS = 50
LEARNING_RATE = 0.00005
C_PUCT = 1.0  # MCTS exploration constant
UPDATE_TARGET_EVERY = 100  # Episodes between target network updates

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class UnoStateEncoder:
    """Handles state encoding for Uno game"""
    def __init__(self):
        self.num_colors = 4
        self.num_card_types = 15
        
    def encode_state(self, player_hand, target_card):
        """Encode game state into neural network input format"""
        state = np.zeros((4, 4, 15))
        
        # Encode player's hand (3 planes for 0, 1, or 2+ cards)
        hand_count = self._count_cards(player_hand)
        for (color_idx, card_type_idx), count in hand_count.items():
            plane_idx = min(count - 1, 2)
            state[plane_idx][color_idx][card_type_idx] = 1
            
        # Encode target card (4th plane)
        color_idx, card_type_idx = self._get_card_indices(target_card)
        state[3][color_idx][card_type_idx] = 1
        
        return state.flatten()
    
    def _count_cards(self, cards):
        count_dict = {}
        for card in cards:
            indices = self._get_card_indices(card)
            count_dict[indices] = count_dict.get(indices, 0) + 1
        return count_dict
    
    def _get_card_indices(self, card):
        """Convert card to color and type indices"""
        # This would need to be implemented based on your specific card representation
        color_map = {'red': 0, 'green': 1, 'blue': 2, 'yellow': 3}
        type_map = {str(i): i for i in range(10)}
        type_map.update({
            'skip': 10, 'reverse': 11, 'draw2': 12,
            'wild': 13, 'wild_draw4': 14
        })
        
        return color_map[card.color], type_map[card.type]

class QNetwork(nn.Module):
    """Neural network for Q-value prediction"""
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, ACTION_SIZE)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class MCTSNode:
    """Node in Monte Carlo Tree Search"""
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.mean_value = 0
        self.prior = 0
        self.legal_actions = None

class DDQNwithMCTS:
    """Main agent class combining DDQN with MCTS"""
    def __init__(self, state_encoder):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_encoder = state_encoder
        
        # Networks
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training components
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.episode_count = 0
        
    def mcts_search(self, state, legal_actions, player_id):
        """Perform MCTS search from current state"""
        root = MCTSNode(state)
        root.legal_actions = legal_actions
        
        for _ in range(MCTS_SIMULATIONS):
            node = root
            search_path = [node]
            
            # Selection and expansion
            while not self.is_terminal(node.state) and node.legal_actions:
                if node.children:
                    action, node = self.select_child(node)
                else:
                    action = self.expand(node)
                    child_state = self.simulate_step(node.state, action, player_id)
                    child = MCTSNode(child_state, parent=node)
                    node.children[action] = child
                    node = child
                search_path.append(node)
            
            # Simulation and backpropagation
            value = self.simulate(node.state, player_id)
            self.backpropagate(search_path, value)
        
        # Calculate Q-values for legal actions
        q_values = {}
        for action in legal_actions:
            if action in root.children:
                child = root.children[action]
                q_values[action] = child.mean_value
            else:
                q_values[action] = self.predict_q_value(state, action)
        
        return q_values
    
    def select_child(self, node):
        """Select child node using UCT formula"""
        def uct_value(child):
            q_value = child.mean_value
            u_value = C_PUCT * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
            return q_value + u_value
        
        return max(node.children.items(), key=lambda x: uct_value(x[1]))
    
    def expand(self, node):
        """Expand node by selecting an unexplored action"""
        # Predict Q-values for all legal actions
        state_tensor = torch.FloatTensor(node.state).to(self.device)
        q_values = self.q_network(state_tensor).detach().cpu().numpy()
        
        # Select unexplored action with highest Q-value
        unexplored_actions = [a for a in node.legal_actions if a not in node.children]
        return max(unexplored_actions, key=lambda a: q_values[a])
    
    def simulate(self, state, player_id):
        """Simulate game from given state until terminal"""
        current_state = copy.deepcopy(state)
        
        while not self.is_terminal(current_state):
            current_player = self.get_current_player(current_state)
            legal_actions = self.get_legal_actions(current_state)
            
            if current_player == player_id:
                # Use network to choose action
                state_tensor = torch.FloatTensor(current_state).to(self.device)
                q_values = self.q_network(state_tensor).detach().cpu().numpy()
                action = legal_actions[q_values[legal_actions].argmax()]
            else:
                # Simulate opponent with random policy
                action = random.choice(legal_actions)
            
            current_state = self.simulate_step(current_state, action, current_player)
            
        return self.get_reward(current_state, player_id)
    
    def backpropagate(self, search_path, value):
        """Backpropagate value through search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            node.mean_value = node.total_value / node.visit_count
    
    def select_action(self, state, legal_actions, player_id):
        """Select action using epsilon-greedy strategy with MCTS"""
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        q_values = self.mcts_search(state, legal_actions, player_id)
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def train_step(self, batch):
        """Perform one training step"""
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (DDQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + GAMMA * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update(self):
        """Update target network and epsilon"""
        self.episode_count += 1
        
        # Update target network
        if self.episode_count % UPDATE_TARGET_EVERY == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def get_shaped_reward(self, state, action, next_state, base_reward, player_id):
        """Shape reward using MCTS results"""
        total_reward = base_reward
        
        if base_reward == 0:  # No immediate reward from environment
            # Run MCTS to estimate action value
            q_values = self.mcts_search(state, [action], player_id)
            mcts_value = q_values[action]
            
            # Shape reward based on MCTS value
            shaped_reward = np.clip(mcts_value * 0.1, -1.0, 1.0)  # Scale factor 0.1
            total_reward += shaped_reward
        
        return total_reward
    
    # Helper methods to be implemented based on your Uno environment
    def is_terminal(self, state):
        raise NotImplementedError
        
    def get_current_player(self, state):
        raise NotImplementedError
        
    def get_legal_actions(self, state):
        raise NotImplementedError
        
    def simulate_step(self, state, action, player_id):
        raise NotImplementedError
        
    def get_reward(self, state, player_id):
        raise NotImplementedError
        
    def predict_q_value(self, state, action):
        raise NotImplementedError