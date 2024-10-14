import gym
from gym import spaces
import numpy as np
from game.game_engine import BigTwoGame
from agent.rl_agent import RLAgentPlayer
from agent.random_agent import RandomPlayer
from game.card import Card
class BigTwoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BigTwoEnv, self).__init__()
        # Define action and observation spaces based on the game's requirements
        # For simplicity, we'll use placeholders here

        # Placeholder action space: Discrete set of possible actions
        self.action_space = spaces.Discrete(self.calculate_action_space_size())

        # Placeholder observation space: Binary vector representing the state
        self.observation_space = spaces.MultiBinary(self.calculate_observation_space_size())

        # Initialize game components
        self.players = [RLAgentPlayer("Agent"), RandomPlayer("Opponent1"), RandomPlayer("Opponent2"), RandomPlayer("Opponent3")]
        self.game = BigTwoGame(self.players)
        self.current_state = None  # Will hold the observation
        self.done = False

    def calculate_action_space_size(self):
        # The number of possible actions can be very large
        # For the initial version, we might limit the action space to a manageable size
        return 1000  # Placeholder value

    def calculate_observation_space_size(self):
        # The observation could be the player's hand and the last played hand
        return 52 * 2  # For example, 52 cards for own hand, 52 for last played hand

    def reset(self):
        # Reset the game
        self.game = BigTwoGame(self.players)
        self.game.deal_cards()
        self.game.find_starting_player()
        self.current_state = self.get_state()
        self.done = False
        return self.current_state

    def step(self, action):
        # Map the action to a move
        move = self.action_to_move(action)
        agent = self.players[self.game.current_player_idx]
        if isinstance(agent, RLAgentPlayer):
            agent.set_next_move(move)
        self.game.play_turn()
        self.current_state = self.get_state()
        reward = self.get_reward()
        self.done = self.game.game_over
        info = {}
        return self.current_state, reward, self.done, info

    def render(self, mode='human'):
        # Optional: Render the game state to the console
        pass

    def close(self):
        pass

    def get_state(self):
        # Convert the game state into an observation (e.g., binary vectors)
        # For simplicity, we'll represent the player's hand and the last played hand
        player_hand = self.players[self.game.current_player_idx].hand
        print(player_hand)
        last_hand = self.game.last_played_hand[1] if self.game.last_played_hand else []
        hand_vector = self.cards_to_binary_vector(player_hand)
        last_hand_vector = self.cards_to_binary_vector(last_hand)
        return np.concatenate([hand_vector, last_hand_vector])

    def cards_to_binary_vector(self, cards):
        vector = np.zeros(52, dtype=int)
        for card in cards:
            index = self.card_to_index(card)
            vector[index] = 1
        return vector

    def card_to_index(self, card):
        suit_idx = Card.suit_order[card.suit]
        rank_idx = list(Card.rank_order.values()).index(Card.rank_order[card.rank])
        return suit_idx * 13 + rank_idx

    def action_to_move(self, action):
        # Map the action integer to a specific move (list of Card objects)
        # This mapping needs to be defined based on how the action space is constructed
        # For now, we will return a placeholder
        possible_moves = self.players[self.game.current_player_idx].get_valid_moves(self.game.get_game_state())
        if action < len(possible_moves):
            return possible_moves[action]
        else:
            return 'pass'

    def get_reward(self):
        # Define the reward based on the game's outcome
        if self.done:
            if self.players[self.game.current_player_idx].name == "Agent":
                return 1  # Agent wins
            else:
                return -1  # Agent loses
        else:
            return 0  # No reward during the game