# big2_env.py

from rlcard.envs import Env
import numpy as np
from collections import OrderedDict

class Big2Env(Env):
    '''Big Two Environment'''

    def __init__(self, config):
        from big2_game import Big2Game
        self.name = 'big2'
        self.game = Big2Game()
        super().__init__(config)
        self.state_shape = [self._get_state_shape() for _ in range(self.num_players)]
        self.action_id_to_action = self.game.get_action_id_to_action()
        self.action_space = list(self.action_id_to_action.keys())
        self._action_to_action_id = {v: k for k, v in self.action_id_to_action.items()}

    def _get_state_shape(self):
        '''Define the shape of the game state'''
        hand_feature_length = 52  # One-hot encoding of hand cards
        last_play_feature_length = 52  # One-hot encoding of last played cards
        other_feature_length = 5  # Additional features (e.g., number of cards left for each player)
        return hand_feature_length + last_play_feature_length + other_feature_length

    def _extract_state(self, state):
        '''Extract the game state for the agent'''

        # One-hot encode the player's hand
        current_hand = self._cards_to_array(state['current_hand'])
        # One-hot encode the last played hand
        last_play = self._cards_to_array(state['last_play'])
        # Additional features
        other_info = np.array(state['other_info'])
        # Concatenate all parts to form the observation
        obs = np.concatenate([current_hand, last_play, other_info])

        legal_actions = self._get_legal_actions()

        extracted_state = OrderedDict({
            'obs': obs,
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': state['legal_actions'],
            'action_record': self.action_recorder
        })
        return extracted_state

    def _get_legal_actions(self):
        '''Get all legal actions for the current state'''
        raw_legal_actions = self.game.get_legal_actions()
        legal_action_ids = {self._action_to_action_id[action]: None for action in raw_legal_actions}
        return legal_action_ids

    def _decode_action(self, action_id):
        '''Map action ID to action in the game'''
        return self.action_id_to_action[action_id]

    def _cards_to_array(self, cards):
        '''Convert list of cards to a one-hot encoded numpy array'''
        card_str_to_id = self.game.get_card_str_to_id()
        array = np.zeros(52, dtype=int)
        for card in cards:
            idx = card_str_to_id[card]
            array[idx] = 1
        return array

    def get_payoffs(self):
        '''Calculate the payoffs for all players'''
        return self.game.get_payoffs()

    def get_perfect_information(self):
        '''Get the perfect information of the current state'''
        state = {
            'players': [{ 'player_id': p.player_id, 'hand': p.hand } for p in self.game.players],
            'history': self.game.history,
            'last_play': self.game.last_play,
            'current_player': self.game.current_player_id,
            'legal_actions': self.game.get_legal_actions()
        }
        return state

def main():
    config = {
        'game_num_players': 4 , 
        'allow_step_back': True
    }
    env = Big2Env(config)
    state, player_id = env.reset()
    print(f"Initial state shape: {state['obs'].shape}")
    print(f"Legal actions: {state['legal_actions']}")
    
if __name__ == '__main__':
    main()