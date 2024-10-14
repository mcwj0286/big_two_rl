from game.player import Player
class RLAgentPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.next_move = None

    def set_next_move(self, move):
        self.next_move = move

    def decide_move(self, game_state):
        if self.next_move is not None:
            move = self.next_move
            self.next_move = None
            return move
        else:
            return 'pass'  # Or some default behavior