from enum import Enum
from pettingzoo_dilemmas.matrix_game.matrix_game_v0 import MatrixGame

class MatchingPennies(MatrixGame):
    metadata = {'name': 'matching_pennies', 'render_modes': ['human']}

    def __init__(self, nrounds: int = 1):

        class Moves(Enum):
            HEADS = 0
            TAILS = 1

        reward_matrix = {
            (Moves.HEADS, Moves.HEADS): (1, 0),
            (Moves.HEADS, Moves.TAILS): (0, 1),
            (Moves.TAILS, Moves.HEADS): (0, 1),
            (Moves.TAILS, Moves.TAILS): (1, 0),
        }

        agents = ('player_A', 'player_B')

        super().__init__(reward_matrix, Moves, nrounds, agents)
