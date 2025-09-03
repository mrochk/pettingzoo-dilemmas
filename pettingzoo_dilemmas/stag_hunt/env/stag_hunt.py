from enum import Enum
from pettingzoo_dilemmas.matrix_game.matrix_game_v0 import MatrixGame

class StagHunt(MatrixGame):
    metadata = {'name': 'stag_hunt', 'render_modes': ['human']}

    def __init__(self, nrounds: int = 1):

        class Moves(Enum):
            STAG = 0
            HARE = 1

        reward_matrix = {
            (Moves.STAG, Moves.STAG): (10, 10),
            (Moves.STAG, Moves.HARE): (1,   8),
            (Moves.HARE, Moves.STAG): (8,   1),
            (Moves.HARE, Moves.HARE): (5,   5),
        }

        agents = ('player_A', 'player_B')

        super().__init__(reward_matrix, Moves, nrounds, agents)
