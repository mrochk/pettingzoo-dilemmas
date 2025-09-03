from enum import Enum
from pettingzoo_dilemmas.matrix_game.matrix_game_v0 import MatrixGame

class PrisonersDilemma(MatrixGame):
    metadata = {'name': 'prisoners_dilemma', 'render_modes': ['human']}

    def __init__(self, nrounds: int = 1):

        class Moves(Enum):
            Cooperate = 0
            Defect = 1

        reward_matrix = {
            (Moves.Cooperate, Moves.Cooperate): (3, 3),
            (Moves.Cooperate, Moves.Defect):    (0, 5),
            (Moves.Defect,    Moves.Cooperate): (5, 0),
            (Moves.Defect,    Moves.Defect):    (1, 1),
        }

        agents = ('prisoner_A', 'prisoner_B')

        super().__init__(reward_matrix, Moves, nrounds, agents)
