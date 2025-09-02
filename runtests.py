import enum
from pettingzoo.test import parallel_api_test
from pettingzoo_dilemmas import (
    matrix_game_v0, 
    prisoners_dilemma_v0,
    matching_pennies_v0,
    stag_hunt_v0,
)

if __name__ == '__main__':
    class Moves(enum.Enum):
        A = 0
        B = 1

    rewardmatrix = {
        (Moves.A, Moves.A): (3, 3),
        (Moves.A, Moves.B): (0, 5),
        (Moves.B, Moves.A): (5, 0),
        (Moves.B, Moves.B): (1, 1),
    }

    for nrounds in [1, 1000]:

        game = matrix_game_v0.env(
            agents=['A', 'B'], 
            Moves=Moves, 
            reward_matrix=rewardmatrix,
            nrounds=nrounds,
        )

        parallel_api_test(game)

    print()

    for nrounds in [1, 1000]:
        game = prisoners_dilemma_v0.env(nrounds=nrounds)
        parallel_api_test(game)

    print()

    for nrounds in [1, 1000]:
        game = matching_pennies_v0.env(nrounds=nrounds)
        parallel_api_test(game)

    print()

    for nrounds in [1, 1000]:
        game = stag_hunt_v0.env(nrounds=nrounds)
        parallel_api_test(game)