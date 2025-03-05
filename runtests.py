import enum
from pettingzoo.test import parallel_api_test

from pettingzoo_dilemmas import matching_pennies_v0
from pettingzoo_dilemmas import prisoners_dilemma_v0
from pettingzoo_dilemmas import stag_hunt_v0
from pettingzoo_dilemmas import subsidy_game_v0
from pettingzoo_dilemmas import custom

def test_env(environment):
    print(environment())
    for nrounds in [1, 10, 100, 1000]:
        env = environment(nrounds=nrounds)
        parallel_api_test(env)

if __name__ == '__main__':
    for env in [matching_pennies_v0.env, prisoners_dilemma_v0.env, stag_hunt_v0.env, subsidy_game_v0.env]:
        test_env(env); print()

    class Moves(enum.Enum):
        A = 0
        B = 1
        NONE = 2

    rewardmatrix = {
        (Moves.A, Moves.A): (3, 3),
        (Moves.A, Moves.B): (0, 5),
        (Moves.B, Moves.A): (5, 0),
        (Moves.B, Moves.B): (1, 1),
    }

    game_factory = custom.MatrixGame(
        agents=['A', 'B'], 
        moves=list(Moves), 
        reward_matrix=rewardmatrix,
    )

    test_env(game_factory.env)
