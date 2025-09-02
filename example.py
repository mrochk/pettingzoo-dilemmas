import enum
from pettingzoo_dilemmas import matrix_game_v0
from pettingzoo_dilemmas import prisoners_dilemma_v0

if __name__ == '__main__':

    class Moves(enum.Enum):
        MoveA = 0
        MoveB = 1

    rewardmatrix = {
        (Moves.MoveA,   Moves.MoveA):   (3, 3),
        (Moves.MoveA,   Moves.MoveB): (0, 5),
        (Moves.MoveB, Moves.MoveA):   (5, 0),
        (Moves.MoveB, Moves.MoveB): (1, 1),
    }

    agents = ['A', 'B']

    env = matrix_game_v0.MatrixGame(
        agents=agents, 
        Moves=Moves, 
        reward_matrix=rewardmatrix,
    )

    env.reset()

    while env.agents:
        actions = {a: env.action_space().sample() for a in env.agents}

        observations, rewards, term, trunc, info = env.step(actions)

        env.render(); print()
