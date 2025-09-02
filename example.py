import enum
from pettingzoo_dilemmas import matrix_game_v0

if __name__ == '__main__':

    class Moves(enum.Enum):
        COOP = 0
        DEFECT = 1

    rewardmatrix = {
        (Moves.COOP,   Moves.COOP):   (3, 3),
        (Moves.COOP,   Moves.DEFECT): (0, 5),
        (Moves.DEFECT, Moves.COOP):   (5, 0),
        (Moves.DEFECT, Moves.DEFECT): (1, 1),
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