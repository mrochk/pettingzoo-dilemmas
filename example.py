import enum
from pettingzoo_dilemmas import matrix_game_v0

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

    env = matrix_game_v0.env(
        Moves=Moves, 
        reward_matrix=rewardmatrix,
        nrounds=2,
    )

    env.reset()

    while env.agents:
        actions = {a: env.action_space().sample() for a in env.agents}

        observations, rewards, term, trunc, info = env.step(actions)

        env.render(); print()
