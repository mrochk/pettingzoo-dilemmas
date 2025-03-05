import enum

from pettingzoo_dilemmas.matching_pennies import matching_pennies_v0
from pettingzoo_dilemmas import custom

if __name__ == '__main__':
    print('-- CREATING A MATCHING PENNIES GAME ENV\n')

    env = matching_pennies_v0.env('human', nrounds=5)
    print(env)
    env.reset()

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}

        observations, rewards, term, trunc, info = env.step(actions)

        print('REWARDS:', rewards)

        env.render(); print()

    print([env.cumreward(a) for a in env.possible_agents])

    print('\n-- CREATING A CUSTOM GAME ENV\n')

    class Moves(enum.Enum):
        COOP = 0
        DEFECT = 1
        NONE = 2

    rewardmatrix = {
        (Moves.COOP,   Moves.COOP):   (3, 3),
        (Moves.COOP,   Moves.DEFECT): (0, 5),
        (Moves.DEFECT, Moves.COOP):   (5, 0),
        (Moves.DEFECT, Moves.DEFECT): (1, 1),
    }

    game_factory = custom.MatrixGame(
        agents=['A', 'B'], 
        moves=list(Moves), 
        reward_matrix=rewardmatrix,
    )

    env = game_factory.env(render_mode='human', nrounds=5)
    print(env)
    env.reset()

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}

        observations, rewards, term, trunc, info = env.step(actions)

        print('REWARDS:', rewards)

        env.render(); print()

    print([env.cumreward(a) for a in env.possible_agents])