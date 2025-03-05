import enum

from pettingzoo_dilemmas import factory

class Moves(enum.Enum):
    COOP = 0
    DEFECT = 1
    NONE = 2

rmatrix = {
    (Moves.COOP,   Moves.COOP):   (3, 3),
    (Moves.COOP,   Moves.DEFECT): (0, 5),
    (Moves.DEFECT, Moves.COOP):   (5, 0),
    (Moves.DEFECT, Moves.DEFECT): (1, 1),
}

fact = factory.MatrixGame(
    agents=['A', 'B'], 
    moves=list(Moves), 
    reward_matrix=rmatrix,
)

env = fact.env(render_mode='human', nrounds=5)
env.reset()

while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}

    observations, rewards, term, trunc, info = env.step(actions)

    print('OBS:', observations)
    print('REWARDS:', rewards)

    env.render(); print()

print([env.cumreward(a) for a in env.possible_agents])