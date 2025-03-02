from pettingzoo.test import parallel_api_test

import prisoners_dilemma_v0

if __name__ == '__main__':
    env = prisoners_dilemma_v0.env('human', nrounds=10)

    parallel_api_test(env)
    print()

    obs, infos = env.reset()

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        o, r, te, tr, _ = env.step(actions)
        env.render()

    for agent in env.possible_agents:
        print(f'Cumulative reward of {agent}: {env.cumreward(agent)}')
