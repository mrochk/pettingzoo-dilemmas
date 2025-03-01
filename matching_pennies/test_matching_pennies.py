from pettingzoo.test import parallel_api_test
from pettingzoo import ParallelEnv

import matching_pennies

if __name__ == '__main__':
    env : ParallelEnv
    env = matching_pennies.env('human')
    parallel_api_test(env)

    obs, infos = env.reset()

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        a, b, c, d, e = env.step(actions)

        print(a)
        print(b)
        print(c)
        print(d)
        print(e)