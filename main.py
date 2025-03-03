from pettingzoo import ParallelEnv

from pettingzoo_dilemmas.stag_hunt import stag_hunt_v0

if __name__ == '__main__':
    env : ParallelEnv = stag_hunt_v0.env(render_mode='human', nrounds=3)
    env.reset()


    print(f'Agents: {env.agents}\n')

    t = 1
    while env.agents:
        # each agent chooses an action
        actions = {a: env.action_space(a).sample() for a in env.agents}

        observations, rewards, terminated, truncated, _ = env.step(actions)

        print(f'At timestep {t}:')
        env.render()
        print(f'Rewards: {rewards}\n')

        t += 1

    for agent in env.possible_agents:
        print(f'Cumulative reward  of {agent} = {env.cumreward(agent) : .2f}.')