# PettingZoo Dilemmas
**PettingZoo environments for 4 classical matrix games.** 

## Usage

### Installation
```
git clone https://github.com/mrochk/pettingzoo-dilemmas
cd pettingzoo_dilemmas
pip install -r requirements.txt # pettingzoo + gymnasium
pip install .
```

You should then be able to run the tests:
```
python3 runtest.py
```

All outputs should be "Passed Parallel API test".

### Using The Environments
*Example with Stag Hunt:*
```python
from pettingzoo import ParallelEnv

from pettingzoo_dilemmas.stag_hunt import stag_hunt_v0

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
```

## Games Description:

### Stag Hunt

https://en.wikipedia.org/wiki/Stag_hunt

### Matching Pennies

https://en.wikipedia.org/wiki/Stag_hunt


### Prisoner’s Dilemma

https://en.wikipedia.org/wiki/Prisoner%27s_dilemma


### Subsidy Game 

This game has the following payoff matrix:  
| **(12, 12)** | **(0, 11)**  |
|--------|--------|
| **(11, 0)**  | **(10, 10)** |

It is in fact a variant of the Stag Hunt game, where players the *(Stag, Stag)* state is less advantageous, and players lose more when trying to catch the Stag. 
