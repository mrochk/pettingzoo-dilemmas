# PettingZoo Dilemmas
**PettingZoo environment for 2 players normal-form games (matrix games).** 

We provide 3 predefined games (Stag Hunt, Prisoner's Dilemma, Matching Pennies), but one can easily create a custom game using the matrix_game module (by providing its own reward matrix).

## Usage

### Installation
```
git clone https://github.com/mrochk/pettingzoo-dilemmas
cd pettingzoo_dilemmas
pip install .
```

You should then be able to run the tests:
```
python3 runtest.py
```

All outputs should be "Passed Parallel API test".

*Example of creating a custom game:*
```python
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
```

## References:

- PettingZoo: https://pettingzoo.farama.org/
- Matrix Games: https://www.matem.unam.mx/~omar/math340/matrix-games.html