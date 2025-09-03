# PettingZoo Dilemmas
**PettingZoo environment for 2 players normal-form games (matrix games).** 

We provide 3 predefined games (Stag Hunt, Prisoner's Dilemma, Matching Pennies), but one can easily create a custom game using the `matrix_game` module (by providing its own reward matrix).

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

### Example
```python
import enum
from pettingzoo_dilemmas import matrix_game_v0

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
```

## References:

- PettingZoo: https://pettingzoo.farama.org/
- Matrix Games: https://www.matem.unam.mx/~omar/math340/matrix-games.html