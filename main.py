from dilemmas.matching_pennies import matching_pennies_v0
from dilemmas.prisoners_dilemma import prisoners_dilemma_v0
from dilemmas.stag_hunt import stag_hunt_v0
from dilemmas.subsidy_game import subsidy_game_v0

from pettingzoo import ParallelEnv

if __name__ == '__main__':
    env : ParallelEnv = subsidy_game_v0.env()
    env.reset()

    print(env)