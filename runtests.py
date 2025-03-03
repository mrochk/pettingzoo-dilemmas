from gymnasium import logger
from pettingzoo.test import parallel_api_test

from pettingzoo_dilemmas import matching_pennies_v0
from pettingzoo_dilemmas import prisoners_dilemma_v0
from pettingzoo_dilemmas import stag_hunt_v0
from pettingzoo_dilemmas import subsidy_game_v0

def test_env(environment):
    print(environment())
    for nrounds in [1, 10, 100, 1000]:
        env = environment(nrounds=nrounds)
        parallel_api_test(env)

if __name__ == '__main__':
    for env in [matching_pennies_v0.env, prisoners_dilemma_v0.env, stag_hunt_v0.env, subsidy_game_v0.env]:
        test_env(env); print()
