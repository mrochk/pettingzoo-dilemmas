import functools
from gymnasium import spaces, logger
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

HEADS = 0
TAILS = 1
MOVES = ["HEADS", "TAILS"]

REWARD_MAP = {
# players reward: (even, odd)
    (HEADS, HEADS): (1, -1),
    (HEADS, TAILS): (-1, 1),
    (TAILS, HEADS): (-1, 1),
    (TAILS, TAILS): (1, -1),
}

def env(render_mode : str = None, nrounds: int = 1):
    internal_render_mode = render_mode
    env = raw_env(render_mode=internal_render_mode, nrounds=nrounds)
    return env

class raw_env(ParallelEnv):
    metadata = {"name": "matching_pennies_v0", "render_modes": ["human"]}

    def __init__(self, render_mode : str = None, nrounds: int = 1):
        self.render_mode = render_mode
        self.nrounds = nrounds
        self.possible_agents = ["even", "odd"]
        self.timestep = None

        self.action_spaces = {a: spaces.Discrete(2) for a in self.possible_agents}
        self.observation_spaces = {a: spaces.Discrete(1) for a in self.possible_agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.timestep = 0

        observations = {a: {} for a in self.agents}
        infos = {a: {} for a in self.agents}

        self.state = observations

        return observations, infos  
    
    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        even_action = actions["even"]
        odd_action = actions["odd"]

        rewards = {
            "even": REWARD_MAP[(even_action, odd_action)][0],
            "odd": REWARD_MAP[(even_action, odd_action)][1]
        }

        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        if self.timestep >= self.nrounds:
            truncations = {a: True for a in self.agents}
            rewards = {a: 0 for a in self.agents}
            self.agents = list()

        self.timestep += 1

        # the observation is the action chosen by the other player
        observations = {"even": odd_action, "odd": even_action}
        self.state = observations

        infos = {a: {} for a in self.agents}

        return observations, rewards, terminations, truncations, infos 

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return spaces.Discrete(2) 

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return spaces.Discrete(1)

    def observe(self, agent):
        return self.ob

    def render(self):
        if self.render_mode is None:
            logger.warn("You are calling render method without specifying any render mode.")
            return
        
        if not self.agents: print("Game Over"); return

        even_move = MOVES[self.state[self.agents[0]]]
        odd_move = MOVES[self.state[self.agents[1]]]

        print(f"Current state: Even: {even_move}, Odd: {odd_move}")

    def close(self): return

