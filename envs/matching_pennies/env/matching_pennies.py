import functools
from gymnasium import spaces, logger
from pettingzoo import ParallelEnv

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

def env(render_mode: str = None, nrounds: int = 1) -> ParallelEnv:
    internal_render_mode = render_mode
    env = raw_env(render_mode=internal_render_mode, nrounds=nrounds)
    return env

class raw_env(ParallelEnv):
    metadata = {
        "name": "matching_pennies_v0", 
        "render_modes": ["human"],
    }

    _render_mode: str
    _nrounds: int
    _possible_agents: list[str]
    _timestep: int
    _cumulative_rewards: dict[str, int]
    action_spaces: dict[str, spaces.Space]
    observations_spaces: dict[str, spaces.Space]

    def __init__(self, render_mode: str = None, nrounds: int = 1):
        self._render_mode = render_mode
        self._nrounds = nrounds
        self.possible_agents = ["even", "odd"]
        self._timestep = None

        self._cumulative_rewards = {a: 0 for a in self.possible_agents}

        self.action_spaces = {a: spaces.Discrete(2) for a in self.possible_agents}
        self.observation_spaces = {a: spaces.Discrete(1) for a in self.possible_agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self._timestep = 0
        self._cumulative_rewards = {a: 0 for a in self.possible_agents}

        observations = {a: {} for a in self.agents}
        infos = {a: {} for a in self.agents}

        self.state = observations

        return observations, infos  
    
    def step(self, actions):
        if not actions or len(actions) != 2:
            logger.error('Actions must be provided for all agents.')
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

        if self._timestep >= self._nrounds:
            truncations = {a: True for a in self.agents}
            rewards = {a: 0 for a in self.agents}
            self.agents = list()

        self._timestep += 1

        for agent in self.agents:
            self._cumulative_rewards[agent] += rewards[agent]

        # the observation is the action chosen by the other player
        observations = {"even": odd_action, "odd": even_action}
        self.state = observations

        infos = {a: {} for a in self.agents}

        return observations, rewards, terminations, truncations, infos 

    def cumreward(self, agent): return self._cumulative_rewards[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return spaces.Discrete(2) 

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return spaces.Discrete(1)

    def observe(self, agent): 
        # the only observation is the last move of the other player
        return self.state[agent]

    def render(self):
        if self._render_mode is None:
            logger.warn("You are calling render method without specifying any render mode.")
            return
        
        if not self.agents: print("Game Over"); return

        even_move = MOVES[self.state[self.agents[0]]]
        odd_move = MOVES[self.state[self.agents[1]]]

        print(f"Current state: Even played: {even_move}, Odd played: {odd_move}")

    def close(self): return

