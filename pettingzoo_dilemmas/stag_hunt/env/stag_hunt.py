import functools
from gymnasium import spaces, logger
from pettingzoo import ParallelEnv

STAG = 0
HARE = 1
NONE = 2

MOVES = ["STAG", "HARE", "NONE"]

REWARD_MAP = {
    (STAG, STAG): (1, 1),
    (STAG, HARE): (0, 2/3),
    (HARE, STAG): (2/3, 0),
    (HARE, HARE): (2/3, 2/3),
}

def env(render_mode: str = None, nrounds: int = 1) -> ParallelEnv:
    internal_render_mode = render_mode
    env = raw_env(render_mode=internal_render_mode, nrounds=nrounds)
    return env

class raw_env(ParallelEnv):
    metadata = {
        "name": "stag_hunt_v0", 
        "render_modes": ["human"],
    }

    _render_mode: str
    _nrounds: int
    _possible_agents: list[str]
    _timestep: int
    _cumulative_rewards: dict[str, int]
    action_spaces: dict[str, spaces.Space]
    observations_spaces: dict[str, spaces.Space]

    def __init__(self, render_mode: str, nrounds: int):
        self._render_mode = render_mode
        self._nrounds = nrounds
        self.possible_agents = ["hunter_A", "hunter_B"]
        self._timestep = None

        self._cumulative_rewards = {a: 0 for a in self.possible_agents}

        self.action_spaces = {a: spaces.Discrete(2) for a in self.possible_agents}
        self.observation_spaces = {a: spaces.Discrete(1) for a in self.possible_agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self._timestep = 0
        self._cumulative_rewards = {a: 0 for a in self.possible_agents}

        observations = {a: NONE for a in self.agents}
        infos = {a: {} for a in self.agents}

        self.state = observations

        return observations, infos  
    
    def step(self, actions):
        if not actions or len(actions) != 2:
            logger.error('Actions must be provided for all agents.')
            self.agents = []
            return {}, {}, {}, {}, {}

        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        observations = {a: NONE for a in self.agents}
        infos = {a: {} for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        if self._timestep >= self._nrounds:
            truncations = {a: True for a in self.agents}
            self.agents = list()
            return observations, rewards, terminations, truncations, infos 

        hA_action = actions["hunter_A"]
        hB_action = actions["hunter_B"]

        rewards = {
            self.agents[0]: REWARD_MAP[(hA_action, hB_action)][0],
            self.agents[1]: REWARD_MAP[(hA_action, hB_action)][1]
        }

        for agent in self.agents:
            self._cumulative_rewards[agent] += rewards[agent]

        # the observation is the action chosen by the other player
        observations = {self.agents[0]: hB_action, self.agents[1]: hA_action}

        # the "state" is the last move played by each player
        self.state = {self.agents[0]: hA_action, self.agents[1]: hB_action}

        self._timestep += 1

        return observations, rewards, terminations, truncations, infos 

    def cumreward(self, agent): return self._cumulative_rewards[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return spaces.Discrete(2) 

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return spaces.Discrete(1)

    def render(self):
        if self._render_mode is None:
            logger.warn("You are calling render method without specifying any render mode.")
            return
        
        if not self.agents: print("Game Over"); return

        hA_move = MOVES[self.state[self.agents[0]]]
        hB_move = MOVES[self.state[self.agents[1]]]

        print(f"Current state: Hunter A played: {hA_move}, Hunter B played: {hB_move}")

    def close(self): return


