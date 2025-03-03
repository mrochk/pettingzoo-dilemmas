import functools
from gymnasium import spaces, logger
from pettingzoo import ParallelEnv

COOPERATE = 0
DEFECT    = 1
NONE      = 2

MOVES = ["COOPERATE", "DEFECT", "NONE"]

REWARD_MAP = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT):    (0, 5),
    (DEFECT,    COOPERATE): (5, 0),
    (DEFECT,    DEFECT):    (1, 1),
}

def env(render_mode: str = None, nrounds: int = 1) -> ParallelEnv:
    internal_render_mode = render_mode
    env = raw_env(render_mode=internal_render_mode, nrounds=nrounds)
    return env

class raw_env(ParallelEnv):
    metadata = {
        "name": "prisoners_dilemma_v0", 
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
        self.possible_agents = ["prisoner_A", "prisoner_B"]
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

        A_action = actions[self.agents[0]]
        B_action = actions[self.agents[1]]

        rewards = {
            self.agents[0]: REWARD_MAP[(A_action, B_action)][0],
            self.agents[1]: REWARD_MAP[(A_action, B_action)][1]
        }

        for agent in self.agents:
            self._cumulative_rewards[agent] += rewards[agent]

        # the observation is the action chosen by the other player
        observations = {self.agents[0]: B_action, self.agents[1]: A_action}

        # the "state" is the last move played by each player
        self.state = {self.agents[0]: A_action, self.agents[1]: B_action}

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

        A_move = MOVES[self.state[self.agents[0]]]
        B_move = MOVES[self.state[self.agents[1]]]

        print(f"Current state: Prisoner_A played: {A_move}, Prisoner_B played: {B_move}")

    def close(self): return




