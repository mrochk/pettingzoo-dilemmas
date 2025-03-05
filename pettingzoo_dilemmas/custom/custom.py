import functools
from gymnasium import spaces, logger
from pettingzoo import ParallelEnv
import enum

class MatrixGame:
    def __init__(self, agents: list[str], moves: list[enum.Enum], reward_matrix: dict):
        self.AGENTS = agents
        self.MOVES = moves
        self.REWARD_MATRIX = reward_matrix

    def env(self, render_mode: str = None, nrounds: int = 1) -> ParallelEnv:
        internal_render_mode = render_mode

        env = self.raw_env(
            render_mode=internal_render_mode, 
            nrounds=nrounds,
            possible_agents=self.AGENTS,
            moves=self.MOVES,
            reward_map=self.REWARD_MATRIX,
        )

        return env

    class raw_env(ParallelEnv):
        metadata = {"name": "custom_matrix_game", "render_modes": ["human"]}

        _render_mode: str
        _nrounds: int
        _possible_agents: list[str]
        _timestep: int
        _cumulative_rewards: dict[str, int]

        action_spaces: dict[str, spaces.Space]
        observations_spaces: dict[str, spaces.Space]

        def __init__(self, render_mode: str, nrounds: int, possible_agents: list[str], moves: list[enum.Enum],
                     reward_map: dict[tuple[enum.Enum, enum.Enum], float]):

            self._render_mode = render_mode
            self._nrounds = nrounds
            self._timestep = None
            self.possible_agents = possible_agents
            self.MOVES = moves
            self.REWARD_MAP = reward_map

            self._cumulative_rewards = {a: 0 for a in self.possible_agents}

            self.action_spaces = {a: spaces.Discrete(len(moves)) for a in self.possible_agents}
            self.observation_spaces = {a: spaces.Discrete(1) for a in self.possible_agents}

        def reset(self, seed=None, options=None):
            self.agents = self.possible_agents.copy()
            self._timestep = 0
            self._cumulative_rewards = {a: 0 for a in self.possible_agents}

            # last move must always be NONE
            assert self.MOVES[-1].name == 'NONE'
            observations = {a: self.MOVES[-1] for a in self.agents}

            infos = {a: {} for a in self.agents}

            self.state = observations

            return observations, infos  
    
        def step(self, actions):
            if not actions or len(actions) != 2:
                logger.error('Actions must be provided for all agents.')
                self.agents = []
                return {}, {}, {}, {}, {}

            for agent, action in zip(actions.keys(), actions.values()):
                assert agent in self.agents
                assert action in self.action_space(agent)

            terminations = {a: False for a in self.agents}
            truncations = {a: False for a in self.agents}
            observations = {a: self.MOVES[-1] for a in self.agents}
            infos = {a: {} for a in self.agents}
            rewards = {a: 0 for a in self.agents}

            if self._timestep >= self._nrounds:
                truncations = {a: True for a in self.agents}
                self.agents = list()
                return observations, rewards, terminations, truncations, infos 

            A_action = self.MOVES[actions[self.agents[0]]]
            B_action = self.MOVES[actions[self.agents[1]]]

            rewards = {
                self.agents[0]: self.REWARD_MAP[(A_action, B_action)][0],
                self.agents[1]: self.REWARD_MAP[(A_action, B_action)][1]
            }

            for agent in self.agents:
                self._cumulative_rewards[agent] += rewards[agent]

            # the observation is the action chosen by the other player
            observations = {self.agents[0]: B_action, self.agents[1]: A_action}

            # the state is the last move played by each player
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

            A_move = self.state[self.agents[0]]
            B_move = self.state[self.agents[1]]

            print(f"Current state: Even played: {A_move}, Odd played: {B_move}")

        def close(self): return




