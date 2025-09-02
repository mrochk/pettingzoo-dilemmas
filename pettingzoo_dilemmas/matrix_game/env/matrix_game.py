from enum import EnumType
from functools import lru_cache
from gymnasium import spaces, logger
from pettingzoo import ParallelEnv

class MatrixGame(ParallelEnv):

    metadata = {
        'name': 'matrix_game', 
        'render_modes': ['human'],
    }

    def __init__(
            self, 
            reward_matrix: dict[tuple, tuple], 
            Moves: EnumType, 
            nrounds: int = 1, 
            agents: tuple = ('agent_A', 'agent_B')
        ):

        assert len(Moves) > 0
        assert isinstance(Moves, EnumType)
        assert len(reward_matrix.keys()) == len(Moves) * 2
        assert len(agents) == 2
        assert nrounds > 0

        self._reward_matrix  = reward_matrix
        self._Moves          = Moves
        self._nrounds        = nrounds
        self.possible_agents = agents
        self._timestep       = None
        self._state          = None
        self._cumrewards     = None
        self.agents          = []

    def reset(self, seed = None, options = None):
        self.agents = self.possible_agents.copy()
        self._timestep = 0
        self._cumrewards = {agent: 0 for agent in self.possible_agents}
        self._state = {agent: None for agent in self.possible_agents}

        observations = {agent: None for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        return observations, infos

    def step(self, actions):
        self._timestep += 1 # init at 0

        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        observations = {a: None for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        rewards = {a: 0 for a in self.possible_agents}

        if self._timestep > self._nrounds:
            truncations = {a: True for a in self.possible_agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos 

        agent_A = self.agents[0]
        agent_B = self.agents[1]

        action_A = actions[agent_A]
        action_B = actions[agent_B]

        if not isinstance(action_A, self._Moves): action_A = self._Moves(action_A)
        if not isinstance(action_B, self._Moves): action_B = self._Moves(action_B)

        payoff_A, payoff_B = self._reward_matrix[(action_A, action_B)]
        rewards = {agent_A: payoff_A, agent_B: payoff_B}

        self._cumrewards[agent_A] += rewards[agent_A]
        self._cumrewards[agent_B] += rewards[agent_B]

        # the "state" is the last move played by both players, with their cumulative reward
        state_A = (action_A, action_B, self.cumulative_reward(agent_A), self.cumulative_reward(agent_B), self._timestep, self._nrounds)
        state_B = (action_B, action_A, self.cumulative_reward(agent_B), self.cumulative_reward(agent_A), self._timestep, self._nrounds)

        self._state = {agent_A: state_A, agent_B: state_B}
        observations = self._state.copy()

        return observations, rewards, terminations, truncations, infos

    def cumulative_reward(self, agent: str): return self._cumrewards[agent]

    @lru_cache()
    def action_space(self, agent = None):
        '''Both agents can play equal number of actions so action space is the same.'''
        return spaces.Discrete(len(self._Moves)) 

    @lru_cache()
    def observation_space(self, agent = None): 
        maxreward = max(
            max(_[0] for _ in self._reward_matrix.values()),
            max(_[1] for _ in self._reward_matrix.values()),
        )

        minreward = max(
            min(_[0] for _ in self._reward_matrix.values()),
            min(_[1] for _ in self._reward_matrix.values()),
        )

        maximum = max(self._nrounds * maxreward, 0)
        minumum = min(self._nrounds * minreward, 0)

        return spaces.Tuple((
            spaces.Discrete(len(self._Moves)),
            spaces.Discrete(len(self._Moves)),
            spaces.Box(low=minumum, high=maximum, dtype=int),
            spaces.Box(low=minumum, high=maximum, dtype=int),
            spaces.Discrete(self._nrounds+1),
            spaces.Box(low=1, high=int(10e9), dtype=int),
        ))

    def render(self):
        if self._timestep is None: return
        if self._timestep == 0: print('Game not started yet.\n'); return
        if self._timestep > self._nrounds: print('Game is over.\n'); return

        agent_A = self.possible_agents[0]
        agent_B = self.possible_agents[1]

        move_A = self._state[agent_A]
        move_B = self._state[agent_B]

        move_A = move_A if move_A is None else move_A[0]
        move_B = move_B if move_B is None else move_B[0]

        cr_A = self.cumulative_reward(agent_A)
        cr_B = self.cumulative_reward(agent_B)

        game = f'Game {self.metadata["name"]}:\n'

        if self._timestep > 0:
            game += f'- Last moves: {agent_A} played {move_A.name}, {agent_B} played {move_B.name}\n'
            rewA, rewB = self._reward_matrix[(move_A, move_B)]
            game += f'- Last rewards: {agent_A} got {rewA}, {agent_B} got {rewB}\n'

        if self._nrounds > 0:
            game += f'- Cumulative rewards: {agent_A} has {cr_A}, {agent_B} has {cr_B}\n'

        game += f'- Timestep: {self._timestep} / {self._nrounds}\n'

        print(game)

    def close(self): return




