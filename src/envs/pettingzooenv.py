from smac.env import MultiAgentEnv
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit
from pettingzoo.butterfly import knights_archers_zombies_v10
from supersuit import flatten_v0, black_death_v3

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info


class PettingZooWrapper(MultiAgentEnv):
    def __init__(self, time_limit=2500, **kwargs):
        self.episode_limit = time_limit
        
        self._env = knights_archers_zombies_v10.env(
                    # https://www.pettingzoo.ml/butterfly/knights_archers_zombies
                    # making environment harder
                    use_typemasks=True,
                    max_arrows=2,
                    line_death=True,
                    max_cycles=time_limit,
                )
        self._env = flatten_v0(black_death_v3(self._env))
        self._env.reset()
        self.n_agents = self._env.num_agents
        self.agents = self._env.possible_agents
        self._obs = None
        self.longest_action_space = max(self._env.action_spaces.values(), key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_spaces.values(), key=lambda x: x.shape
        )
        # TODO: Add env keys?
        #self._seed = kwargs["seed"]
        #self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        # This returns to episode runner that the episode is finished
        if True in self._env.terminations.values():
            # Check if rewards still have value after termination
            return 0.0, True, {}
        
        actions = [int(a) for a in actions]
        self._obs = [None for _ in actions]
        reward = [None for _ in actions]
        done = [None for _ in actions]
        
        for agent, action, i in zip(self._env.rewards.keys(), actions, range(self.n_agents)):
            self._env.step(action)
            self._obs[i] = self._env.observe(agent)
            reward[i] = self._env.rewards[agent]
            done[i] = self._env.terminations[agent]
            
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        
        return float(sum(reward)), all(done), self._env.rewards

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._envs.observe(agent_id)

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.longest_observation_space.shape[0]

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * self.longest_observation_space.shape[0]

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        agent_id = self._env.agents[agent_id]
        valid = self._env.action_space(agent_id).n * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.longest_action_space.n

    def reset(self):
        """ Returns initial observations and states"""
        self._env.reset()
        observation = []
        for agent in self._env.possible_agents:
            observation.append(self._env.observe(agent))
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in observation
        ]
        
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}