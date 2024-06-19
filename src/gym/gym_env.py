# Code adapted from https://github.com/aravindr93/mjrl/blob/83d35df95eb64274c5e93bb32a0a4e2f6576638a/mjrl/utils/gym_env.py

"""
Wrapper around a gym env that provides convenience functions
"""

import gym
import numpy as np
from typing import Callable, Tuple, Union


class EnvSpec(object):
    def __init__(self, obs_dim: int, act_dim: int, horizon: int) -> None:
        """
        Initializing EnvSpec.
        :param obs_dim: dimension of the observation space.
        :param act_dim: dimension of the action space.
        :param horizon: maximum episode length.
        """
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon


class GymEnv(object):
    def __init__(
        self,
        env: Union[str, gym.Env, Callable],
        policy_observation_dim: int,
        env_kwargs: dict = None,
        obs_mask: np.array = None,
        act_repeat: int = 1,
        *args: dict,
        **kwargs: dict,
    ) -> None:
        """
        Initializing GymEnv.
        :param env: environment to simulate.
        :param policy_observation_dim: dimension of the observation vector fed to the policy.
        :param env_kwargs: additional arguments.
        :param obs_mask: observation mask.
        :param act_repeat: Number of times to repeat the same selected action.
        """
        # getting the correct environment behavior
        if type(env) == str:
            env = gym.make(env)
        elif isinstance(env, gym.Env):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError

        self.env = env
        self.env_id = env.unwrapped.spec.id
        self.act_repeat = act_repeat

        # Horizon
        try:
            self._horizon = env.spec.max_episode_steps
        except AttributeError:
            self._horizon = env.spec._horizon

        assert self._horizon % act_repeat == 0
        self._horizon = self._horizon // self.act_repeat

        # Action dim
        try:
            self._action_dim = self.env.action_space.shape[0]
        except AttributeError:
            self._action_dim = self.env.unwrapped.action_dim

        # Observation dim
        self._observation_dim = policy_observation_dim

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon)

        # obs mask
        self.obs_mask = np.ones(self._observation_dim) if obs_mask is None else obs_mask

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def observation_space(self) -> gym.spaces.box.Box:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.spaces.box.Box:
        return self.env.action_space

    @property
    def horizon(self) -> int:
        return self._horizon

    def reset(self, seed: int = None) -> dict:
        """
        Resetting the environment.
        :param seed: Random seed.
        """
        try:
            self.env._elapsed_steps = 0
            return self.env.unwrapped.reset_model(seed=seed)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset()

    def reset_model(self, seed: int = None) -> dict:
        # overloading for legacy code
        return self.reset(seed)

    def step(
        self, action: np.array
    ) -> Tuple[np.array, np.array, np.array, np.float64, bool, dict]:
        """
        Updating the environment after applying the selected action.
        :param action: Selected action to execute.
        """
        action = action.clip(self.action_space.low, self.action_space.high)
        if self.act_repeat == 1:
            obs, cum_reward, done, info = self.env.step(action)
        else:
            cum_reward = 0.0
            for _ in range(self.act_repeat):
                obs, reward, done, info = self.env.step(action)
                cum_reward += reward
                if done:
                    break

        # Getting observation vector (feature vector from visual encoder)
        # and input rgb image (that was fed to visual encoder)
        o, rgb_o = obs["obs"], obs["rgb_obs"]

        return self.obs_mask * o, rgb_o, cum_reward, done, info

    def render(self) -> None:
        try:
            self.env.unwrapped.mujoco_render_frames = True
            self.env.unwrapped.mj_render()
        except:
            self.env.render()

    def set_seed(self, seed: int = 123):
        try:
            self.env.seed(seed)
        except AttributeError:
            self.env._seed(seed)

    def get_obs(self) -> np.array:
        try:
            return self.obs_mask * self.env.get_obs()
        except:
            return self.obs_mask * self.env._get_obs()

    def get_env_infos(self) -> dict:
        try:
            return self.env.unwrapped.get_env_infos()
        except:
            return {}
