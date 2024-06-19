# Code adapted from https://github.com/facebookresearch/eai-vc/blob/main/cortexbench/mujoco_vc/src/mujoco_vc/gym_wrapper.py

import gym
from gym.spaces.box import Box
import numpy as np
import torch
from torch import Tensor
from typing import Union

from src.gym.gym_env import GymEnv
from src.models.models import VisionModel
from src.utils.constants import ENV_TO_SUITE
from src.utils.utils import load_pretrained_model


def env_constructor(
    env_name: str,
    pixel_based: bool = True,
    device: str = "cuda",
    image_width: int = 256,
    image_height: int = 256,
    camera_name: str = None,
    embedding_name: str = "resnet50",
    history_window: int = 1,
    fuse_embeddings: callable = None,
    seed: int = 123,
    add_proprio: bool = False,
    vision_model: VisionModel = None,
    policy_cond: bool = None,
    task_embedding: Tensor = None,
    policy_observation_dim: int = None,
    highest_action_dim: int = 30,
    *args: dict,
    **kwargs: dict,
) -> GymEnv:
    """
    Creating an environment.
    :param env_name: the name of the environment.
    :param pixel_based: whether to render pixel images.
    :param device: where to allocate the model.
    :param image_width: width of the rendered images.
    :param image_height: height of the rendered images.
    :param camera_name: camera configuration (specified by its name).
    :param embedding_name: name of the embedding to use (name of config).
    :param history_window: timesteps of observation embedding to incorporate into observation (state).
    :param fuse_embeddings: function for fusing the embeddings into a state.
    :param seed: Random seed to use.
    :param add_proprio: whether proprioception should be appended to observation.
    :param vision_model: vision model to use to encode observations.
    :param policy_cond: whether the policy is conditioned in the task embedding.
    :param task_embedding: predicted task embedding to use for this specific task.
    :param policy_observation_dim: dimension of the observation vector fed to the policy.
    :param highest_action_dim: largest action space size among all tasks.
    """
    # construct basic gym environment
    assert env_name in ENV_TO_SUITE.keys()
    suite = ENV_TO_SUITE[env_name]
    if suite == "metaworld":
        # Meta world natively misses many specs. We will explicitly add them here.
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        from collections import namedtuple

        e = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
        e._freeze_rand_vec = False
        e.spec = namedtuple("spec", ["id", "max_episode_steps"])
        e.spec.id = env_name
        e.spec.max_episode_steps = 500
    else:
        e = gym.make(env_name)
    # seed the environment for reproducibility
    e.seed(seed)

    # get correct camera name
    camera_name = (
        None if (camera_name == "None" or camera_name == "default") else camera_name
    )
    # Use appropriate observation wrapper
    if pixel_based:
        e = MuJoCoPixelObsWrapper(
            env=e,
            width=image_width,
            height=image_height,
            camera_name=camera_name,
            device_id=0,
        )

        e = FrozenEmbeddingWrapper(
            env=e,
            embedding_name=embedding_name,
            suite=suite,
            history_window=history_window,
            fuse_embeddings=fuse_embeddings,
            device=device,
            add_proprio=add_proprio,
            vision_model=vision_model,
            policy_cond=policy_cond,
            task_embedding=task_embedding,
            highest_action_dim=highest_action_dim,
        )
        e = GymEnv(e, policy_observation_dim=policy_observation_dim)
    else:
        e = GymEnv(e, policy_observation_dim=policy_observation_dim)

    # Output wrapped env
    e.set_seed(seed)
    return e


class MuJoCoPixelObsWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        width: int,
        height: int,
        camera_name: str,
        device_id: int = -1,
        depth: bool = False,
        *args: dict,
        **kwargs: dict,
    ) -> None:
        """
        Initializing MuJoCoPixelObsWrapper.
        :param env: environment.
        :param width: width of the rendered images.
        :param height: height of the rendered images.
        :param camera_name: camera configuration (specified by its name).
        :param device_id: id of the device to use to render images.
        :param depth: whether to render depth frames.
        """
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0.0, high=255.0, shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id

    def get_image(self) -> np.array:
        """
        Rendering the current image frame.
        """
        if self.spec.id.startswith("dmc"):
            # dmc backend
            # dmc expects camera_id as an integer and not name
            if self.camera_name == None or self.camera_name == "None":
                self.camera_name = 0
            img = self.env.unwrapped.render(
                mode="rgb_array",
                width=self.width,
                height=self.height,
                camera_id=int(self.camera_name),
            )
        else:
            # mujoco-py backend
            img = self.sim.render(
                width=self.width,
                height=self.height,
                depth=self.depth,
                camera_name=self.camera_name,
                device_id=self.device_id,
            )
            img = img[::-1, :, :]
        return img

    def observation(self, observation: np.array) -> np.array:
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        # Output format is (H, W, 3)
        return self.get_image()


class FrozenEmbeddingWrapper(gym.ObservationWrapper):
    """
    This wrapper places a frozen vision model over the image observation.
    """

    def __init__(
        self,
        env,
        embedding_name: str,
        suite: str,
        history_window: int = 1,
        fuse_embeddings: callable = None,
        obs_dim: int = None,
        device: str = "cuda",
        add_proprio: bool = False,
        vision_model: VisionModel = None,
        policy_cond: bool = None,
        task_embedding: Tensor = None,
        highest_action_dim: int = 30,
        *args: dict,
        **kwargs: dict,
    ) -> None:
        """
        Initializing FrozenEmbeddingWrapper.
        :param env: environment.
        :param suite: category of environment ["dmc", "adroit", "metaworld"].
        :param embedding_name: name of the embedding to use (name of config).
        :param history_window: timesteps of observation embedding to incorporate into observation (state).
        :param fuse_embeddings: function for fusing the embeddings into a state.
        :param obs_dim: dimensionality of the observation space. Inferred if not specified.
        :param device: where to allocate the model.
        :param seed: Random seed to use.
        :param add_proprio: whether proprioception should be appended to observation.
        :param vision_model: vision model to use to encode observations.
        :param policy_cond: whether the policy is conditioned in the task embedding.
        :param task_embedding: predicted task embedding to use for this specific task.
        :param policy_observation_dim: dimension of the observation vector fed to the policy.
        :param highest_action_dim: largest action space size among all tasks.
        """
        gym.ObservationWrapper.__init__(self, env)

        self.embedding_buffer = (
            []
        )  # buffer to store raw embeddings of the image observation
        self.obs_buffer = []  # temp variable, delete this line later
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        if device == "cuda" and torch.cuda.is_available():
            print("Using CUDA.")
            device = torch.device("cuda")
        else:
            print("Not using CUDA.")
            device = torch.device("cpu")
        self.device = device

        # get the embedding dim and transforms
        config_path = f"config/{embedding_name}.yaml"
        embedding, embedding_dim, transforms = load_pretrained_model(
            config_path=config_path
        )
        embedding.to(device)
        self.embedding, self.embedding_dim, self.transforms = (
            embedding,
            embedding_dim,
            transforms,
        )

        # related to task-conditioned adapters
        self.vision_model = vision_model
        self.policy_cond = policy_cond
        self.task_embedding = task_embedding
        self.highest_action_dim = highest_action_dim
        self.suite = suite

        # proprioception
        if add_proprio:
            self.get_proprio = lambda: get_proprioception(self.unwrapped, suite)
            proprio = self.get_proprio()
            self.proprio_dim = 0 if proprio is None else proprio.shape[0]
        else:
            self.proprio_dim = 0
            self.get_proprio = None

        # final observation space
        obs_dim = (
            obs_dim
            if obs_dim != None
            else int(self.history_window * self.embedding_dim + self.proprio_dim)
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    def observation(self, observation: np.array) -> dict:
        """
        Computing the observation vector fed to the policy from the RGB observation with the vision model.
        :param observation: RGB observation (image frame).
        :return: embedding from the vision model and original RGB observation.
        """
        # observation shape : (H, W, 3)
        rgb_obs = observation.copy()

        with torch.no_grad():
            # observation shape : (H, W, 3)
            inp = self.transforms(
                observation
            )  # numpy to PIL to torch.Tensor. Final dimension: (1, 3, H, W)
            inp = inp.to(self.device).unsqueeze(1)
            with torch.no_grad():
                emb = self.vision_model(inp, self.task_embedding)
                emb = emb.squeeze(dim=1).cpu().numpy()

        # update observation buffer
        if len(self.embedding_buffer) < self.history_window:
            # initialization
            self.embedding_buffer = [emb.copy()] * self.history_window
        else:
            # fixed size buffer, replace oldest entry
            for i in range(self.history_window - 1):
                self.embedding_buffer[i] = self.embedding_buffer[i + 1].copy()
            self.embedding_buffer[-1] = emb.copy()

        # fuse embeddings to obtain observation
        if self.fuse_embeddings != None:
            embedding_buffer_tensor = torch.Tensor(
                np.concatenate(self.embedding_buffer, axis=0)
            ).unsqueeze(0)
            obs = self.fuse_embeddings(embedding_buffer_tensor).squeeze(0).numpy()

        if self.proprio_dim > 0:
            proprio = self.get_proprio()
            action_dims = len(proprio)
            proprio = np.concatenate(
                [proprio, np.zeros(self.highest_action_dim - action_dims)]
            )
        else:
            proprio = np.zeros((self.highest_action_dim,))
        obs = np.concatenate([obs, proprio])

        if self.policy_cond:
            obs = np.concatenate([obs, self.task_embedding.cpu().numpy()[0]])

        return {"obs": obs, "rgb_obs": rgb_obs}

    def get_obs(self) -> dict:
        return self.observation(self.env.observation(None))

    def get_image(self) -> np.array:
        return self.env.get_image()

    def reset(self) -> dict:
        """
        Resetting the environment.
        """
        self.embedding_buffer = []  # reset to empty buffer
        return super().reset()


def get_proprioception(env: gym.Env, suite: str) -> Union[np.ndarray, None]:
    """
    Retrieving the proprioception information depending on the benchmark.
    :param env: environment.
    :param suite: category of environment ["dmc", "adroit", "metaworld"].
    :return: proprioception information (None if no proprioception is computed).
    """
    assert isinstance(env, gym.Env)
    if suite == "metaworld":
        return env.unwrapped._get_obs()[:4]
    elif suite == "adroit":
        # In adroit, in-hand tasks like pen lock the base of the hand
        # while other tasks like relocate allow for movement of hand base
        # as if attached to an arm
        if env.unwrapped.spec.id == "pen-v0":
            return env.unwrapped.get_obs()[:24]
        elif env.unwrapped.spec.id == "relocate-v0":
            return env.unwrapped.get_obs()[:30]
        if env.unwrapped.spec.id == "door-v0":
            return env.unwrapped.get_obs()[:28]
        elif env.unwrapped.spec.id == "hammer-v0":
            return env.unwrapped.get_obs()[:26]
        else:
            print("Unsupported environment. Proprioception is defaulting to None.")
            return None
    elif suite == "dmc":
        # no proprioception used for dm-control
        return None
    else:
        print("Unsupported environment. Proprioception is defaulting to None.")
        return None
