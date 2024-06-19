import cv2
import gc
import hydra
from mjrl.policies.gaussian_mlp import MLP
import numpy as np
from omegaconf import OmegaConf
import os
from PIL import Image
import torch
from torch import Tensor
from tqdm import tqdm
from typing import Callable, Tuple, Union
from vc_models.models.vit.vit import VisionTransformer

from src.gym.gym_env import GymEnv


# Code adapted from https://github.com/facebookresearch/eai-vc/blob/main/cortexbench/mujoco_vc/visual_imitation/train_loop.py#L28
def set_seed(seed: int) -> None:
    """
    Setting all seeds to make results reproducible.
    :param seed: an integer to your choosing.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Code adapted from https://github.com/facebookresearch/eai-vc/blob/e782c3551e288403f925f6022d37984349aa0102/cortexbench/mujoco_vc/src/mujoco_vc/model_loading.py#L49
def fuse_embeddings_flare(embeddings: Tensor) -> Tensor:
    """
    Fusing frame embeddings from the H previous timesteps (including current timestep).
    :param embeddings: input torch tensor with shape (B, L, E) where,
        - B: batch size
        - L: history window length
        - E: visual embedding dim.
    :return: output fused tensor with shape (B, L*E).
    """
    history_window = embeddings.shape[1]
    delta = [embeddings[:, i + 1] - embeddings[:, i] for i in range(history_window - 1)]
    delta.append(embeddings[:, -1])
    return torch.cat(delta, dim=1)


# Code adapted from https://github.com/facebookresearch/eai-vc/blob/17a27aeae31bf1088daf686d73faab7e43f78f41/cortexbench/mujoco_vc/src/mujoco_vc/model_loading.py#L19
def load_pretrained_model(config_path: str) -> Tuple[VisionTransformer, int, Callable]:
    """
    Loading the pretrained model based on the config.
    :param config_path: string path to the model (and associated information) to load.
    """
    config = OmegaConf.load(config_path)
    model, embedding_dim, transforms, _ = hydra.utils.call(config)
    model = model.eval()

    def final_transforms(transforms):
        return lambda input: transforms(Image.fromarray(input)).unsqueeze(0)

    return model, embedding_dim, final_transforms(transforms)


# Code adapted from https://github.com/aravindr93/mjrl/blob/83d35df95eb64274c5e93bb32a0a4e2f6576638a/mjrl/utils/tensor_utils.py#L63
def stack_tensor_list(tensor_list):
    """
    Stacking a list of tensors.
    :param tensor_list: a list of tensors.
    :return: a stacked tensor.
    """
    return np.array(tensor_list, dtype=object)


# Code adapted from https://github.com/aravindr93/mjrl/blob/83d35df95eb64274c5e93bb32a0a4e2f6576638a/mjrl/utils/tensor_utils.py#L71
def stack_tensor_dict_list(tensor_dict_list):
    """
    Stacking a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}.
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


# Code adapted from https://github.com/aravindr93/mjrl/blob/83d35df95eb64274c5e93bb32a0a4e2f6576638a/mjrl/samplers/core.py#L14
def do_rollout(
    num_trajs: int,
    env: Union[str, GymEnv, Callable],
    policy: MLP,
    eval_mode: bool = False,
    horizon: int = 1e6,
    base_seed: int = None,
    env_kwargs: dict = None,
) -> list:
    """
    Running episodes.
    :param num_trajs: number of trajectories.
    :param env: environment (env class, str with env_name, or factory function).
    :param policy: policy to use for action selection.
    :param eval_mode: use evaluation mode for action computation.
    :param horizon: max horizon length for rollout (<= env.horizon).
    :param base_seed: base seed for rollouts.
    :param env_kwargs: dictionary with parameters, will be passed to env generator.
    :return: list of episodes.
    """
    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError

    if base_seed is not None:
        env.set_seed(base_seed)
        np.random.seed(base_seed)
    else:
        np.random.seed()
    horizon = min(horizon, env.horizon)
    paths = []

    for ep in tqdm(range(num_trajs)):
        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            env.set_seed(seed)
            np.random.seed(seed)

        observations = []
        rgb_observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        o = env.reset()
        done = False
        t = 0

        while t < horizon and done != True:
            rgb_o = o["rgb_obs"]
            o = o["obs"]
            a, agent_info = policy.get_action(o)
            if eval_mode:
                a = agent_info["evaluation"]
            env_info_base = env.get_env_infos()

            # Selecting action dims required for the task
            a = a[: env.spec.action_dim]

            next_o, next_rgb_o, r, done, env_info_step = env.step(a)
            # below is important to ensure correct env_infos for the timestep
            env_info = env_info_step if env_info_base == {} else env_info_base
            observations.append(o)
            rgb_observations.append(rgb_o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = {"obs": next_o, "rgb_obs": next_rgb_o}
            t += 1

        path = dict(
            observations=np.array(observations),
            rgb_observations=np.array(rgb_observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list(env_infos),
            terminated=done,
        )
        paths.append(path)

    del env
    gc.collect()
    return paths


# Code adapted from https://github.com/aravindr93/mjrl/blob/83d35df95eb64274c5e93bb32a0a4e2f6576638a/mjrl/samplers/core.py#L101
def sample_paths(
    num_trajs: int,
    env: Union[str, GymEnv, Callable],
    policy: MLP,
    eval_mode: bool = False,
    horizon: int = 1e6,
    base_seed: int = None,
    env_kwargs: dict = None,
) -> list:
    """
    Running episodes.
    :param num_trajs: number of trajectories.
    :param env: environment (env class, str with env_name, or factory function).
    :param policy: policy to use for action selection.
    :param eval_mode: use evaluation mode for action computation.
    :param horizon: max horizon length for rollout (<= env.horizon).
    :param base_seed: base seed for rollouts.
    :param env_kwargs: dictionary with parameters, will be passed to env generator.
    :return: list of episodes.
    """
    input_dict = dict(
        num_trajs=num_trajs,
        env=env,
        policy=policy,
        eval_mode=eval_mode,
        horizon=horizon,
        base_seed=base_seed,
        env_kwargs=env_kwargs,
    )
    return do_rollout(**input_dict)


# Code adapted from https://github.com/facebookresearch/eai-vc/blob/main/cortexbench/mujoco_vc/visual_imitation/train_loop.py#L265
def compute_metrics_from_paths(
    env: GymEnv,
    suite: str,
    paths: list,
) -> Tuple[float, float]:
    """
    Computing mean metrics from episodes data.
    :param env: environment (env class, str with env_name, or factory function).
    :param suite: benchmark the task belongs to.
    :param paths: episodes data.
    :return: mean return and mean score (depending on the benchmark -- DMC: normalised return / Adroit, Metaworld: mean success).
    """
    mean_return = np.mean([np.sum(p["rewards"]) for p in paths]).item()
    if suite == "dmc":
        # we evaluate dmc based on returns, not success
        mean_score = (
            mean_return / 10
        )  # dividing return by 1000 (max return), and multiplying by 100 to have a score between 0 and 100.
    if suite == "adroit":
        mean_score = env.env.unwrapped.evaluate_success(paths)
    if suite == "metaworld":
        sc = []
        for path in paths:
            sc.append(path["env_infos"]["success"][-1])
        mean_score = np.mean(sc).item() * 100
    return mean_return, mean_score


def generate_videos(paths: list, videos_path: str) -> None:
    """
    Generating episode videos.
    :param paths: episodes data.
    :param videos_path: path to save videos.
    """
    for i, path in enumerate(paths):
        rgbs = path["rgb_observations"]
        rgbs = rgbs[..., ::-1]
        traj_return = np.sum(path["rewards"])

        height, width, _ = rgbs[0].shape
        video_path = os.path.join(
            videos_path, f"episode_{i}_traj_return_{traj_return}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, 20, (width, height))

        for i in range(rgbs.shape[0]):
            video.write(rgbs[i])

        cv2.destroyAllWindows()
        video.release()
