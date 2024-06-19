import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple

from src.utils.constants import ENV_TO_ID, ENV_TO_SUITE
from src.utils.utils import load_pretrained_model


def data_loading(base_expert_data: str = "data/train") -> Tuple[list, int]:
    """
    Loading training expert demonstrations.
    :param base_expert_data: string path to the expert data to load.
    :return timesteps_all_envs: list containing required information for each timestep in all expert trajectories for all tasks.
    :return highest_action_dim: highest action space dim across tasks.
    """
    timesteps_all_envs = []
    timestep = 0
    highest_action_dim = 0

    for env in tqdm(
        list(ENV_TO_ID.keys())[:12]
    ):  # Only 12 known tasks (training demonstrations)
        # Paths infos
        demo_paths_loc = os.path.join(
            base_expert_data, f"{ENV_TO_SUITE[env]}-expert-v1.0", f"{env}.pickle"
        )
        try:
            demo_paths = pickle.load(open(demo_paths_loc, "rb"))
        except:
            print("Unable to load the data.")
            print(demo_paths_loc)
            quit()

        env_info_dict = {}
        nb_paths = len(demo_paths)
        nb_paths -= (
            5  # removing last 5 trajs at training time to use them for ablations
        )

        env_info_dict["nb_paths"] = nb_paths
        env_info_dict["beginning_timesteps"] = []
        for i in range(nb_paths):
            path_timestep = 0
            env_info_dict["beginning_timesteps"].append(timestep)
            for j in range(len(demo_paths[i]["actions"])):
                # Information relative to current timestep
                info_dict = {}
                info_dict["actions"] = demo_paths[i]["actions"][j]
                info_dict["images"] = demo_paths[i]["images"][j]
                info_dict["task_id"] = ENV_TO_ID[env]
                info_dict["task_instance_id"] = i
                info_dict["path_timestep"] = path_timestep

                # Proprioception
                if "proprio" in demo_paths[i]["env_infos"].keys():
                    assert "gripper_proprio" not in demo_paths[i]["env_infos"].keys()
                    info_dict["proprio_input"] = demo_paths[i]["env_infos"]["proprio"][
                        j
                    ]
                elif "gripper_proprio" in demo_paths[i]["env_infos"].keys():
                    assert "proprio" not in demo_paths[i]["env_infos"].keys()
                    info_dict["proprio_input"] = demo_paths[i]["env_infos"][
                        "gripper_proprio"
                    ][j]
                else:
                    info_dict["proprio_input"] = None

                # Keeping track of the highest actions dim
                if highest_action_dim < len(demo_paths[i]["actions"][j]):
                    highest_action_dim = len(demo_paths[i]["actions"][j])

                path_timestep += 1
                timesteps_all_envs.append(info_dict)
                timestep += 1
    return timesteps_all_envs, highest_action_dim


# Code adapted from https://github.com/facebookresearch/eai-vc/blob/main/cortexbench/mujoco_vc/visual_imitation/train_loop.py#L290
class FrozenEmbeddingDataset(Dataset):
    def __init__(
        self,
        timesteps_all_envs: list,
        history_window: int = 1,
        highest_action_dim: int = 30,
    ):
        """
        Initializing dataset information.
        :param timesteps_all_envs: list containing required information for each timestep in all expert trajectories for all tasks.
        :param history_window: history window length.
        :param highest_action_dim: highest action space dim across tasks.
        """

        self.timesteps_all_envs = timesteps_all_envs
        self.history_window = history_window
        self.highest_action_dim = highest_action_dim

        # Loading images transforms
        config_path = "config/vc1_vitb.yaml"
        _, _, self.transforms = load_pretrained_model(config_path=config_path)

    def __len__(self) -> int:
        """
        Returning the length of the dataset.
        :return: dataset length.
        """
        return len(self.timesteps_all_envs)

    def __getitem__(self, index: int) -> dict:
        """
        Returning dataset element at input index.
        :param index: integer dataset index.
        :return: dictionary of relevant information.
        """

        # Actions
        # different dims depending on tasks -> padding to reach max actions dim
        actions = self.timesteps_all_envs[index]["actions"]
        action_dims = len(actions)
        actions_mask = np.zeros(
            self.highest_action_dim,
        )
        actions_mask[:action_dims] = 1
        actions = np.concatenate(
            [actions, np.zeros(self.highest_action_dim - action_dims)]
        )

        # Proprioception
        proprio_input = self.timesteps_all_envs[index]["proprio_input"]
        if proprio_input is None:
            proprio_input = np.zeros((self.highest_action_dim,))
        else:
            assert action_dims == len(proprio_input)
            proprio_input = np.concatenate(
                [proprio_input, np.zeros(self.highest_action_dim - action_dims)]
            )

        task_id = self.timesteps_all_envs[index]["task_id"]
        task_instance_id = self.timesteps_all_envs[index]["task_instance_id"]
        path_timestep = self.timesteps_all_envs[index]["path_timestep"]

        # Policy input
        images = []
        for k in range(self.history_window):
            if path_timestep - k >= 0:
                index_ = index - k
                assert self.timesteps_all_envs[index_]["path_timestep"] == (
                    path_timestep - k
                )
            else:
                index_ = index - path_timestep

            assert self.timesteps_all_envs[index_]["task_id"] == task_id
            assert (
                self.timesteps_all_envs[index_]["task_instance_id"] == task_instance_id
            )

            images_ = self.timesteps_all_envs[index_]["images"]
            images_ = self.transforms(images_)
            images.append(images_)
        images = images[::-1]  # images[-1] should be most recent embedding

        # Converting data to arrays
        images = torch.cat(images, dim=0)
        path_timestep = np.array(path_timestep)
        task_id = np.array(task_id)

        return {
            "actions": actions,
            "actions_mask": actions_mask,
            "images": images,
            "proprio_input": proprio_input,
            "task_id": task_id,
        }
