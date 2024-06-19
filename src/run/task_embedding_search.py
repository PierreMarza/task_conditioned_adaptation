import argparse
import numpy as np
import os
import pickle
import random
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.gym.gym_env import EnvSpec
from src.models.models import Policy, VisionModel
from src.run.arguments import get_args
from src.utils.utils import fuse_embeddings_flare, load_pretrained_model, set_seed


class SearchDataset(Dataset):
    def __init__(
        self,
        video_trajs: list,
        proprio_trajs: list,
        action_trajs: list,
        nb_samples: int,
    ) -> None:
        """
        Initializing SearchDataset.
        :param video_trajs: a list of sequences of RGB frames (videos from expert demonstrations).
        :param proprio_trajs: a list of sequences of proprioception inputs (proprio from expert demonstrations).
        :param action_trajs: a list of sequences of taken actions (actions from expert demonstrations).
        :param nb_samples: the number of samples to consider during the optimization (here number of gradient steps * batch size).
        """
        self.video_trajs = video_trajs
        self.proprio_trajs = proprio_trajs
        self.action_trajs = action_trajs
        self.nb_samples = nb_samples
        self.highest_action_dim = 30

        config_path = "config/vc1_vitb.yaml"
        _, _, self.transforms = load_pretrained_model(config_path=config_path)

    def __len__(self) -> int:
        """
        Returning the dataset length.
        :return: dataset length.
        """
        return self.nb_samples

    def __getitem__(self, index: int) -> dict:
        """
        Retrieving one data sample
        :param index: index in the dataset.
        :return: a dictionary of RGB images, proprioception input and ground-truth actions.
        """
        # Sampling video
        nb_videos = len(self.video_trajs)
        video_id = random.randrange(nb_videos)
        video_traj = self.video_trajs[video_id]
        proprio_traj = self.proprio_trajs[video_id]
        action_traj = self.action_trajs[video_id]

        # Sampling frames
        video_len = len(video_traj)
        index = random.randrange(2, video_len - 2)
        images = []
        for k in range(3):
            index_ = index - k
            images_ = video_traj[index_]
            images_ = self.transforms(images_)
            images.append(images_)
        images = images[::-1]  # images[-1] should be most recent embedding
        images = torch.cat(images, dim=0)

        # Proprioception
        proprio_input = proprio_traj[index]
        action_dim = proprio_input.shape[0]
        proprio_input = np.concatenate(
            [proprio_input, np.zeros(self.highest_action_dim - action_dim)]
        )

        # Action
        tar = action_traj[index]

        return {
            "images": images,
            "proprio_input": proprio_input,
            "tar": tar,
        }


class SearchTaskEmbeddingPredictor(nn.Module):
    def __init__(self, task_emb_dim: int) -> None:
        """
        Initializing SearchTaskEmbeddingPredictor.
        :param task_emb_dim: the dimensionality of the task embedding.
        """
        super().__init__()
        self.task_embedding = nn.Parameter(torch.zeros(1, task_emb_dim))

    def forward(self, batch_size: int) -> Tensor:
        """
        Forward pass.
        :param batch_size: size of the input batch.
        :return: learnt task embedding for all batch elements.
        """
        return torch.cat([self.task_embedding] * batch_size, dim=0)


def task_embedding_search(
    args: argparse.Namespace,
    writer: SummaryWriter,
    env: str,
    vision_model: VisionModel,
    policy: Policy,
    video_trajs: list,
    proprio_trajs: list,
    action_trajs: list,
    action_dim: int,
) -> SearchTaskEmbeddingPredictor:
    """
    Performing the task embedding search.
    :param args: dictionary of arguments.
    :param writer: tensorboard writer.
    :param env: environment name.
    :param vision_model: vision model extracting visual embeddings.
    :param policy: neural policy predicting the best action.
    :param video_trajs: a list of sequences of RGB frames (videos from expert demonstrations).
    :param proprio_trajs: a list of sequences of proprioception inputs (proprio from expert demonstrations).
    :param action_trajs: a list of sequences of taken actions (actions from expert demonstrations).
    :param action_dim: the dimensionality of the action space of the considered task.
    """
    # Dataset and dataloader
    dataset = SearchDataset(
        video_trajs=video_trajs,
        proprio_trajs=proprio_trajs,
        action_trajs=action_trajs,
        nb_samples=50000,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    print("Dataset length: ", len(dataset))
    print("Dataloader length: ", len(dataloader))

    # Task embedding parameter
    search_task_embedding_predictor = SearchTaskEmbeddingPredictor(
        task_emb_dim=args.task_emb_dim,
    ).to(args.device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(search_task_embedding_predictor.parameters(), lr=1e-1)
    loss_func = torch.nn.MSELoss(reduction="none")

    # Search
    for mb_idx, batch in tqdm(enumerate(dataloader)):
        images = batch["images"].float().to(args.device)
        proprio_input = batch["proprio_input"].float().to(args.device)
        tar = batch["tar"].float().to(args.device)

        # Zeroing gradients
        optimizer.zero_grad()

        # Predicting the task embedding
        task_embedding = search_task_embedding_predictor(batch_size=images.shape[0])

        # Vision model forward pass
        if (
            args.middle_adapter_type == "middle_adapter_cond"
            or args.top_adapter_type == "top_adapter_cond"
        ):
            policy_emb = vision_model(images, task_embedding)
        policy_feat = fuse_embeddings_flare(policy_emb)

        # Adding proprioception input
        policy_feat = torch.cat([policy_feat, proprio_input], dim=-1)

        # Adding task embedding search
        if args.policy_type == "policy_cond":
            policy_feat = torch.cat([policy_feat, task_embedding], dim=-1)

        # Policy forward pass
        policy_pred = policy(policy_feat)
        policy_pred = policy_pred[:, :action_dim]

        # Loss computation and backward pass
        loss = loss_func(policy_pred, tar.detach()).mean()
        loss.backward()
        optimizer.step()

        writer.add_scalar(f"{env}/loss", loss, mb_idx)

    return search_task_embedding_predictor


if __name__ == "__main__":
    # Hyperparameters
    args = get_args()
    args.use_cls = args.use_cls == 1

    # Loading model checkpoints
    ckpts_state_dict = torch.load(args.eval_model_ckpt_path)

    # Setting random seed
    set_seed(args.seed)

    # Extracting data
    base_expert_data = f"data/few_shot/"
    env = args.eval_env
    demo_paths_loc = os.path.join(base_expert_data, f"{env}.pickle")
    try:
        demo_paths = pickle.load(open(demo_paths_loc, "rb"))
    except:
        print("Unable to load the data. Check the data path.")
        print(demo_paths_loc)
        quit()

    video_trajs = []
    proprio_trajs = []
    action_trajs = []
    action_dim = len(demo_paths[0]["actions"][0])
    for traj_id in range(args.eval_te_nb_demos):
        # RGB frames
        video_traj = demo_paths[traj_id]["images"]
        video_trajs.append(video_traj)

        # Proprioception
        if "proprio" in demo_paths[traj_id]["env_infos"].keys():
            proprio_traj = demo_paths[traj_id]["env_infos"]["proprio"]
        elif "gripper_proprio" in demo_paths[traj_id]["env_infos"].keys():
            proprio_traj = demo_paths[traj_id]["env_infos"]["gripper_proprio"]
        else:
            proprio_traj = None
        proprio_trajs.append(proprio_traj)

        # Actions
        action_traj = demo_paths[traj_id]["actions"]
        action_trajs.append(action_traj)

    # Vision Model
    vision_model = VisionModel(
        args.middle_adapter_type,
        args.top_adapter_type,
        args.img_emb_size,
        args.task_emb_dim,
        args.use_cls,
    ).to(args.device)
    vision_model.load_state_dict(ckpts_state_dict["vision_model_state_dict"])
    vision_model.eval()

    # Policy
    policy_state_dict = ckpts_state_dict["policy_state_dict"]
    policy_horizon = None  # not used by the policy
    policy_observation_dim = policy_state_dict["policy.fc_layers.0.weight"].shape[-1]
    policy_action_dim = policy_state_dict["policy.fc_layers.3.weight"].shape[0]
    env_spec = EnvSpec(policy_observation_dim, policy_action_dim, policy_horizon)

    policy = Policy(
        env_spec=env_spec,
        hidden_sizes=args.hidden_sizes,
        nonlinearity=args.nonlinearity,
        dropout=args.dropout,
    ).to(args.device)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    # Tensorboard
    tb_path = (
        f"logs/te_search/tb/{args.seed}_{args.middle_adapter_type}"
        f"_{args.top_adapter_type}_{args.policy_type}_{args.use_cls}"
        f"_{args.expe_name}"
    )
    writer = SummaryWriter(tb_path)
    writer.add_text("Args", str(args), 0)

    ckpts_path = tb_path.replace("/tb/", "/ckpts/")
    if not os.path.exists(ckpts_path):
        os.makedirs(ckpts_path)

    # Task embedding search
    search_task_embedding_predictor = task_embedding_search(
        args,
        writer,
        env,
        vision_model,
        policy,
        video_trajs,
        proprio_trajs,
        action_trajs,
        action_dim,
    )

    # Saving ckpt
    ckpt_dict = {
        "search_task_embedding_predictor_state_dict": search_task_embedding_predictor.state_dict(),
    }
    torch.save(ckpt_dict, os.path.join(ckpts_path, f"ckpt_{env}.pth"))
