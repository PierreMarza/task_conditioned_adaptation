import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.gym.gym_env import EnvSpec
from src.models.models import (
    Policy,
    VisionModel,
)
from src.run.arguments import get_args
from src.utils.constants import ENV_TO_ID
from src.utils.data import data_loading, FrozenEmbeddingDataset
from src.utils.utils import fuse_embeddings_flare, set_seed


if __name__ == "__main__":
    # Hyperparameters
    args = get_args()
    task_conditioning = (
        args.middle_adapter_type == "middle_adapter_cond"
        or args.top_adapter_type == "top_adapter_cond"
        or args.policy_type == "policy_cond"
    )
    args.use_cls = args.use_cls == 1

    # Setting random seed
    set_seed(args.seed)

    # Tensorboard
    tb_path = (
        f"logs/train/tb/{args.seed}_{args.middle_adapter_type}"
        f"_{args.top_adapter_type}_{args.policy_type}_{args.use_cls}"
        f"_{args.expe_name}"
    )
    writer = SummaryWriter(tb_path)
    writer.add_text("Args", str(args), 0)

    # ckpt saving
    ckpts_path = tb_path.replace("/tb/", "/ckpts/")
    if not os.path.exists(ckpts_path):
        os.makedirs(ckpts_path)

    # Data loading
    (
        timesteps_all_envs,
        highest_action_dim,
    ) = data_loading()

    # Dataset and dataloader
    dataset = FrozenEmbeddingDataset(
        timesteps_all_envs=timesteps_all_envs,
        history_window=args.history_window,
        highest_action_dim=highest_action_dim,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    print("Dataset length: ", len(dataset))
    print("Dataloader length: ", len(dataloader))

    # Policy
    # embedding per history frame + proprioception for highest action dim
    observation_dim = args.history_window * args.img_emb_size + highest_action_dim
    if args.policy_type == "policy_cond":
        observation_dim += args.task_emb_dim  # adding input task embedding dim
    horizon = None  # not used by the policy
    env_spec = EnvSpec(observation_dim, highest_action_dim, horizon)
    policy = Policy(
        env_spec=env_spec,
        hidden_sizes=args.hidden_sizes,
        nonlinearity=args.nonlinearity,
        dropout=args.dropout,
    ).to(args.device)
    policy.train()

    # Vision Model
    vision_model = VisionModel(
        args.middle_adapter_type,
        args.top_adapter_type,
        args.img_emb_size,
        args.task_emb_dim,
        args.use_cls,
    ).to(args.device)
    vision_model.train()

    # Task embedding predictor
    if task_conditioning:
        task_embedding_predictor = nn.Embedding(args.ntasks, args.task_emb_dim).to(
            args.device
        )
        task_embedding_predictor.train()
    else:
        task_embedding_predictor = None

    # Optimizer and loss
    optimized_weights = []
    name_optimized_weights = []
    optimized_weights += list(policy.parameters())
    for name, _ in policy.named_parameters():
        name_optimized_weights.append(name)

    vision_params_to_train = []
    for name, param in vision_model.named_parameters():
        if (
            (
                args.middle_adapter_type != "no_middle_adapter"
                and "middle_adapters" in name
            )
            or (args.top_adapter_type != "no_top_adapter" and "top_adapter" in name)
            or "fc_agr" in name
        ):
            vision_params_to_train.append(param)
            name_optimized_weights.append(name)
    optimized_weights += vision_params_to_train

    if task_embedding_predictor is not None:
        optimized_weights += list(task_embedding_predictor.parameters())
        for name, _ in task_embedding_predictor.named_parameters():
            name_optimized_weights.append(name)
    print("Optimized weights: ", name_optimized_weights)

    optimizer = torch.optim.Adam(optimized_weights, lr=args.lr)
    loss_func = torch.nn.MSELoss(reduction="none")

    # Training
    for epoch in tqdm(range(args.epochs)):
        running_loss = 0.0
        per_task_running_loss = [[] for _ in range(args.ntasks)]
        for mb_idx, batch in tqdm(enumerate(dataloader)):
            # Zeroing gradients
            optimizer.zero_grad()

            # Data
            actions_mask = batch["actions_mask"].bool().to(args.device)
            proprio_input = batch["proprio_input"].float().to(args.device)
            images = batch["images"].float().to(args.device)
            task_id = batch["task_id"].to(args.device)
            tar = batch["actions"].float().to(args.device)

            # Task embedding prediction
            if task_conditioning:
                task_embedding = task_embedding_predictor(task_id)
            else:
                task_embedding = None

            # Vision model forward pass
            emb = vision_model(
                images,
                task_embedding,
            )

            # Fusing embeddings for the last 3 frames
            feat = fuse_embeddings_flare(emb)

            # Concatenating the proprioception input
            feat = torch.cat([feat, proprio_input], dim=-1)

            # Concatening the task embedding to the input to the
            # policy if the latter is conditioned
            if args.policy_type == "policy_cond":
                feat = torch.cat([feat, task_embedding], dim=-1)

            # Policy forward pass
            pred = policy(feat)

            # Computing loss
            loss = loss_func(pred, tar.detach())

            # Computing per-task loss
            for i in range(args.ntasks):
                task_loss = loss[task_id == i]
                task_actions_mask = actions_mask[task_id == i]
                task_loss = task_loss.view(-1)
                task_actions_mask = task_actions_mask.view(-1)
                task_loss = task_loss[task_actions_mask]
                per_task_running_loss[i].append(task_loss)

            # Masking loss
            loss = loss.view(-1)
            actions_mask = actions_mask.view(-1)
            loss = loss[actions_mask]
            loss = loss.mean()
            running_loss += loss.to("cpu").data.numpy().ravel()[0]

            # Backward pass
            loss.backward()
            optimizer.step()

        # Logging average loss for the epoch
        writer.add_scalar("Loss/train", running_loss / (mb_idx + 1), epoch + 1)

        # Logging average per-task loss for the epoch
        for env in tqdm(list(ENV_TO_ID.keys())[:args.ntasks]):
            env_id = ENV_TO_ID[env]
            writer.add_scalar(
                f"Loss/train_policy_{env}",
                torch.cat(per_task_running_loss[env_id]).mean().item(),
                epoch + 1,
            )

        # Saving ckpts
        if (epoch % args.ckpt_save_frequency == 0 and epoch > 0) or (
            epoch == args.epochs - 1
        ):
            ckpt_dict = {
                "seed": args.seed,
                "epoch": epoch,
                "epoch_loss": running_loss / (mb_idx + 1),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            ckpt_dict["policy_state_dict"] = policy.state_dict()
            ckpt_dict["vision_model_state_dict"] = vision_model.state_dict()

            if task_conditioning:
                ckpt_dict["task_embedding_predictor_state_dict"] = (
                    task_embedding_predictor.state_dict()
                )

            torch.save(ckpt_dict, os.path.join(ckpts_path, f"ckpt_{epoch}.pth"))
