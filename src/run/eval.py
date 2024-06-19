import json
import os
from tabulate import tabulate
import torch
import torch.nn as nn

from src.gym.gym_env import EnvSpec
from src.gym.gym_wrapper import env_constructor
from src.models.models import BatchNormMLP, VisionModel
from src.run.arguments import get_args

from src.utils.constants import ENV_TO_ID, ENV_TO_SUITE
from src.utils.utils import (
    compute_metrics_from_paths,
    fuse_embeddings_flare,
    generate_videos,
    sample_paths,
    set_seed,
)
from src.run.task_embedding_search import SearchTaskEmbeddingPredictor

# Registering env suites
import dmc2gym, gym, mj_envs, mjrl.envs


if __name__ == "__main__":
    # Hyperparameters
    args = get_args()
    task_conditioning = (
        args.middle_adapter_type == "middle_adapter_cond"
        or args.top_adapter_type == "top_adapter_cond"
        or args.policy_type == "policy_cond"
    )
    args.use_cls = args.use_cls == 1

    # Eval-specific hyperparameters
    if args.eval_type == "val_known":
        args.seed = 400
    elif args.eval_type in ["test_known", "test_unknown"]:
        args.seed = 500

    # Picking benchmark
    suite = ENV_TO_SUITE[args.eval_env]
    if suite in ["adroit", "metaworld"]:
        add_proprio = True
        if suite == "adroit":
            proprio_key = "proprio"
            camera = "vil_camera"
        else:
            proprio_key = "gripper_proprio"
            camera = "top_cap2"
    elif suite == "dmc":
        add_proprio = False
        proprio_key = None
        camera = "0"

    # Loading model checkpoints
    ckpts_state_dict = torch.load(args.eval_model_ckpt_path)

    # Creating logging folders
    eval_folder = (
        f"logs/{args.eval_type}/{args.seed}_{args.middle_adapter_type}"
        f"_{args.top_adapter_type}_{args.policy_type}_{args.use_cls}"
        f"_{args.expe_name}"
    )
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    eval_videos_path = os.path.join(eval_folder, "videos", args.eval_env)
    if not os.path.exists(eval_videos_path):
        os.makedirs(eval_videos_path)
    logs_path = os.path.join(eval_folder, "logs", args.eval_env)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # Environment configuration
    env_kwargs = {
        "env_name": args.eval_env,
        "suite": suite,
        "device": args.device,
        "image_width": 256,
        "image_height": 256,
        "camera_name": camera,
        "embedding_name": "vc1_vitb",
        "pixel_based": True,
        "seed": args.seed,
        "history_window": 3,
        "add_proprio": add_proprio,
        "proprio_key": proprio_key,
    }

    # Setting random seed
    set_seed(args.seed)

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

    # Task embedding predictor
    if task_conditioning:
        if args.eval_type != "test_unknown":
            weight_shape = ckpts_state_dict["task_embedding_predictor_state_dict"][
                "weight"
            ].shape
            task_embedding_predictor = nn.Embedding(
                weight_shape[0], weight_shape[1]
            ).to(args.device)
            task_embedding_predictor.load_state_dict(
                ckpts_state_dict["task_embedding_predictor_state_dict"]
            )
            task_embedding_predictor.eval()
            task_id = ENV_TO_ID[args.eval_env]
            with torch.no_grad():
                task_embedding = task_embedding_predictor(
                    torch.Tensor([task_id]).long().to(args.device)
                )
        else:
            task_embedding_predictor = SearchTaskEmbeddingPredictor(
                task_emb_dim=args.task_emb_dim,
            ).to(args.device)
            searched_task_emb_ckpt_state_dict = torch.load(args.eval_te_ckpt_path)
            task_embedding_predictor.load_state_dict(
                searched_task_emb_ckpt_state_dict[
                    "search_task_embedding_predictor_state_dict"
                ]
            )
            task_embedding_predictor.eval()
            with torch.no_grad():
                task_embedding = task_embedding_predictor(batch_size=1)
    else:
        task_embedding = None

    # Policy
    policy_state_dict = {}
    for k in ckpts_state_dict["policy_state_dict"].keys():
        policy_state_dict[k.replace("policy.", "")] = ckpts_state_dict[
            "policy_state_dict"
        ][k]

    policy_horizon = None  # not used by the policy
    policy_observation_dim = policy_state_dict["fc_layers.0.weight"].shape[-1]
    policy_action_dim = policy_state_dict["fc_layers.3.weight"].shape[0]
    env_spec = EnvSpec(policy_observation_dim, policy_action_dim, policy_horizon)

    policy = BatchNormMLP(
        env_spec=env_spec,
        hidden_sizes=args.hidden_sizes,
        seed=args.seed,
        nonlinearity=args.nonlinearity,
        dropout=args.dropout,
    )
    policy.model.load_state_dict(policy_state_dict)
    policy.model.eval()

    # Environment creation
    env_kwargs["vision_model"] = vision_model
    policy_cond = args.policy_type == "policy_cond"
    env_kwargs["policy_cond"] = policy_cond
    env_kwargs["task_embedding"] = task_embedding
    env_kwargs["policy_observation_dim"] = policy_observation_dim
    env_kwargs["highest_action_dim"] = policy_action_dim
    e = env_constructor(
        **env_kwargs,
        fuse_embeddings=fuse_embeddings_flare,
    )

    # Running episodes
    paths = sample_paths(
        num_trajs=args.eval_num_trajs,
        env=e,
        policy=policy,
        eval_mode=True,
        horizon=e.horizon,
        base_seed=args.seed,
    )

    # Computing evaluation metrics
    mean_return, mean_score = compute_metrics_from_paths(
        env=e,
        suite=suite,
        paths=paths,
    )
    eval_log = {}
    eval_log["eval_env"] = args.eval_env
    eval_log["eval_type"] = args.eval_type
    eval_log["expe_name"] = args.expe_name
    eval_log["seed"] = args.seed
    eval_log["eval_num_trajs"] = args.eval_num_trajs
    eval_log["eval_model_ckpt_path"] = args.eval_model_ckpt_path

    if eval_log["eval_type"] == "test_unknown":
        eval_log["eval_te_ckpt_path"] = args.eval_te_ckpt_path

    eval_log["mean_return"] = mean_return
    eval_log["mean_score"] = mean_score

    # Generating sample videos
    generate_videos(paths, eval_videos_path)

    # Logging
    print(tabulate(sorted(eval_log.items())))
    with open(os.path.join(logs_path, "eval_log.json"), "w") as fp:
        json.dump(eval_log, fp)
