import argparse


def get_args():
    """
    Parsing run arguments.
    """

    parser = argparse.ArgumentParser(description="Task-conditioned Adaptation")

    # Training arguments
    parser.add_argument(
        "--expe_name",
        type=str,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--seed", type=int, default=100, help="Random seed (default: 100)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Training learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--ckpt_save_frequency",
        type=int,
        default=5,
        help="Frequency (in epochs) of model checkpoint saving (default: 5)",
    )

    # Multi-task policy
    parser.add_argument(
        "--history_window",
        type=int,
        default=3,
        help="Number of past frames to fuse in policy (default: 3)",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=tuple,
        default=(256, 256, 256),
        help="Policy hidden layer sizes (default: (256, 256, 256))",
    )
    parser.add_argument(
        "--nonlinearity",
        type=str,
        default="relu",
        help="Policy activation function (default: relu)",
    )
    parser.add_argument(
        "--dropout", type=int, default=0, help="Policy dropout (default: 0)"
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        choices=["policy_no_cond", "policy_cond"],
        help="Type of policy to use (conditioned or not on the task embedding)",
    )

    # Visual encoder and adapters
    parser.add_argument(
        "--img_emb_size",
        type=int,
        default=768,
        help="Size of visual embeddings (default: 768)",
    )
    parser.add_argument(
        "--use_cls",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to output the 'CLS' token as input to the policy (1) or "
        "learn a FC layer aggregating patches (0) (default: 0)",
    )
    parser.add_argument(
        "--middle_adapter_type",
        type=str,
        choices=[
            "no_middle_adapter",
            "middle_adapter_no_cond",
            "middle_adapter_cond",
        ],
        help="Type of middle adapter to use",
    )
    parser.add_argument(
        "--top_adapter_type",
        type=str,
        choices=[
            "no_top_adapter",
            "top_adapter_no_cond",
            "top_adapter_cond",
        ],
        help="Type of top adapter to use",
    )
    parser.add_argument(
        "--task_emb_dim",
        type=int,
        default=1024,
        help="Dimension of the task embedding (default: 1024)",
    )
    parser.add_argument(
        "--ntasks", type=int, default=12, help="Number of tasks (default: 12)"
    )

    # Eval-specific hyperparameters
    parser.add_argument(
        "--eval_model_ckpt_path",
        type=str,
        help="Path to the model checkpoints.",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=[
            "val_known",
            "test_known",
            "test_unknown",
        ],
        help="Type of evaluation (val on known tasks, test on known tasks, test on unknown tasks)",
    )
    parser.add_argument(
        "--eval_env",
        type=str,
        help="Name of the environment to evaluate on",
    )
    parser.add_argument(
        "--eval_num_trajs",
        type=int,
        default=100,
        help="Number of evaluation trajectories for a task (default: 100)",
    )
    parser.add_argument(
        "--eval_te_ckpt_path",
        type=str,
        help="Path to the optimized task embedding.",
    )
    parser.add_argument(
        "--eval_te_nb_demos",
        type=int,
        default=5,
        help="Number of expert demonstrations to use to optimize the task embedding (default: 5).",
    )

    args = parser.parse_args()
    return args
