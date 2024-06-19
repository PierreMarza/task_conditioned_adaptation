from mjrl.policies.gaussian_mlp import MLP
import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import Variable

from src.gym.gym_env import EnvSpec
from src.utils.utils import load_pretrained_model


# Code adapted from https://github.com/aravindr93/mjrl/blob/83d35df95eb64274c5e93bb32a0a4e2f6576638a/mjrl/utils/fc_network.py#L67
class FCNetworkWithBatchNorm(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple = (64, 64),
        nonlinearity: str = "relu",
        dropout: int = 0,
    ) -> None:
        """
        Initializing FCNetworkWithBatchNorm.
        :param obs_dim: integer dim of each observation.
        :param act_dim: integer dim of the action space.
        :param hidden_sizes: tuple of sizes of the different linear layers (default: (64, 64)).
        :param nonlinearity: activation function -- either "tanh" or "relu" (default: "relu")
        :param dropout: probability to dropout activations -- 0 means no dropout (default: 0)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layer_sizes = (obs_dim,) + hidden_sizes + (act_dim,)

        # Hidden layers
        self.fc_layers = nn.ModuleList(
            [
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
                for i in range(len(self.layer_sizes) - 1)
            ]
        )
        self.nonlinearity = torch.relu if nonlinearity == "relu" else torch.tanh
        self.input_batchnorm = nn.BatchNorm1d(num_features=obs_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass.
        :param src: input tensor with shape (B, P) where,
            - B: batch size
            - P: policy input dim.
        :return: output tensor with shape (B, A) where,
            - B: batch size
            - A: action dim.
        """
        out = self.input_batchnorm(src)
        for i in range(len(self.fc_layers) - 1):
            out = self.fc_layers[i](out)
            out = self.dropout(out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out


# Code adapted from https://github.com/aravindr93/mjrl/blob/83d35df95eb64274c5e93bb32a0a4e2f6576638a/mjrl/policies/gaussian_mlp.py#L148
class BatchNormMLP(MLP):
    def __init__(
        self,
        env_spec: EnvSpec,
        hidden_sizes: tuple = (64, 64),
        min_log_std: int = -3,
        init_log_std: int = 0,
        seed: int = None,
        nonlinearity: str = "relu",
        dropout: float = 0,
        *args: dict,
        **kwargs: dict,
    ) -> None:
        """
        :param env_spec: specifications of the env (see utils/gym_env.py).
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only).
        :param min_log_std: log_std is clamped at this value and can't go below.
        :param init_log_std: initial log standard deviation.
        :param seed: random seed.
        :param nonlinearity: activation function to use.
        :param dropout: dropout probability.
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std

        # Setting seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        self.model = FCNetworkWithBatchNorm(
            self.n, self.m, hidden_sizes, nonlinearity, dropout
        )
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
            param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]
        self.model.eval()

        # Old Policy network
        self.old_model = FCNetworkWithBatchNorm(
            self.n, self.m, hidden_sizes, nonlinearity, dropout
        )
        self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()
        self.old_model.eval()

        # Easy access variables
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)


class Policy(nn.Module):
    def __init__(
        self,
        env_spec: EnvSpec,
        hidden_sizes: tuple = (64, 64),
        nonlinearity: str = "relu",
        dropout: int = 0,
    ) -> None:
        """
        Initializing Policy.
        :param env_spec: EnvSpec object containing information about task/env.
        :param hidden_sizes: tuple of sizes of the different linear layers (default: (64, 64)).
        :param nonlinearity: activation function -- either "tanh" or "relu" (default: "relu")
        :param dropout: probability to dropout activations  -- 0 means no dropout (default: 0)
        """
        super().__init__()
        self.policy = FCNetworkWithBatchNorm(
            env_spec.observation_dim,
            env_spec.action_dim,
            hidden_sizes,
            nonlinearity,
            dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass.
        :param src: input tensor with shape (B, P) where,
            - B: batch size
            - P: policy input dim.
        :return: output tensor with shape (B, A) where,
            - B: batch size
            - A: action dim.
        """
        return self.policy(src)


class VisionModel(nn.Module):
    def __init__(
        self,
        middle_adapter_type: str,
        top_adapter_type: str,
        img_emb_size: int,
        task_emb_dim: int,
        use_cls: bool,
    ) -> None:
        """
        Initializing VisionModel.
        :param middle_adapter_type: type of middle adapter to use -- "no_middle_adapter", "middle_adapter_no_cond", "middle_adapter_cond".
        :param top_adapter_type: type of top adapter to use -- "no_top_adapter", "top_adapter_no_cond", "top_adapter_cond".
        :param img_emb_size: Embedding size of a visual frame embedding.
        :param task_emb_dim: Embedding size of the task embedding.
        :param use_cls: Whether to use the 'CLS' token as the final representation or not.
        """
        super().__init__()

        # Loading pre-trained VC-1 model
        config_path = "config/vc1_vitb.yaml"
        self.model, _, _ = load_pretrained_model(config_path=config_path)
        self.nb_vit_blocks = len(self.model.blocks)

        # Middle adapters
        self.middle_adapter_type = middle_adapter_type
        if middle_adapter_type != "no_middle_adapter":
            self.middle_adapters = []
            in_dim_middle = img_emb_size
            if middle_adapter_type == "middle_adapter_cond":
                in_dim_middle += task_emb_dim
            for _ in range(self.nb_vit_blocks):
                adapter = nn.Sequential(
                    nn.Linear(in_dim_middle, img_emb_size // 2),
                    nn.GELU(),
                    nn.Linear(img_emb_size // 2, img_emb_size),
                )
                self.middle_adapters.append(adapter)
            self.middle_adapters = nn.ModuleList(self.middle_adapters)
        else:
            self.middle_adapters = None

        # Top adapter
        self.top_adapter_type = top_adapter_type
        if top_adapter_type != "no_top_adapter":
            in_dim_top = img_emb_size
            if top_adapter_type == "top_adapter_cond":
                in_dim_top += task_emb_dim
            self.top_adapter = nn.Sequential(
                nn.Linear(in_dim_top, img_emb_size),
                nn.ReLU(),
                nn.Linear(img_emb_size, img_emb_size),
            )
        else:
            self.top_adapter = None

        # Aggregating FC layer
        self.use_cls = use_cls
        if not use_cls:
            self.fc_agr = nn.Sequential(nn.Linear(14 * 14 * 768, 768), nn.ReLU())

    # Code adapted from https://github.com/facebookresearch/eai-vc/blob/main/vc_models/src/vc_models/models/vit/vit.py#L101
    def forward_vit_features(
        self, src: Tensor, task_embedding: Tensor, history_window: int
    ) -> Tensor:
        """
        ViT forward pass.
        :param src: input tensor with shape (B*L, C, H, W) where,
            - B: batch size
            - L: history window length
            - C: number of image channels (C=3)
            - H: image height
            - W: image width.
        :param task_embedding: task embedding tensor with shape (B, F) where,
            - B: bacth size
            - F: task embedding dim.
        :param history_window: history window length.
        :return: output tensor with shape (B*L, N, E) where,
            - B: batch size
            - L: history window length
            - N: number of ViT tokens
            - E: visual embedding dim.
        """
        B = src.shape[0]
        src = self.model.patch_embed(src)

        # Adding pos embed w/o cls token
        src = src + self.model.pos_embed[:, 1:, :]

        # Append cls token
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        cls_token = cls_token.expand(B, -1, -1)
        src = torch.cat((cls_token, src), dim=1)

        if self.middle_adapter_type == "middle_adapter_cond":
            # Processing task_embedding
            task_embedding = task_embedding.unsqueeze(dim=1)
            task_embedding = torch.cat(history_window * [task_embedding], dim=1)
            task_embedding = task_embedding.view(-1, task_embedding.shape[-1])
            task_embedding = task_embedding.unsqueeze(dim=1)
            task_embedding = torch.cat(src.shape[1] * [task_embedding], dim=1)

        # Block and adapter inference
        for i in range(self.nb_vit_blocks):
            block = self.model.blocks[i]
            src = block(src)

            if self.middle_adapters is not None:
                adapter = self.middle_adapters[i]

            if self.middle_adapter_type == "middle_adapter_cond":
                src = src + adapter(torch.cat([src, task_embedding], dim=-1))
            elif self.middle_adapter_type == "middle_adapter_no_cond":
                src = src + adapter(src)

        return self.model.norm(src)

    def forward(
        self,
        src: Tensor,
        task_embedding: Tensor,
    ) -> Tensor:
        """
        Forward pass.
        :param src: input tensor with shape (B, L, C, H, W) where,
            - B: batch size
            - L: history window length
            - C: number of image channels (C=3)
            - H: image height
            - W: image width.
        :param task_embedding: task embedding tensor with shape (B, F) where,
            - B: bacth size
            - F: task embedding dim.
        :return: output tensor with shape (B, L, E) where,
            - B: batch size
            - L: history window length
            - E: visual embedding dim.
        """
        assert src.shape[2] == 3 and src.shape[3] == src.shape[4] == 224  # RGB input
        batch, history, c, h, w = src.shape
        src = src.view(-1, c, h, w)
        src = self.forward_vit_features(src, task_embedding, history)

        if self.use_cls:
            src = src[:, 0]
        else:
            src = src[:, 1:]
            src = self.fc_agr(src.reshape(src.shape[0], -1))

        if self.top_adapter is not None:
            src = src.view(batch, history, src.shape[-1])
            img_emb_dim = src.shape[-1]
            if self.top_adapter_type == "top_adapter_cond":
                _, task_emb_dim = task_embedding.shape
                task_embedding = task_embedding.unsqueeze(1).repeat(1, history, 1)
                src = src.view(-1, img_emb_dim)
                task_embedding = task_embedding.view(-1, task_emb_dim)
                src = torch.cat([src, task_embedding], dim=1)

            output = self.top_adapter(src)
            output = output.view(batch, history, img_emb_dim)
            return output
        else:
            return src.view(batch, history, src.shape[-1])
