import os

from models.params import COMMON_PARAMS as CP
from models.params import R_HEAD_PARAMS as RHP
from models.basic_models.linear import UniformLinearNetwork

import torch as T
import torch.nn as nn
import torch.optim as optim


class RewardHeadNetwork(nn.Module):
    def __init__(self, rew_name, input_dim, hidden_dim=RHP.HIDDEN_DIM, output_dim=RHP.OUTPUT_DIM,
                 alpha=RHP.LEARNING_RATE, num_layers=RHP.NUM_LAYERS,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=RHP.CHKPT_FILE):
        super(RewardHeadNetwork, self).__init__()

        self.ulinear = UniformLinearNetwork(input_dim, output_dim=hidden_dim, num_layers=num_layers - 1, alpha=alpha)
        self.linear = nn.Linear(hidden_dim, output_dim)

        self.checkpoint_file = os.path.join(chkpt_dir, f"{chkpt_filename}_{rew_name}")
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        features = self.ulinear(x.to(self.device))
        out = self.linear(features)
        return out
