import os

from models.params import COMMON_PARAMS as CP
from models.params import STATE_CRITIC_PARAMS as SCP
from models.basic_models.linear import UniformLinearNetwork

import torch as T
import torch.nn as nn
import torch.optim as optim


class StateCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=SCP.HIDDEN_DIM,
                 alpha=SCP.LEARNING_RATE, activation=SCP.ACTIVATION, num_layers=SCP.NUM_LAYERS,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=SCP.CHKPT_FILE):
        super(StateCritic, self).__init__()
        self.activation = activation
        self.ulinear = UniformLinearNetwork(input_dim, output_dim=hidden_dim, num_layers=num_layers - 1,
                                            alpha=alpha, activation=activation)
        self.linear = nn.Linear(hidden_dim, 1)

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        features = self.ulinear(x.to(self.device))
        out = self.linear(features)
        return out
