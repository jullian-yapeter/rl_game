import os

from models.params import COMMON_PARAMS as CP
from models.params import STATE_ACTOR_PARAMS as SAP
from models.basic_models.linear import UniformLinearNetwork

import torch as T
import torch.nn as nn
import torch.optim as optim


class StateActor(nn.Module):
    def __init__(self, input_dim, output_dim=SAP.NUM_ACTIONS, hidden_dim=SAP.HIDDEN_DIM,
                 alpha=SAP.LEARNING_RATE, inner_activation=SAP.IN_ACTIVATION, out_activation=SAP.OUT_ACTIVATION,
                 num_layers=SAP.NUM_LAYERS, chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=SAP.CHKPT_FILE):
        super(StateActor, self).__init__()
        self.inner_activation = inner_activation
        self.ulinear = UniformLinearNetwork(input_dim, output_dim=hidden_dim, num_layers=num_layers - 1,
                                            alpha=alpha, activation=self.inner_activation)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.out_activation = out_activation

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x.to(self.device)
        features = self.ulinear(x)
        out = self.linear(features)
        return self.out_activation(out)
