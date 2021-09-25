import os

from params import COMMON_PARAMS as CP
from params import LINEAR_PARAMS as LP
from params import U_LINEAR_PARAMS as ULP

import torch as T
import torch.nn as nn
import torch.optim as optim


class LinearNetwork(nn.Module):
    def __init__(self, linear_layer_params, alpha=LP.LEARNING_RATE,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=LP.CHKPT_FILE):

        super(LinearNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        linear_layers = []
        for i, llp in enumerate(linear_layer_params):
            linear_layers.append(nn.Linear(llp["in_dim"], llp["out_dim"]))
            if i < len(linear_layer_params) - 1:
                linear_layers.append(nn.ReLU())

        self.model = nn.Sequential(
            *linear_layers
        )

    def forward(self, state):
        out = self.model(state)
        return out


class UniformLinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=ULP.OUTPUT_DIM, num_layers=ULP.NUM_LAYERS, alpha=ULP.LEARNING_RATE,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=ULP.CHKPT_FILE):

        super(LinearNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        linear_layer_params = ULP.generate_uniform_linear_layers(input_dim, output_dim, num_layers)
        self.l_net = LinearNetwork(linear_layer_params, alpha=alpha,
                                   chkpt_dir=chkpt_dir, chkpt_filename=chkpt_filename)

    def forward(self, state):
        out = self.l_net(state)
        return out
