import os

from models.basic_models.params import COMMON_PARAMS as CP
from models.basic_models.params import LINEAR_PARAMS as LP
from models.basic_models.params import U_LINEAR_PARAMS as ULP
from models.basic_models.params import D_LINEAR_PARAMS as DLP

import torch as T
import torch.nn as nn
import torch.optim as optim


class LinearNetwork(nn.Module):
    def __init__(self, linear_layer_params, alpha=LP.LEARNING_RATE, activation=LP.ACTIVATION,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=LP.CHKPT_FILE):
        super(LinearNetwork, self).__init__()
        self.activation = activation
        linear_layers = []
        for i, llp in enumerate(linear_layer_params):
            linear_layers.append(nn.Linear(llp["in_dim"], llp["out_dim"]))
            if i < len(linear_layer_params) - 1:
                linear_layers.append(self.activation)

        self.model = nn.Sequential(
            *linear_layers
        )

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x.to(self.device)
        out = self.model(x)
        return out


class UniformLinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=ULP.OUTPUT_DIM, num_layers=ULP.NUM_LAYERS,
                 alpha=ULP.LEARNING_RATE, activation=ULP.ACTIVATION,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=ULP.CHKPT_FILE):
        super(UniformLinearNetwork, self).__init__()
        self.activation = activation
        linear_layer_params = ULP.generate_uniform_linear_layers(input_dim, output_dim, num_layers)
        self.l_net = LinearNetwork(linear_layer_params, alpha=alpha, activation=self.activation)

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x.to(self.device)
        out = self.l_net(x)
        return out


class DoublingLinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=DLP.OUTPUT_DIM,
                 alpha=DLP.LEARNING_RATE, activation=DLP.ACTIVATION,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=DLP.CHKPT_FILE):
        super(DoublingLinearNetwork, self).__init__()
        self.activation = activation
        linear_layer_params = DLP.generate_doubling_linear_layers(input_dim, output_dim)
        self.l_net = LinearNetwork(linear_layer_params, alpha=alpha, activation=self.activation)

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x.to(self.device)
        out = self.l_net(x)
        return out
