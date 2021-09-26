import os

from models.params import COMMON_PARAMS as CP
from models.params import IM_ENCODER_PARAMS as IEP
from models.basic_models.conv_encoder import DoublingConvEncoderNetwork
from models.basic_models.linear import UniformLinearNetwork

import torch as T
import torch.nn as nn
import torch.optim as optim


class ImageEncoderNetwork(nn.Module):
    def __init__(self, input_dims, output_dim=IEP.OUTPUT_DIM,
                 alpha=IEP.LEARNING_RATE, activation=IEP.ACTIVATION,
                 num_linear_layers=IEP.NUM_LINEAR_LAYERS, chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=IEP.CHKPT_FILE):
        super(ImageEncoderNetwork, self).__init__()
        self.activation = activation
        self.conv_enc = DoublingConvEncoderNetwork(input_dims, output_dim=output_dim,
                                                   alpha=alpha, activation=self.activation)
        self.linear = UniformLinearNetwork(output_dim, output_dim=output_dim, num_layers=num_linear_layers,
                                           alpha=alpha, activation=self.activation)

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        features = self.conv_enc(x.to(self.device))
        out = self.linear(features)
        return out
