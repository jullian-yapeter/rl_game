import os

from params import COMMON_PARAMS as CP
from params import CONV_ENCODER_PARAMS as CEP
from params import D_CONV_ENCODER_PARAMS as DCEP

import torch as T
import torch.nn as nn
import torch.optim as optim


class ConvEncoderNetwork(nn.Module):
    def __init__(self, conv_layer_params, output_dim=CEP.OUTPUT_DIM,
                 alpha=CEP.LEARNING_RATE,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=CEP.CHKPT_FILE):

        super(ConvEncoderNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        conv_layers = []
        for clp in conv_layer_params:
            conv_layers.append(nn.Conv2d(clp["in_channels"], clp["out_channels"], clp.get("kernel_size"),
                                         stride=clp.get("stride"), padding=clp.get("padding")))
            conv_layers.append(nn.ReLU())
            self.last_conv_size = clp["out_channels"]
        self.model = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(self.last_conv_size, output_dim),
        )

    def forward(self, state):
        out = self.model(state)
        return out


class DoublingConvEncoderNetwork(nn.Module):
    def __init__(self, input_dims, output_dim=DCEP.OUTPUT_DIM, alpha=DCEP.LEARNING_RATE,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=DCEP.CHKPT_FILE):

        super(DoublingConvEncoderNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        _, channels, height, _ = input_dims
        conv_layer_params = DCEP.generate_doubling_conv_layers(channels, height)
        self.conv_enc = ConvEncoderNetwork(conv_layer_params, input_dims, output_dim=output_dim, alpha=alpha,
                                           chkpt_dir=chkpt_dir, chkpt_filename=chkpt_filename)

    def forward(self, state):
        out = self.conv_enc(state)
        return out
