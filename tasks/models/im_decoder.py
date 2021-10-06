import os

from models.params import COMMON_PARAMS as CP
from models.params import IM_DECODER_PARAMS as IDP
from models.basic_models.linear import DoublingLinearNetwork

import torch as T
import torch.nn as nn
import torch.optim as optim


class ImageDecoderNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=IDP.OUTPUT_DIM, output_im_dims=IDP.OUTPUT_IM_DIMS,
                 alpha=IDP.LEARNING_RATE, in_activation=IDP.IN_ACTIVATION, out_activation=IDP.OUT_ACTIVATION,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=IDP.CHKPT_FILE):
        super(ImageDecoderNetwork, self).__init__()
        self.out_activation = out_activation
        self.output_im_dims = output_im_dims
        self.d_linear = DoublingLinearNetwork(input_dim, output_dim=output_dim, activation=in_activation)

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x.to(self.device)
        features = self.d_linear(x)
        out = self.out_activation(features)
        return out.view(-1, self.output_im_dims[0], self.output_im_dims[1], self.output_im_dims[2])
