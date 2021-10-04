import os

from models.params import COMMON_PARAMS as CP
from models.params import STATE_CRITIC_PARAMS as SCP
from models.params import VISUAL_CRITIC_PARAMS as VCP
from models.im_encoder import ImageEncoderNetwork
from models.basic_models.linear import UniformLinearNetwork
from utils.model_utils import load_checkpoint

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
        x.to(self.device)
        features = self.ulinear(x)
        out = self.linear(features)
        return out


class VisualCritic(nn.Module):
    def __init__(self, input_dims, hidden_dim=VCP.HIDDEN_DIM,
                 alpha=VCP.LEARNING_RATE, activation=VCP.ACTIVATION, num_layers=VCP.NUM_LAYERS,
                 load_encoder=VCP.LOAD_ENC, enc_output_dim=VCP.ENC_OUTPUT_DIM,
                 num_encoder_linear_layers=VCP.NUM_ENC_LIN_LAYERS, retrain_encoder=VCP.RETRAIN_ENC,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=VCP.CHKPT_FILE):
        super(VisualCritic, self).__init__()

        self.encoder = ImageEncoderNetwork(input_dims, output_dim=enc_output_dim,
                                           num_linear_layers=num_encoder_linear_layers)
        self.retrain_encoder = retrain_encoder
        if load_encoder:
            load_checkpoint(self.encoder, task_name="rewirl")
        if not self.retrain_encoder:
            self.encoder.parameters()
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.head = StateCritic(enc_output_dim, hidden_dim=hidden_dim, alpha=alpha,
                                activation=activation, num_layers=num_layers)

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x.to(self.device)
        if not self.retrain_encoder:
            with T.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        features = self.encoder(x)
        out = self.head(features)
        return out
