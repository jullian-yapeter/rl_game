import os

from models.params import COMMON_PARAMS as CP
from models.params import STATE_ACTOR_PARAMS as SAP
from models.params import VISUAL_ACTOR_PARAMS as VAP
from models.im_encoder import ImageEncoderNetwork
from models.basic_models.linear import UniformLinearNetwork
from utils.model_utils import load_checkpoint

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


class VisualActor(nn.Module):
    def __init__(self, input_dims, output_dim=VAP.NUM_ACTIONS, hidden_dim=VAP.HIDDEN_DIM,
                 alpha=VAP.LEARNING_RATE, inner_activation=VAP.IN_ACTIVATION, out_activation=VAP.OUT_ACTIVATION,
                 num_layers=VAP.NUM_LAYERS, load_encoder=VAP.LOAD_ENC, enc_output_dim=VAP.ENC_OUTPUT_DIM,
                 num_encoder_linear_layers=VAP.NUM_ENC_LIN_LAYERS, retrain_encoder=VAP.RETRAIN_ENC,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=VAP.CHKPT_FILE):
        super(VisualActor, self).__init__()

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

        self.head = StateActor(enc_output_dim, output_dim=output_dim, hidden_dim=hidden_dim, alpha=alpha,
                               inner_activation=inner_activation, out_activation=out_activation,
                               num_layers=num_layers)

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
        out = self.head(features)
        return out
