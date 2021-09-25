import os

from envs.sprites.sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator
from utils.general_utils import AttrDict
from params import COMMON_PARAMS as CP
from params import REWIRL_AGENT_PARAMS as RAP
from params import REWIRL_TRAINER_PARAMS as RTP
from models.im_encoder import ImageEncoderNetwork
from models.reward_head import RewardHeadNetwork
from utils.model_utils import shuffled_indices, save_checkpoint

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class RewirlAgent(nn.Module):
    def __init__(self, input_dims, reward_names, enc_output_dim=RAP.ENC_OUTPUT_DIM, head_output_dim=RAP.HEAD_OUTPUT_DIM,
                 alpha=RAP.LEARNING_RATE,
                 num_encoder_linear_layers=RAP.NUM_ENC_LIN_LAYERS, num_head_layers=RAP.NUM_HEAD_LAYERS,
                 head_hidden_dim=RAP.HEAD_HIDDEN_DIM,
                 chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=RAP.CHKPT_FILE):

        super(RewirlAgent, self).__init__()
        self.im_enc = ImageEncoderNetwork(input_dims, output_dim=enc_output_dim, alpha=alpha,
                                          num_linear_layers=num_encoder_linear_layers)
        self.rew_heads = nn.ModuleDict()
        for rew_name in reward_names:
            self.rew_heads[rew_name] = RewardHeadNetwork(rew_name, enc_output_dim, hidden_dim=head_hidden_dim,
                                                         output_dim=head_output_dim, alpha=alpha,
                                                         num_layers=num_head_layers)

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, rew_name):
        features = self.im_enc(x.to(self.device))
        out = self.rew_heads[rew_name](features)
        return out


class RewirlTrainer():
    def __init__(self, im_dim=RTP.IM_DIM, max_speed=RTP.MAX_SPEED, obj_size=RTP.OBJ_SIZE,
                 num_distractors=RTP.NUM_DISTRACTORS, reward_types=RTP.REWARD_TYPES,
                 epochs=RTP.EPOCHS, batch_size=RTP.BATCH_SIZE):
        self.task_name = RTP.TASK_NAME
        self.im_dims = (1, 1, im_dim, im_dim)
        self.reward_names = [r.NAME for r in reward_types]
        self.epochs = epochs
        self.spec = AttrDict(
            resolution=im_dim,
            max_seq_len=batch_size,
            max_speed=max_speed,
            obj_size=obj_size,
            shapes_per_traj=2 + num_distractors,      # number of shapes per trajectory
            rewards=reward_types,
        )
        self.gen = DistractorTemplateMovingSpritesGenerator(self.spec)
        self.ra = RewirlAgent(self.im_dims, self.reward_names)
        self.loss_fn = F.mse_loss

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def train(self):
        best_loss = float('inf')
        losses = []
        for i in range(self.epochs):
            traj = self.gen.gen_trajectory()
            images = RewirlTrainer._preprocess_images(traj.images)
            indices = shuffled_indices(images.shape[0])
            images = images[indices]
            train_data = T.tensor(images)
            total_loss = 0
            for reward_name in self.reward_names:
                rewards = traj.rewards.get(reward_name)
                rewards = rewards[indices]
                train_label = T.tensor(rewards)
                pred_label = self.ra(train_data, reward_name)
                loss = self.loss_fn(T.squeeze(pred_label).to(self.device), train_label.to(self.device), reduction="sum")
                total_loss += loss
            losses.append(total_loss.item())
            avg_loss = np.mean(losses[-100:])
            if avg_loss < best_loss:
                self.save_models()
                best_loss = avg_loss
            if i % 100 == 0:
                print(f"iter = {i}, loss = {total_loss}, avg_loss = {avg_loss}")
            self.ra.optimizer.zero_grad()
            total_loss.backward()
            self.ra.optimizer.step()
        plt.plot(losses)
        plt.show()

    def save_models(self):
        print("...saving models...")
        save_checkpoint(self.ra.im_enc, task_name=self.task_name)
        for reward_name in self.reward_names:
            save_checkpoint(self.ra.rew_heads[reward_name], task_name=self.task_name)

    @staticmethod
    # Go from (N, H, W) of [0, 255] to (N, 1, H, W) of [-1, 1]
    def _preprocess_images(images):
        return images[:, None].astype(np.float32) / (255./2) - 1.0


if __name__ == "__main__":
    rt = RewirlTrainer()
    rt.train()
