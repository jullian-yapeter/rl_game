import os

from envs.sprites.sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator
from envs.sprites.sprites_datagen.rewards import ZeroReward
from utils.general_utils import AttrDict
from params import COMMON_PARAMS as CP
from params import REWIRL_AGENT_PARAMS as RAP
from params import REWIRL_AUTOENCODER_PARAMS as RAEP
from params import REWIRL_TRAINER_PARAMS as RTP
from params import REWIRL_TESTER_PARAMS as RTEP
from params import REWIRL_DECODER_TRAINER_PARAMS as RDTP
from params import REWIRL_DECODER_TESTER_PARAMS as RDTEP
from models.im_encoder import ImageEncoderNetwork
from models.im_decoder import ImageDecoderNetwork
from models.reward_head import RewardHeadNetwork
from utils.model_utils import shuffled_indices, save_checkpoint, load_checkpoint

import cv2
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


class RewirlAutoencoder(nn.Module):
    def __init__(self, im_dim, enc_output_dim=RAEP.ENC_OUTPUT_DIM, retrain_encoder=RAEP.RETRAIN_ENC,
                 load_encoder=RAEP.LOAD_ENC, num_encoder_linear_layers=RAEP.NUM_ENC_LIN_LAYERS,
                 alpha=RAEP.LEARNING_RATE, chkpt_dir=CP.CHECKPOINT_DIR, chkpt_filename=RAEP.CHKPT_FILE):

        super(RewirlAutoencoder, self).__init__()
        self.encoder = ImageEncoderNetwork((1, 1, im_dim, im_dim), output_dim=enc_output_dim,
                                           num_linear_layers=num_encoder_linear_layers)
        self.decoder = ImageDecoderNetwork(enc_output_dim, output_dim=im_dim * im_dim,
                                           output_im_dims=(1, im_dim, im_dim))

        self.retrain_encoder = retrain_encoder
        if load_encoder:
            load_checkpoint(self.encoder, task_name="rewirl")
        if not self.retrain_encoder:
            self.encoder.parameters()
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_filename)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        if not self.retrain_encoder:
            with T.no_grad():
                features = self.encoder(x.to(self.device))
        else:
            features = self.encoder(x.to(self.device))
        out = self.decoder(features)
        return out


class RewirlTrainer():
    def __init__(self, im_dim=RTP.IM_DIM, max_speed=RTP.MAX_SPEED, obj_size=RTP.OBJ_SIZE,
                 num_distractors=RTP.NUM_DISTRACTORS, reward_types=RTP.REWARD_TYPES,
                 epochs=RTP.EPOCHS, batch_size=RTP.BATCH_SIZE, window=RTP.WINDOW, figure_file=RTP.FIG_FILE):
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
        self.figure_file = figure_file
        self.window = window

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
            avg_loss = np.mean(losses[-self.window:])
            if avg_loss < best_loss:
                self.save_models()
                best_loss = avg_loss
            if i % self.window == 0:
                print(f"iter = {i}, loss = {total_loss}, avg_loss = {avg_loss}")
                plt.plot(losses)
                plt.title("Visual representation losses")
                plt.savefig(self.figure_file)
            self.ra.optimizer.zero_grad()
            total_loss.backward()
            self.ra.optimizer.step()

    def save_models(self):
        print("...saving models...")
        save_checkpoint(self.ra)
        save_checkpoint(self.ra.im_enc, task_name=self.task_name)
        for reward_name in self.reward_names:
            save_checkpoint(self.ra.rew_heads[reward_name], task_name=self.task_name)

    @staticmethod
    # Go from (N, H, W) of [0, 255] to (N, 1, H, W) of [-1, 1]
    def _preprocess_images(images):
        return images[:, None].astype(np.float32) / (255./2) - 1.0


class RewirlTester():
    def __init__(self, im_dim=RTEP.IM_DIM, seq_len=RTEP.SEQ_LEN, max_speed=RTEP.MAX_SPEED, obj_size=RTEP.OBJ_SIZE,
                 num_distractors=RTEP.NUM_DISTRACTORS, reward_types=RTEP.REWARD_TYPES,
                 num_trials=RTEP.TRIALS, figure_file=RTEP.FIG_FILE):
        self.num_trials = num_trials
        self.im_dims = (1, 1, im_dim, im_dim)
        self.reward_names = [r.NAME for r in reward_types]
        self.spec = AttrDict(
            resolution=im_dim,
            max_seq_len=seq_len,
            max_speed=max_speed,
            obj_size=obj_size,
            shapes_per_traj=2 + num_distractors,      # number of shapes per trajectory
            rewards=reward_types,
        )
        self.gen = DistractorTemplateMovingSpritesGenerator(self.spec)
        self.ra = RewirlAgent(self.im_dims, self.reward_names)
        load_checkpoint(self.ra)
        self.ra.eval()
        self.loss_fn = F.mse_loss

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.figure_file = figure_file

    def test(self):
        losses = []
        for _ in range(self.num_trials):
            traj = self.gen.gen_trajectory()
            images = RewirlTrainer._preprocess_images(traj.images)
            test_data = T.tensor(images)
            total_loss = 0
            for reward_name in self.reward_names:
                rewards = traj.rewards.get(reward_name)
                test_label = T.tensor(rewards)
                pred_label = self.ra(test_data, reward_name)
                # print(f"test: {test_label}")
                # print(f"pred: {T.squeeze(pred_label)}")
                loss = self.loss_fn(T.squeeze(pred_label).to(self.device), test_label.to(self.device), reduction="sum")
                total_loss += loss
            losses.append(total_loss.item())
        print(f"average loss:{np.mean(losses)}")
        plt.plot(losses)
        plt.title("Visual representation losses")
        plt.savefig(self.figure_file)

    @staticmethod
    # Go from (N, H, W) of [0, 255] to (N, 1, H, W) of [-1, 1]
    def _preprocess_images(images):
        return images[:, None].astype(np.float32) / (255./2) - 1.0


class RewirlDecoderTrainer():
    def __init__(self, task_name=RDTP.TASK_NAME,
                 im_dim=RDTP.IM_DIM, seq_len=RDTP.SEQ_LEN, max_speed=RDTP.MAX_SPEED, obj_size=RDTP.OBJ_SIZE,
                 num_distractors=RDTP.NUM_DISTRACTORS, window=RDTP.WINDOW,
                 enc_output_dim=RDTP.ENC_OUTPUT_DIM, num_encoder_linear_layers=RDTP.NUM_ENC_LIN_LAYERS,
                 load_encoder=RDTP.LOAD_ENC, retrain_encoder=RDTP.RETRAIN_ENC,
                 num_trials=RDTP.TRIALS, figure_file=RDTP.FIG_FILE):
        self.task_name = task_name
        self.window = window
        self.num_trials = num_trials
        self.im_dims = (1, 1, im_dim, im_dim)
        self.spec = AttrDict(
            resolution=im_dim,
            max_seq_len=seq_len,
            max_speed=max_speed,
            obj_size=obj_size,
            shapes_per_traj=2 + num_distractors,      # number of shapes per trajectory
            rewards=[ZeroReward],
        )
        self.gen = DistractorTemplateMovingSpritesGenerator(self.spec)

        self.retrain_encoder = retrain_encoder
        self.autoencoder = RewirlAutoencoder(im_dim, enc_output_dim=enc_output_dim,
                                             retrain_encoder=self.retrain_encoder, load_encoder=load_encoder,
                                             num_encoder_linear_layers=num_encoder_linear_layers)

        self.loss_fn = nn.BCELoss(reduction="sum")
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.figure_file = figure_file

    def train(self):
        best_loss = float('inf')
        losses = []
        for i in range(self.num_trials):
            traj = self.gen.gen_trajectory()
            test_label = T.tensor(traj.images / 255., dtype=T.float)
            preproc_images = RewirlTrainer._preprocess_images(traj.images)
            test_data = T.tensor(preproc_images, dtype=T.float)
            pred_label = self.autoencoder(test_data)
            # print(test_label)
            # print(pred_label)
            loss = self.loss_fn(T.squeeze(pred_label).to(self.device), test_label.to(self.device))
            losses.append(loss.item())
            avg_loss = np.mean(losses[-self.window:])
            if avg_loss < best_loss:
                self.save_models()
                best_loss = avg_loss
            if i % self.window == 0:
                print(f"iter = {i}, loss = {loss.item()}, avg_loss = {avg_loss}")
                plt.plot(losses)
                plt.title("RewIRL Decoder losses")
                plt.savefig(self.figure_file)
            self.autoencoder.optimizer.zero_grad()
            loss.backward()
            self.autoencoder.optimizer.step()
        print(f"average loss:{np.mean(losses)}")

    def save_models(self):
        print("...saving models...")
        save_checkpoint(self.autoencoder)
        if self.retrain_encoder:
            save_checkpoint(self.autoencoder.encoder, task_name=self.task_name)
        save_checkpoint(self.autoencoder.decoder, task_name=self.task_name)

    @staticmethod
    # Go from (N, H, W) of [0, 255] to (N, 1, H, W) of [-1, 1]
    def _preprocess_images(images):
        return images[:, None].astype(np.float32) / (255./2) - 1.0


class RewirlDecoderTester():
    def __init__(self, task_name=RDTEP.TASK_NAME, show=RDTEP.SHOW,
                 im_dim=RDTEP.IM_DIM, seq_len=RDTEP.SEQ_LEN, max_speed=RDTEP.MAX_SPEED, obj_size=RDTEP.OBJ_SIZE,
                 num_distractors=RDTEP.NUM_DISTRACTORS, window=RDTEP.WINDOW,
                 enc_output_dim=RDTEP.ENC_OUTPUT_DIM, num_encoder_linear_layers=RDTEP.NUM_ENC_LIN_LAYERS,
                 load_encoder=RDTEP.LOAD_ENC, num_trials=RDTEP.TRIALS, figure_file=RDTEP.FIG_FILE):
        self.task_name = task_name
        self.show = show
        self.window = window
        self.num_trials = num_trials
        self.im_dims = (1, 1, im_dim, im_dim)
        self.spec = AttrDict(
            resolution=im_dim,
            max_seq_len=seq_len,
            max_speed=max_speed,
            obj_size=obj_size,
            shapes_per_traj=2 + num_distractors,      # number of shapes per trajectory
            rewards=[ZeroReward],
        )
        self.gen = DistractorTemplateMovingSpritesGenerator(self.spec)

        self.autoencoder = RewirlAutoencoder(im_dim, enc_output_dim=enc_output_dim,
                                             retrain_encoder=False, load_encoder=load_encoder,
                                             num_encoder_linear_layers=num_encoder_linear_layers)
        load_checkpoint(self.autoencoder)

        self.loss_fn = nn.BCELoss(reduction="sum")
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.figure_file = figure_file

    def test(self):
        losses = []
        for i in range(self.num_trials):
            traj = self.gen.gen_trajectory()
            for im in traj.images:
                print(np.max(im))
                print(im.shape)
                test_label = T.tensor(im / 255., dtype=T.float)
                preproc_image = RewirlTrainer._preprocess_images(im[None])
                print(preproc_image.shape)
                test_data = T.tensor(preproc_image, dtype=T.float)
                pred_label = T.squeeze(self.autoencoder(test_data))
                print(pred_label.shape)
                # print(test_label)
                # print(pred_label)
                loss = self.loss_fn(pred_label.to(self.device), test_label.to(self.device))
                losses.append(loss.item())
                if self.show:
                    cv2.imshow(self.task_name, im[:, :, None].repeat(3, axis=2).astype(np.float32))
                    cv2.imshow(self.task_name+"_pred",
                               pred_label.detach().numpy()[:, :, None].repeat(3, axis=2).astype(np.float32))
                    cv2.waitKey(0)
                avg_loss = np.mean(losses[-self.window:])
                if i % self.window == 0:
                    print(f"iter = {i}, loss = {loss.item()}, avg_loss = {avg_loss}")
                    plt.plot(losses)
                    plt.title("RewIRL Decoder losses")
                    plt.savefig(self.figure_file)
                self.autoencoder.optimizer.zero_grad()
                loss.backward()
                self.autoencoder.optimizer.step()
        print(f"average loss:{np.mean(losses)}")

    @staticmethod
    # Go from (N, H, W) of [0, 255] to (N, 1, H, W) of [-1, 1]
    def _preprocess_images(images):
        return images[:, None].astype(np.float32) / (255./2) - 1.0


if __name__ == "__main__":
    print_architecture = True
    train_rewirl = False
    test_rewirl = False
    train_rewirl_decoder = True
    test_rewirl_decoder = False

    if train_rewirl:
        rtr = RewirlTrainer()
        if print_architecture:
            print(rtr.ra)
        rtr.train()

    if test_rewirl:
        rte = RewirlTester()
        if print_architecture:
            print(rte.ra)
        rte.test()

    if train_rewirl_decoder:
        rdt = RewirlDecoderTrainer()
        if print_architecture:
            print(rdt)
        rdt.train()

    if test_rewirl_decoder:
        rdte = RewirlDecoderTester()
        if print_architecture:
            print(rdte)
        rdte.test()
