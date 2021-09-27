import torch.nn as nn


# Common default params
class COMMON_PARAMS:
    CHECKPOINT_DIR = "saved_models"


# Image encoder default params
class IM_ENCODER_PARAMS:
    LEARNING_RATE = 0.001
    ACTIVATION = nn.ReLU()
    OUTPUT_DIM = 64
    NUM_LINEAR_LAYERS = 1
    CHKPT_FILE = "im_encoder"


# Reward head default params
class R_HEAD_PARAMS:
    LEARNING_RATE = 0.001
    ACTIVATION = nn.ReLU()
    HIDDEN_DIM = 32
    OUTPUT_DIM = 1
    NUM_LAYERS = 3
    CHKPT_FILE = "r_head"


# State-based Actor network default params
class STATE_ACTOR_PARAMS:
    LEARNING_RATE = 0.001
    IN_ACTIVATION = nn.Tanh()
    OUT_ACTIVATION = nn.Softmax(dim=-1)
    HIDDEN_DIM = 32
    NUM_ACTIONS = 4
    NUM_LAYERS = 3
    CHKPT_FILE = "st_actor"


# State-based Critic network default params
class STATE_CRITIC_PARAMS:
    LEARNING_RATE = 0.001
    ACTIVATION = nn.ReLU()
    HIDDEN_DIM = 32
    NUM_LAYERS = 3
    CHKPT_FILE = "st_critic"
