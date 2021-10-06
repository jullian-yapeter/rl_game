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


# Image decoder default params
class IM_DECODER_PARAMS:
    LEARNING_RATE = 0.001
    IN_ACTIVATION = nn.ReLU()
    OUT_ACTIVATION = nn.Sigmoid()
    OUTPUT_DIM = 4096
    OUTPUT_IM_DIMS = (1, 64, 64)
    CHKPT_FILE = "im_decoder"


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
    USE_ENCODER = False
    LOAD_ENC = False
    IM_DIM = 64
    NUM_ENC_LIN_LAYERS = 1
    RETRAIN_ENC = False


# State-based Critic network default params
class STATE_CRITIC_PARAMS:
    LEARNING_RATE = 0.001
    ACTIVATION = nn.ReLU()
    HIDDEN_DIM = 32
    NUM_LAYERS = 3
    CHKPT_FILE = "st_critic"


# Visual-based Actor network default params
class VISUAL_ACTOR_PARAMS:
    LEARNING_RATE = 0.001
    IN_ACTIVATION = nn.Tanh()
    OUT_ACTIVATION = nn.Softmax(dim=-1)
    HIDDEN_DIM = 32
    NUM_ACTIONS = 4
    NUM_LAYERS = 3
    LOAD_ENC = True
    ENC_OUTPUT_DIM = 64
    NUM_ENC_LIN_LAYERS = 1
    RETRAIN_ENC = False
    CHKPT_FILE = "vis_actor"


# Visual-based Critic network default params
class VISUAL_CRITIC_PARAMS:
    LEARNING_RATE = 0.001
    ACTIVATION = nn.ReLU()
    HIDDEN_DIM = 32
    NUM_LAYERS = 3
    LOAD_ENC = True
    ENC_OUTPUT_DIM = 64
    NUM_ENC_LIN_LAYERS = 1
    RETRAIN_ENC = False
    CHKPT_FILE = "vis_critic"
