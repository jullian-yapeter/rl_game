# Common default params
class COMMON_PARAMS:
    CHECKPOINT_DIR = "saved_models"


# Image encoder default params
class IM_ENCODER_PARAMS:
    LEARNING_RATE = 0.001
    OUTPUT_DIM = 64
    NUM_LINEAR_LAYERS = 1
    CHKPT_FILE = "im_encoder"


# Reward head default params
class R_HEAD_PARAMS:
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 32
    OUTPUT_DIM = 1
    NUM_LAYERS = 3
    CHKPT_FILE = "r_head"
