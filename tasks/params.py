from envs.sprites.sprites_datagen.rewards import AgentXReward, AgentYReward, TargetXReward, TargetYReward


# Common default params
class COMMON_PARAMS:
    CHECKPOINT_DIR = "saved_models"


# Image encoder default params
class REWIRL_AGENT_PARAMS:
    LEARNING_RATE = 0.001
    ENC_OUTPUT_DIM = 64
    HEAD_HIDDEN_DIM = 32
    HEAD_OUTPUT_DIM = 1
    NUM_ENC_LIN_LAYERS = 1
    NUM_HEAD_LAYERS = 3
    CHKPT_FILE = "rewirl"


# Rewirl trainer default params
class REWIRL_TRAINER_PARAMS:
    TASK_NAME = "rewirl"
    REWARD_TYPES = [AgentXReward, AgentYReward, TargetXReward, TargetYReward]
    IM_DIM = 64
    MAX_SPEED = 0.05
    OBJ_SIZE = 0.2
    NUM_DISTRACTORS = 0
    EPOCHS = 1000
    BATCH_SIZE = 30


# Rewirl trainer default params
class REWIRL_TESTER_PARAMS:
    REWARD_TYPES = [AgentXReward, AgentYReward, TargetXReward, TargetYReward]
    IM_DIM = 64
    SEQ_LEN = 10
    MAX_SPEED = 0.05
    OBJ_SIZE = 0.2
    NUM_DISTRACTORS = 0
    TRIALS = 10
