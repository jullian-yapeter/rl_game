from envs.sprites.sprites_datagen.rewards import AgentXReward, AgentYReward, TargetXReward, TargetYReward


# Common default params
class COMMON_PARAMS:
    CHECKPOINT_DIR = "saved_models"


# Reward-Induced Representation RL (Rewirl) Agent default params
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
    EPOCHS = 10
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


# State-based PPO agent default params
class STATE_PPO_PARAMS:
    TASK_NAME = "PPO"
    NUM_ACTIONS = 5
    ACT_LR = 0.0001
    CRIT_LR = 0.001
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    POL_CLIP = 0.2


# State-based PPO trainer default params
class STATE_PPO_TRAINER_PARAMS:
    TASK_NAME = "PPO"
    ENV_NAME = 'envs.sprites.sprites_env:SpritesState-v0'
    NUM_GAMES = 10000
    STATE_DIM = 4
    RESOLUTION = 64
    MAX_EP_LEN = 2049
    OBJ_SIZE = 0.2
    SPEED = 0.05
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARN_TRIGGER = 192
    FIG_FILE = "plots/state_ppo.png"
