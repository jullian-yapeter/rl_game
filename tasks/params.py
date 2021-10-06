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
    EPOCHS = 20000
    BATCH_SIZE = 30
    WINDOW = 10
    FIG_FILE = "plots/rewirl_train.png"


# Rewirl trainer default params
class REWIRL_TESTER_PARAMS:
    REWARD_TYPES = [AgentXReward, AgentYReward, TargetXReward, TargetYReward]
    IM_DIM = 64
    SEQ_LEN = 10
    MAX_SPEED = 0.05
    OBJ_SIZE = 0.2
    NUM_DISTRACTORS = 0
    TRIALS = 10
    FIG_FILE = "plots/rewirl_test.png"


# State-based PPO agent default params
class STATE_PPO_PARAMS:
    TASK_NAME = "PPO"
    NUM_ACTIONS = 4
    ACT_LR = 0.0001  # 0.0003
    CRIT_LR = 0.001  # 0.0003
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    POL_CLIP = 0.2  # 0.1
    USE_ENCODER = False
    ENC_OUTPUT_DIM = None
    NUM_ENC_LIN_LAYERS = None
    LOAD_ENC = False
    RETRAIN_ENC = False


# State-based PPO trainer default params
class STATE_PPO_TRAINER_PARAMS:
    TASK_NAME = "PPO"
    ENV_NAME = 'envs.sprites.sprites_env:SpritesState-v0'
    NUM_GAMES = 3000
    STATE_DIM = 4
    ACT_DICT = [[1, 0], [0, -1], [-1, 0], [0, 1]]
    RESOLUTION = 64
    MAX_EP_LEN = 2049
    OBJ_SIZE = 0.2
    SPEED = 0.05
    BATCH_SIZE = 256
    EPOCHS = 10
    LEARN_TRIGGER = 2048
    AVG_WINDOW = 10
    SHOW_FREQ = 100
    FIG_FILE = "plots/state_ppo.png"
    ENCODER = False
    USE_ENCODER = False
    ENC_OUTPUT_DIM = None
    NUM_ENC_LIN_LAYERS = None
    LOAD_ENC = False
    RETRAIN_ENC = False


# State-based PPO tester default params
class STATE_PPO_TESTER_PARAMS:
    TASK_NAME = "PPO"
    ENV_NAME = 'envs.sprites.sprites_env:SpritesState-v0'
    NUM_GAMES = 100
    STATE_DIM = 4
    ACT_DICT = [[1, 0], [0, -1], [-1, 0], [0, 1]]
    RESOLUTION = 64
    MAX_EP_LEN = 2049
    OBJ_SIZE = 0.2
    SPEED = 0.05
    AVG_WINDOW = 10
    SHOW = False
    USE_ENCODER = False
    ENC_OUTPUT_DIM = None
    NUM_ENC_LIN_LAYERS = None
    FIG_FILE = "plots/state_ppo_test.png"


# Visual-based PPO agent default params
class VISUAL_PPO_PARAMS:
    TASK_NAME = "VIS_PPO"
    NUM_ACTIONS = 4
    ACT_LR = 0.0001  # 0.0003
    CRIT_LR = 0.001  # 0.0003
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    POL_CLIP = 0.2  # 0.1


# Visual-based PPO trainer default params
class VISUAL_PPO_TRAINER_PARAMS:
    TASK_NAME = "VIS_PPO"
    ENV_NAME = 'envs.sprites.sprites_env:Sprites-v0'
    NUM_GAMES = 3000
    RESOLUTION = 64
    STATE_DIM = (1, 1, RESOLUTION, RESOLUTION)
    ACT_DICT = [[1, 0], [0, -1], [-1, 0], [0, 1]]
    MAX_EP_LEN = 2049
    OBJ_SIZE = 0.2
    SPEED = 0.05
    BATCH_SIZE = 256
    EPOCHS = 10
    LEARN_TRIGGER = 2048
    AVG_WINDOW = 10
    SHOW_FREQ = 100
    FIG_FILE = "plots/visual_ppo.png"
    USE_ENCODER = True
    ENC_OUTPUT_DIM = 64
    NUM_ENC_LIN_LAYERS = 1
    LOAD_ENC = True
    RETRAIN_ENC = False


# Visual-based PPO tester default params
class VISUAL_PPO_TESTER_PARAMS:
    TASK_NAME = "VIS_PPO"
    ENV_NAME = 'envs.sprites.sprites_env:Sprites-v0'
    NUM_GAMES = 100
    ACT_DICT = [[1, 0], [0, -1], [-1, 0], [0, 1]]
    RESOLUTION = 64
    STATE_DIM = (1, 1, RESOLUTION, RESOLUTION)
    MAX_EP_LEN = 2049
    OBJ_SIZE = 0.2
    SPEED = 0.05
    AVG_WINDOW = 10
    SHOW = True
    USE_ENCODER = True
    ENC_OUTPUT_DIM = 64
    NUM_ENC_LIN_LAYERS = 1
    FIG_FILE = "plots/visual_ppo_test.png"
