from params import STATE_PPO_PARAMS as SPP
from params import STATE_PPO_TRAINER_PARAMS as SPTP
from params import STATE_PPO_TESTER_PARAMS as SPTEP
from params import VISUAL_PPO_TRAINER_PARAMS as VPTP
from params import VISUAL_PPO_TESTER_PARAMS as VPTEP
from models.actor import StateActor, VisualActor
from models.critic import StateCritic, VisualCritic
from utils.general_utils import AttrDict
from utils.model_utils import shuffled_indices, save_checkpoint, load_checkpoint, plot_learning_curve

import cv2
import gym
import numpy as np
import torch as T
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start_indices = np.arange(0, n_states, self.batch_size)
        indices = shuffled_indices(n_states)
        batches = [indices[i:i + self.batch_size] for i in batch_start_indices]
        return [np.array(self.states),
                np.array(self.actions),
                np.array(self.probs),
                np.array(self.values),
                np.array(self.rewards),
                np.array(self.dones),
                batches]

    def store_memory(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []


class PPOAgent():
    def __init__(self, input_dim, task_name=SPP.TASK_NAME, num_actions=SPP.NUM_ACTIONS,
                 a_alpha=SPP.ACT_LR, c_alpha=SPP.CRIT_LR,
                 gamma=SPP.GAMMA, gae_lambda=SPP.GAE_LAMBDA, policy_clip=SPP.POL_CLIP,
                 use_encoder=SPP.USE_ENCODER, enc_output_dim=SPP.ENC_OUTPUT_DIM,
                 num_encoder_linear_layers=SPP.NUM_ENC_LIN_LAYERS, load_encoder=SPP.LOAD_ENC,
                 retrain_encoder=SPP.RETRAIN_ENC):
        if use_encoder:
            self.actor = VisualActor(input_dim, output_dim=num_actions, alpha=a_alpha, enc_output_dim=enc_output_dim,
                                     num_encoder_linear_layers=num_encoder_linear_layers,
                                     load_encoder=load_encoder, retrain_encoder=retrain_encoder)
            self.critic = VisualCritic(input_dim, alpha=c_alpha, enc_output_dim=enc_output_dim,
                                       num_encoder_linear_layers=num_encoder_linear_layers,
                                       load_encoder=load_encoder, retrain_encoder=retrain_encoder)
        else:
            self.actor = StateActor(input_dim, output_dim=num_actions, alpha=a_alpha)
            self.critic = StateCritic(input_dim, alpha=c_alpha)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.task_name = task_name

    def save_models(self):
        print("...saving models...")
        save_checkpoint(self.actor, task_name=self.task_name)
        save_checkpoint(self.critic, task_name=self.task_name)

    def load_models(self):
        print("...loading models...")
        load_checkpoint(self.actor, task_name=self.task_name)
        load_checkpoint(self.critic, task_name=self.task_name)

    def choose_action(self, state, eps=0.2):
        value = self.critic(state)
        act_probs = self.actor(state)
        p = np.random.rand()
        if p > eps:
            action = T.argmax(act_probs)
            prob = T.log(act_probs[0, action])
        else:
            dist = Categorical(act_probs)
            action = dist.sample()
            prob = dist.log_prob(action)
        prob = T.squeeze(prob).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, prob, value


class PPOTrainer():
    def __init__(self, task_name=SPTP.TASK_NAME, env_name=SPTP.ENV_NAME, state_dim=SPTP.STATE_DIM,
                 action_dict=SPTP.ACT_DICT, resolution=SPTP.RESOLUTION, max_ep_len=SPTP.MAX_EP_LEN,
                 obj_size=SPTP.OBJ_SIZE, speed=SPTP.SPEED, num_games=SPTP.NUM_GAMES, batch_size=SPTP.BATCH_SIZE,
                 epochs=SPTP.EPOCHS, learn_trigger=SPTP.LEARN_TRIGGER, avg_window=SPTP.AVG_WINDOW,
                 show_freq=SPTP.SHOW_FREQ, figure_file=SPTP.FIG_FILE, use_encoder=SPTP.USE_ENCODER,
                 enc_output_dim=SPTP.ENC_OUTPUT_DIM, num_encoder_linear_layers=SPTP.NUM_ENC_LIN_LAYERS,
                 load_encoder=SPTP.LOAD_ENC, retrain_encoder=SPTP.RETRAIN_ENC):
        self.task_name = task_name
        self.num_games = num_games
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_trigger = learn_trigger
        self.ppo_memory = PPOMemory(batch_size)
        self.action_dict = action_dict
        self.use_encoder = use_encoder
        self.ppo_agent = PPOAgent(state_dim, task_name=task_name, num_actions=len(self.action_dict),
                                  use_encoder=use_encoder, enc_output_dim=enc_output_dim,
                                  num_encoder_linear_layers=num_encoder_linear_layers,
                                  load_encoder=load_encoder, retrain_encoder=retrain_encoder)
        self.data_spec = AttrDict(
            resolution=resolution,
            max_ep_len=max_ep_len,
            max_speed=speed,  # total image range [0, 1]
            obj_size=obj_size,  # size of objects, full images is 1.0
            follow=True
        )
        self.env = gym.make(env_name)
        self.env.set_config(self.data_spec)
        self.avg_window = avg_window
        self.show_freq = show_freq
        self.figure_file = figure_file

    def learn(self):
        for _ in range(self.epochs):
            states, actions, old_probs, values, rewards, dones, batches = self.ppo_memory.generate_batches()
            advantage = np.zeros(len(rewards), dtype=np.float32)
            discount = self.ppo_agent.gamma * self.ppo_agent.gae_lambda
            for t in reversed(range(len(rewards)-1)):
                delta_t = rewards[t] + self.ppo_agent.gamma * values[t+1] * (1 - int(dones[t])) - values[t]
                advantage[t] = delta_t + discount * (1 - int(dones[t])) * advantage[t+1]
            advantage = T.tensor(advantage)
            values = T.tensor(values)

            for batch in batches:
                batch_states = T.tensor(states[batch], dtype=T.float)
                batch_old_probs = T.tensor(old_probs[batch])
                batch_actions = T.tensor(actions[batch])

                probs = self.ppo_agent.actor(batch_states)
                critic_value = self.ppo_agent.critic(batch_states)
                dist = Categorical(probs)
                critic_value = T.squeeze(critic_value)
                batch_new_probs = dist.log_prob(batch_actions)
                prob_ratio = batch_new_probs.exp() / batch_old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.ppo_agent.policy_clip,
                                                 1 + self.ppo_agent.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.ppo_agent.actor.optimizer.zero_grad()
                self.ppo_agent.critic.optimizer.zero_grad()
                total_loss.backward()
                self.ppo_agent.actor.optimizer.step()
                self.ppo_agent.critic.optimizer.step()
        self.ppo_memory.clear_memory()

    def train(self):
        best_score = self.env.reward_range[0]
        score_history = []
        learn_iters = 0
        avg_score = 0
        n_steps = 0

        for i in range(self.num_games):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                if self.use_encoder:
                    state = PPOTrainer._preprocess_images(state)
                    agent_input = T.tensor([state], dtype=T.float)
                else:
                    agent_input = T.tensor([state], dtype=T.float)
                action, prob, val = self.ppo_agent.choose_action(agent_input, eps=1 - (i / self.num_games))
                state_, reward, done, _, im = self.env.step(self.action_dict[action])
                if self.show_freq and (i % self.show_freq == 0):
                    cv2.imshow(self.task_name, im[:, :, None].repeat(3, axis=2).astype(np.float32) * 255.)
                    cv2.waitKey(5)
                n_steps += 1
                score += reward
                self.ppo_memory.store_memory(state, action, prob, val, reward, done)
                if n_steps % self.learn_trigger == 0:
                    self.learn()
                    learn_iters += 1
                state = state_
            score_history.append(score)
            avg_score = np.mean(score_history[-self.avg_window:])
            if avg_score > best_score:
                best_score = avg_score
                self.ppo_agent.save_models()

            print(f"episode: {i}, score: {score}, avg score: {avg_score},\
                  time_steps: {n_steps}, learning_steps: {learn_iters}")
            plot_learning_curve([i+1 for i in range(len(score_history))], score_history,
                                self.avg_window, self.figure_file)

    @staticmethod
    # Go from (H, W) of [0, 1] to (1, H, W) of [-1, 1]
    def _preprocess_images(images):
        return images[None, :, :].astype(np.float32) * 2.0 - 1.0


class PPOTester():
    def __init__(self, task_name=SPTP.TASK_NAME, env_name=SPTEP.ENV_NAME, state_dim=SPTEP.STATE_DIM,
                 action_dict=SPTEP.ACT_DICT, resolution=SPTEP.RESOLUTION, max_ep_len=SPTEP.MAX_EP_LEN,
                 obj_size=SPTEP.OBJ_SIZE, speed=SPTEP.SPEED, num_games=SPTEP.NUM_GAMES, avg_window=SPTEP.AVG_WINDOW,
                 show=SPTEP.SHOW, figure_file=SPTEP.FIG_FILE, use_encoder=SPTEP.USE_ENCODER,
                 enc_output_dim=SPTEP.ENC_OUTPUT_DIM, num_encoder_linear_layers=SPTEP.NUM_ENC_LIN_LAYERS):
        self.task_name = task_name
        self.num_games = num_games
        self.action_dict = action_dict
        self.data_spec = AttrDict(
            resolution=resolution,
            max_ep_len=max_ep_len,
            max_speed=speed,  # total image range [0, 1]
            obj_size=obj_size,  # size of objects, full images is 1.0
            follow=True
        )
        self.env = gym.make(env_name)
        self.env.set_config(self.data_spec)
        self.avg_window = avg_window
        self.figure_file = figure_file
        self.show = show

        self.use_encoder = use_encoder
        if use_encoder:
            self.actor = VisualActor(state_dim, output_dim=len(self.action_dict), enc_output_dim=enc_output_dim,
                                     num_encoder_linear_layers=num_encoder_linear_layers,
                                     load_encoder=False, retrain_encoder=False)
        else:
            self.actor = StateActor(state_dim, output_dim=len(self.action_dict))
        load_checkpoint(self.actor, task_name=self.task_name)
        self.actor.eval()

    def test(self):
        score_history = []

        for i in range(self.num_games):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                with T.no_grad():
                    if self.use_encoder:
                        state = PPOTrainer._preprocess_images(state)
                    act_probs = self.actor(T.tensor([state], dtype=T.float))
                    action = T.argmax(act_probs)
                state, reward, done, _, im = self.env.step(self.action_dict[action])
                if self.show:
                    cv2.imshow(self.task_name, im[:, :, None].repeat(3, axis=2).astype(np.float32) * 255.)
                    cv2.waitKey(10)
                score += reward
            score_history.append(score)
            avg_score = np.mean(score_history[-self.avg_window:])

            print(f"episode: {i}, score: {score}, avg score: {avg_score}")
            plot_learning_curve([i+1 for i in range(len(score_history))], score_history,
                                self.avg_window, self.figure_file)


if __name__ == "__main__":
    print_architecture = True
    train_state_ppo = False
    test_state_ppo = False
    train_scratch_ppo = True
    test_scratch_ppo = False
    train_static_rewirl_ppo = False
    test_static_rewirl_ppo = False
    train_finetune_rewirl_ppo = False
    test_finetune_rewirl_ppo = False

    if train_state_ppo:
        spt = PPOTrainer(task_name=SPTP.TASK_NAME, env_name=SPTP.ENV_NAME, state_dim=SPTP.STATE_DIM,
                         action_dict=SPTP.ACT_DICT, resolution=SPTP.RESOLUTION, max_ep_len=SPTP.MAX_EP_LEN,
                         obj_size=SPTP.OBJ_SIZE, speed=SPTP.SPEED, num_games=SPTP.NUM_GAMES, batch_size=SPTP.BATCH_SIZE,
                         epochs=SPTP.EPOCHS, learn_trigger=SPTP.LEARN_TRIGGER, avg_window=SPTP.AVG_WINDOW,
                         show_freq=SPTP.SHOW_FREQ, figure_file=SPTP.FIG_FILE, use_encoder=SPTP.USE_ENCODER,
                         enc_output_dim=SPTP.ENC_OUTPUT_DIM, num_encoder_linear_layers=SPTP.NUM_ENC_LIN_LAYERS,
                         load_encoder=SPTP.LOAD_ENC, retrain_encoder=SPTP.RETRAIN_ENC)
        if print_architecture:
            print(spt.ppo_agent.actor)
            print(spt.ppo_agent.critic)
        spt.train()

    if test_state_ppo:
        spte = PPOTester(task_name=SPTEP.TASK_NAME, env_name=SPTEP.ENV_NAME, state_dim=SPTEP.STATE_DIM,
                         action_dict=SPTEP.ACT_DICT, resolution=SPTEP.RESOLUTION, max_ep_len=SPTEP.MAX_EP_LEN,
                         obj_size=SPTEP.OBJ_SIZE, speed=SPTEP.SPEED, num_games=SPTEP.NUM_GAMES,
                         avg_window=SPTEP.AVG_WINDOW, show=SPTEP.SHOW, figure_file=SPTEP.FIG_FILE,
                         use_encoder=SPTEP.USE_ENCODER, enc_output_dim=SPTEP.ENC_OUTPUT_DIM,
                         num_encoder_linear_layers=SPTEP.NUM_ENC_LIN_LAYERS)
        if print_architecture:
            print(spte.actor)
        spte.test()

    if train_scratch_ppo:
        srpt = PPOTrainer(task_name=f"{VPTP.TASK_NAME}_SCRATCH", env_name=VPTP.ENV_NAME, state_dim=VPTP.STATE_DIM,
                          action_dict=VPTP.ACT_DICT, resolution=VPTP.RESOLUTION, max_ep_len=VPTP.MAX_EP_LEN,
                          obj_size=VPTP.OBJ_SIZE, speed=VPTP.SPEED, num_games=VPTP.NUM_GAMES,
                          batch_size=VPTP.BATCH_SIZE, epochs=VPTP.EPOCHS, learn_trigger=VPTP.LEARN_TRIGGER,
                          avg_window=VPTP.AVG_WINDOW, show_freq=VPTP.SHOW_FREQ,
                          figure_file="plots/visual_scratch_ppo.png",
                          use_encoder=VPTP.USE_ENCODER, enc_output_dim=VPTP.ENC_OUTPUT_DIM,
                          num_encoder_linear_layers=VPTP.NUM_ENC_LIN_LAYERS, load_encoder=False,
                          retrain_encoder=True)
        if print_architecture:
            print(srpt.ppo_agent.actor)
            print(srpt.ppo_agent.critic)
        srpt.train()

    if test_scratch_ppo:
        srpte = PPOTester(task_name=f"{VPTEP.TASK_NAME}_SCRATCH", env_name=VPTEP.ENV_NAME, state_dim=VPTEP.STATE_DIM,
                          action_dict=VPTEP.ACT_DICT, resolution=VPTEP.RESOLUTION, max_ep_len=VPTEP.MAX_EP_LEN,
                          obj_size=VPTEP.OBJ_SIZE, speed=VPTEP.SPEED, num_games=VPTEP.NUM_GAMES,
                          avg_window=VPTEP.AVG_WINDOW, show=VPTEP.SHOW, figure_file="plots/visual_scratch_ppo_test.png",
                          use_encoder=VPTEP.USE_ENCODER, enc_output_dim=VPTEP.ENC_OUTPUT_DIM,
                          num_encoder_linear_layers=VPTEP.NUM_ENC_LIN_LAYERS)
        if print_architecture:
            print(srpte.actor)
        srpte.test()

    if train_static_rewirl_ppo:
        srpt = PPOTrainer(task_name=VPTP.TASK_NAME, env_name=VPTP.ENV_NAME, state_dim=VPTP.STATE_DIM,
                          action_dict=VPTP.ACT_DICT, resolution=VPTP.RESOLUTION, max_ep_len=VPTP.MAX_EP_LEN,
                          obj_size=VPTP.OBJ_SIZE, speed=VPTP.SPEED, num_games=VPTP.NUM_GAMES,
                          batch_size=VPTP.BATCH_SIZE, epochs=VPTP.EPOCHS, learn_trigger=VPTP.LEARN_TRIGGER,
                          avg_window=VPTP.AVG_WINDOW, show_freq=VPTP.SHOW_FREQ, figure_file=VPTP.FIG_FILE,
                          use_encoder=VPTP.USE_ENCODER, enc_output_dim=VPTP.ENC_OUTPUT_DIM,
                          num_encoder_linear_layers=VPTP.NUM_ENC_LIN_LAYERS, load_encoder=VPTP.LOAD_ENC,
                          retrain_encoder=VPTP.RETRAIN_ENC)
        if print_architecture:
            print(srpt.ppo_agent.actor)
            print(srpt.ppo_agent.critic)
        srpt.train()

    if test_static_rewirl_ppo:
        srpte = PPOTester(task_name=VPTEP.TASK_NAME, env_name=VPTEP.ENV_NAME, state_dim=VPTEP.STATE_DIM,
                          action_dict=VPTEP.ACT_DICT, resolution=VPTEP.RESOLUTION, max_ep_len=VPTEP.MAX_EP_LEN,
                          obj_size=VPTEP.OBJ_SIZE, speed=VPTEP.SPEED, num_games=VPTEP.NUM_GAMES,
                          avg_window=VPTEP.AVG_WINDOW, show=VPTEP.SHOW, figure_file=VPTEP.FIG_FILE,
                          use_encoder=VPTEP.USE_ENCODER, enc_output_dim=VPTEP.ENC_OUTPUT_DIM,
                          num_encoder_linear_layers=VPTEP.NUM_ENC_LIN_LAYERS)
        if print_architecture:
            print(srpte.actor)
        srpte.test()

    if train_finetune_rewirl_ppo:
        frpt = PPOTrainer(task_name=f"{VPTP.TASK_NAME}_FINETUNE", env_name=VPTP.ENV_NAME, state_dim=VPTP.STATE_DIM,
                          action_dict=VPTP.ACT_DICT, resolution=VPTP.RESOLUTION, max_ep_len=VPTP.MAX_EP_LEN,
                          obj_size=VPTP.OBJ_SIZE, speed=VPTP.SPEED, num_games=VPTP.NUM_GAMES,
                          batch_size=VPTP.BATCH_SIZE, epochs=VPTP.EPOCHS, learn_trigger=VPTP.LEARN_TRIGGER,
                          avg_window=VPTP.AVG_WINDOW, show_freq=VPTP.SHOW_FREQ,
                          figure_file="plots/visual_finetune_ppo.png",
                          use_encoder=VPTP.USE_ENCODER, enc_output_dim=VPTP.ENC_OUTPUT_DIM,
                          num_encoder_linear_layers=VPTP.NUM_ENC_LIN_LAYERS, load_encoder=VPTP.LOAD_ENC,
                          retrain_encoder=True)
        if print_architecture:
            print(frpt.ppo_agent.actor)
            print(frpt.ppo_agent.critic)
        frpt.train()

    if test_finetune_rewirl_ppo:
        frpte = PPOTester(task_name=f"{VPTEP.TASK_NAME}_FINETUNE", env_name=VPTEP.ENV_NAME, state_dim=VPTEP.STATE_DIM,
                          action_dict=VPTEP.ACT_DICT, resolution=VPTEP.RESOLUTION, max_ep_len=VPTEP.MAX_EP_LEN,
                          obj_size=VPTEP.OBJ_SIZE, speed=VPTEP.SPEED, num_games=VPTEP.NUM_GAMES,
                          avg_window=VPTEP.AVG_WINDOW, show=VPTEP.SHOW,
                          figure_file="plots/visual_finetune_ppo_test.png",
                          use_encoder=VPTEP.USE_ENCODER, enc_output_dim=VPTEP.ENC_OUTPUT_DIM,
                          num_encoder_linear_layers=VPTEP.NUM_ENC_LIN_LAYERS)
        if print_architecture:
            print(frpte.actor)
        frpte.test()
