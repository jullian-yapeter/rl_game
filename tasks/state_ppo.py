from params import STATE_PPO_PARAMS as SPP
from models.actor import StateActor
from models.critic import StateCritic
from utils.model_utils import shuffled_indices, save_checkpoint, load_checkpoint

import numpy as np
import torch as T
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
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
                np.array(self.vals),
                np.array(self.rewards),
                np.array(self.dones),
                batches]

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class StatePPOAgent():
    def __init__(self, input_dim, num_actions=SPP.NUM_ACTIONS,
                 a_alpha=SPP.ACT_LR, c_alpha=SPP.CRIT_LR,
                 gamma=SPP.GAMMA, gae_lambda=SPP.GAE_LAMBDA, policy_clip=SPP.POL_CLIP):
        self.actor = StateActor(input_dim, output_dim=num_actions, alpha=a_alpha)
        self.critic = StateCritic(input_dim, alpha=c_alpha)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.memory = PPOMemory()
        self.task_name = SPP.TASK_NAME

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("...saving models...")
        save_checkpoint(self.actor, task_name=self.task_name)
        save_checkpoint(self.critic, task_name=self.task_name)

    def load_models(self):
        print("...loading models...")
        load_checkpoint(self.actor, task_name=self.task_name)
        load_checkpoint(self.critic, task_name=self.task_name)

    def choose_action(self, state):
        state = state.to(self.actor.device)
        value = self.critic(state)
        act_probs = self.actor(state)
        dist = Categorical(act_probs)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value
