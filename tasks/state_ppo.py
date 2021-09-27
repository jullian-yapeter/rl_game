from params import STATE_PPO_PARAMS as SPP
from params import STATE_PPO_TRAINER_PARAMS as SPTP
from models.actor import StateActor
from models.critic import StateCritic
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


class StatePPOAgent():
    def __init__(self, input_dim, num_actions=SPP.NUM_ACTIONS,
                 a_alpha=SPP.ACT_LR, c_alpha=SPP.CRIT_LR,
                 gamma=SPP.GAMMA, gae_lambda=SPP.GAE_LAMBDA, policy_clip=SPP.POL_CLIP):
        self.actor = StateActor(input_dim, output_dim=num_actions, alpha=a_alpha)
        self.critic = StateCritic(input_dim, alpha=c_alpha)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.task_name = SPP.TASK_NAME

    def save_models(self):
        print("...saving models...")
        save_checkpoint(self.actor, task_name=self.task_name)
        save_checkpoint(self.critic, task_name=self.task_name)

    def load_models(self):
        print("...loading models...")
        load_checkpoint(self.actor, task_name=self.task_name)
        load_checkpoint(self.critic, task_name=self.task_name)

    def choose_action(self, state):
        value = self.critic(state)
        act_probs = self.actor(state)
        dist = Categorical(act_probs)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value


class StatePPOTrainer():
    def __init__(self, env_name=SPTP.ENV_NAME, state_dim=SPTP.STATE_DIM, resolution=SPTP.RESOLUTION,
                 max_ep_len=SPTP.MAX_EP_LEN, obj_size=SPTP.OBJ_SIZE, speed=SPTP.SPEED, num_games=SPTP.NUM_GAMES,
                 batch_size=SPTP.BATCH_SIZE, epochs=SPTP.EPOCHS, learn_trigger=SPTP.LEARN_TRIGGER,
                 figure_file=SPTP.FIG_FILE):
        self.num_games = num_games
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_trigger = learn_trigger
        self.ppo_memory = PPOMemory(batch_size)
        self.action_dict = [[0, 0], [speed, 0],
                            [0, -speed],
                            [-speed, 0],
                            [0, speed]]
        self.st_ppo_agent = StatePPOAgent(state_dim, len(self.action_dict))
        self.data_spec = AttrDict(
            resolution=resolution,
            max_ep_len=max_ep_len,
            max_speed=speed,  # total image range [0, 1]
            obj_size=obj_size,  # size of objects, full images is 1.0
            follow=True
        )
        self.env = gym.make(env_name)
        self.env.set_config(self.data_spec)
        self.figure_file = figure_file

    def learn(self):
        for _ in range(self.epochs):
            states, actions, old_probs, values, rewards, dones, batches = self.ppo_memory.generate_batches()
            advantage = np.zeros(len(rewards), dtype=np.float32)
            discount = self.st_ppo_agent.gamma * self.st_ppo_agent.gae_lambda
            for t in reversed(range(len(rewards)-1)):
                delta_t = rewards[t] + self.st_ppo_agent.gamma * values[t+1] * (1 - int(dones[t])) - values[t]
                advantage[t] = delta_t + discount * (1 - int(dones[t])) * advantage[t+1]
            advantage = T.tensor(advantage)
            values = T.tensor(values)

            for batch in batches:
                batch_states = T.tensor(states[batch], dtype=T.float)
                batch_old_probs = T.tensor(old_probs[batch])
                batch_actions = T.tensor(actions[batch])

                probs = self.st_ppo_agent.actor(batch_states)
                critic_value = self.st_ppo_agent.critic(batch_states)
                dist = Categorical(probs)
                critic_value = T.squeeze(critic_value)
                batch_new_probs = dist.log_prob(batch_actions)
                prob_ratio = batch_new_probs.exp() / batch_old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.st_ppo_agent.policy_clip,
                                                 1 + self.st_ppo_agent.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.st_ppo_agent.actor.optimizer.zero_grad()
                self.st_ppo_agent.critic.optimizer.zero_grad()
                total_loss.backward()
                self.st_ppo_agent.actor.optimizer.step()
                self.st_ppo_agent.critic.optimizer.step()
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
                action, prob, val = self.st_ppo_agent.choose_action(T.tensor([state], dtype=T.float))
                state_, reward, done, _, im = self.env.step(self.action_dict[action])
                # cv2.imshow("1", im[:, :, None].repeat(3, axis=2).astype(np.float32) * 255.)
                # cv2.waitKey(500)
                n_steps += 1
                score += reward
                self.ppo_memory.store_memory(state, action, prob, val, reward, done)
                if n_steps % self.learn_trigger == 0:
                    self.learn()
                    learn_iters += 1
                state = state_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if avg_score > best_score:
                best_score = avg_score
                self.st_ppo_agent.save_models()

            print(f"episode: {i}, score: {score}, avg score: {avg_score}, \
                time_steps: {n_steps}, learning_steps: {learn_iters}")

        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, self.figure_file)


if __name__ == "__main__":
    spt = StatePPOTrainer()
    spt.train()
    # print(spt.st_ppo_agent.actor)
    # print(spt.st_ppo_agent.critic)
