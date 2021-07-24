import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, mini_batch_num, obs_shape):

        self.mini_batch_num = mini_batch_num
        self.obs = torch.zeros(num_steps + 1, obs_shape)
        self.rewards = torch.zeros(num_steps, 1)
        self.value_preds = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.action_log_probs = torch.zeros(num_steps, 1)
        self.action = torch.zeros(num_steps, 1, dtype=torch.long)
        self.masks = torch.ones(num_steps + 1, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action = self.action.to(device)
        self.masks = self.masks.to(device)

    def reset(self, obs):
        self.obs[self.step].copy_(obs.squeeze(0))
        self.masks[self.step].copy_(torch.zeros(1))

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.action[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs.squeeze())
        self.value_preds[self.step].copy_(value_preds.squeeze())
        self.rewards[self.step].copy_(rewards.squeeze())
        self.obs[self.step + 1].copy_(obs.squeeze())
        self.masks[self.step + 1].copy_(masks.squeeze())

        self.step = self.step + 1

    def update_reward(self, intrinsic_reward):
        intrinsic_reward = intrinsic_reward.clamp(-1, 1)
        num_steps = intrinsic_reward.size()[0]
        # print('self.rewards', self.rewards, 'intrinsic_reward', intrinsic_reward)
        for i in range(num_steps):
            self.rewards[i] = self.rewards[i] + intrinsic_reward[i]
        # self.rewards = self.rewards.clamp(-1, 1)

    def icm_tuple(self):
        obs = self.obs[:-1].clone().detach()
        next_obs = self.obs[1:].clone().detach()
        action = self.action.clone().detach()
        return obs, next_obs, action

    def after_update(self, obs):
        self.step = 0
        self.obs[0].copy_(obs.squeeze())
        self.masks[0].copy_(torch.zeros(1))

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages):
        mini_batch_size = self.num_steps // self.mini_batch_num
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps)), mini_batch_size, drop_last=False)
        for indices in sampler:
            next_indices = [indice + 1 for indice in indices]
            obs_batch = self.obs[indices]
            next_obs_batch = self.obs[next_indices]
            action_batch = self.action[indices]
            value_pred_batch = self.value_preds[indices]
            return_batch = self.returns[indices]
            old_action_log_probs_batch = self.action_log_probs[indices]
            advantages_batch = advantages[indices]
            masks_batch = self.masks[indices]
            yield obs_batch, next_obs_batch, action_batch, value_pred_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, advantages_batch
