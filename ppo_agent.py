import torch
from storage import RolloutStorage
from model import Model
import torch.optim as optim


class PPOAgent():
    def __init__(self, state_dim, device, lr, exploration_steps, mini_batch_num, use_gae, gamma,
                 gae_param, ppo_epoch, clip, value_coeff, clip_coeff, ent_coeff):
        self.local_ppo_model = Model(state_dim, 6, device)
        self.optimizer = optim.Adam(list(self.local_ppo_model.parameters()), lr=lr)
        self.rollout = RolloutStorage(exploration_steps, mini_batch_num, state_dim)
        self.rollout.to(device)
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_param = gae_param
        self.ppo_epoch = ppo_epoch
        self.clip = clip
        self.value_coeff = value_coeff
        self.clip_coeff = clip_coeff
        self.ent_coeff = ent_coeff

    def act(self, obs):
        value, action, action_log_probs = self.local_ppo_model.act(obs)
        return value, action, action_log_probs

    def insert(self, obs, action, action_log_probs, value, reward, masks):
        self.rollout.insert(obs, action, action_log_probs, value, reward, masks)

    def after_update(self, obs):
        self.rollout.after_update(obs)

    def load_model(self, path, device, test_device):
        self.local_ppo_model.load_state_dict(torch.load(path, map_location={device: test_device}))

    def reset(self, obs):
        self.rollout.reset(obs)

    def update(self, done):
        beta = 0.2
        with torch.no_grad():
            if done:
                next_value = torch.zeros(1)
            else:
                next_value = self.local_ppo_model.get_value(self.rollout.obs[-1:])

        self.rollout.compute_returns(next_value.detach(), self.use_gae, self.gamma, self.gae_param)

        advantages = self.rollout.returns[:-1] - self.rollout.value_preds[:-1]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        av_value_loss = 0
        av_policy_loss = 0
        av_ent_loss = 0
        loss_cnt = 0

        for _ in range(self.ppo_epoch):
            data_generator = self.rollout.feed_forward_generator(advantages)
            for samples in data_generator:
                # signal_init = traffic_light.get()
                torch.cuda.empty_cache()
                obs_batch, next_obs_batch, action_batch, old_values, return_batch, masks_batch, \
                old_action_log_probs, advantages_batch = samples

                cur_values, cur_action_log_probs, dist_entropy = self.local_ppo_model.evaluate_actions(obs_batch,
                                                                                                       action_batch)

                # ----------use ppo clip to compute loss------------------------
                ratio = torch.exp(cur_action_log_probs - old_action_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages_batch

                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_values + (cur_values - old_values).clamp(-self.clip, self.clip)
                value_losses = (cur_values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                # value_loss = torch.mean((return_batch - cur_values)**2)

                value_loss = value_loss * self.value_coeff
                action_loss = action_loss * self.clip_coeff
                ent_loss = dist_entropy * self.ent_coeff
                # ------------------ for curiosity driven--------------------------
                total_loss = value_loss + action_loss - ent_loss
                self.local_ppo_model.zero_grad()
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                av_value_loss += float(value_loss)
                av_policy_loss += float(action_loss)
                av_ent_loss += float(ent_loss)
                loss_cnt += 1

        return av_value_loss / loss_cnt, av_policy_loss / loss_cnt, av_ent_loss / loss_cnt
