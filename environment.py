import numpy as np
import torch


class Env(object):
    def __init__(self, user_num, state_dim, device, env_args):
        self.device = device

        self.user_num = user_num
        self.action_dim = user_num
        self.state_dim = state_dim

        relationship = env_args.V['relationship']
        unit_cost = env_args.V['cost']

        self.battery_budget = env_args.V['user_battery_budget']
        self.task_num = env_args.V['task_num']
        self.task_budget = env_args.V['R']
        prob = env_args.V['prob']
        self.prob = np.zeros(self.task_num)
        for i in range(self.task_num):
            self.prob[i] = prob[i]

        self.unit_cost = np.zeros(user_num)
        for i in range(user_num):
            self.unit_cost[i] = unit_cost[i]

        self.remain_energy = np.zeros(user_num)
        for i in range(user_num):
            self.remain_energy[i] = self.battery_budget[i]

        self.relationship = np.zeros((user_num, user_num))
        for i in range(user_num):
            for j in range(user_num):
                self.relationship[i][j] = relationship[i][j]

        self.server_reward = 0
        self.total_server_reward = []
        self.beta = np.zeros(user_num)
        beta = env_args.V['quality']
        for i in range(self.task_num):
            self.beta[i] = beta[i]

        self.R = 0
        self.task_index = 0
        self.epoch = 0
        self.total_obtain_sensing_data = 0
        # self.max_completion_ratio = 0

        self.complete_task = 0
        self.total_task = 0

        self.task_cnt = np.zeros(self.task_num)
        self.obtain_sensing_data = np.zeros(self.task_num)

        self.final_contrib_data = 0

        self.intrinsic_reward = 0
        self.extrinsic_reward = 0

    def get_collected_data(self):
        return self.final_contrib_data

    def close(self):
        return None

    def plot_server_reward(self, episode):
        server_reward = self.server_reward
        self.server_reward = 0
        return server_reward

    def plot_complete_ratio(self, episode):
        obtain_sensing_data_list = []
        for i in range(self.task_num):
            obtain_sensing_data = self.obtain_sensing_data[i] / self.task_cnt[i]
            obtain_sensing_data_list.append(obtain_sensing_data)
        self.total_obtain_sensing_data = 0
        self.epoch = 0
        self.obtain_sensing_data = np.zeros(self.task_num)
        self.task_cnt = np.zeros(self.task_num)

    def reset(self):
        for i in range(self.user_num):
            self.remain_energy[i] = self.battery_budget[i]

        self.task_index = np.random.choice(self.task_num, 1, False, self.prob)[0]
        # self.task_index = 0
        self.R = self.task_budget[self.task_index]

        state = np.zeros((self.user_num, self.state_dim))
        for i in range(self.user_num):
            state[i, self.user_num:self.user_num + 1] = self.unit_cost[i] / 10
            state[i, self.user_num + 1:self.user_num + 2] = self.remain_energy[i] / 50
            state[i, self.user_num + 2:self.user_num + 3] = self.R / 10

        return torch.from_numpy(state).float().to(self.device)

    def get_completion_ratio(self):
        completion_ratio = self.complete_task / self.total_task
        self.complete_task = 0
        self.total_task = 0
        return completion_ratio

    def get_reward(self):
        extrinsic_reward = self.extrinsic_reward / self.user_num
        intrinsic_reward = self.intrinsic_reward / self.user_num
        self.extrinsic_reward = 0
        self.intrinsic_reward = 0
        return extrinsic_reward, intrinsic_reward

    def step(self, action):
        action = 0.2 * action.float().numpy()

        # -------standard action----------------------
        for i in range(self.user_num):
            if action[i] > self.remain_energy[i] / self.unit_cost[i]:
                action[i] = self.remain_energy[i] / self.unit_cost[i]
        phi = np.zeros(self.user_num, 'float')

        for i in range(self.user_num):
            for j in range(self.user_num):
                phi[i] += self.relationship[i][j] * action[i] * action[j]
        total_sensing = action.sum()

        sensing_data = total_sensing
        self.total_obtain_sensing_data += sensing_data
        self.task_cnt[self.task_index] += 1
        self.obtain_sensing_data[self.task_index] += sensing_data
        self.final_contrib_data += total_sensing / self.R
        self.epoch += 1

        quality_sensing_data = action
        total_quality_sensing_data = np.sum(quality_sensing_data)
        # print(np.shape(action))
        reward = np.zeros(self.user_num)
        self.server_reward += total_quality_sensing_data * self.beta[self.task_index] - self.R
        intrinsic_reward = 0
        extrinsic_reward = 0
        self.total_task += 1
        if total_sensing > 0.001:
            self.complete_task += 1
            for i in range(self.user_num):
                # reward[i] = action[i] / total_sensing * self.R - self.unit_cost[i] * action[i] + phi[i]
                reward[i] = quality_sensing_data[i] / total_quality_sensing_data * self.R - self.unit_cost[i] * action[
                    i] + phi[i]
                extrinsic_reward += quality_sensing_data[i] / total_quality_sensing_data * self.R - self.unit_cost[i] * \
                                    action[i]
                intrinsic_reward += phi[i]
                self.remain_energy[i] -= self.unit_cost[i] * action[i]
                if self.remain_energy[i] <= 0.0001:
                    self.remain_energy[i] = 0
        self.intrinsic_reward += intrinsic_reward
        self.extrinsic_reward += extrinsic_reward
        self.task_index = np.random.choice(self.task_num, 1, False, self.prob)[0]
        # self.task_index = 0
        self.R = self.task_budget[self.task_index]

        state = np.zeros((self.user_num, self.state_dim))
        for i in range(self.user_num):
            state[i, 0:self.user_num] = action
            state[i, self.user_num:self.user_num + 1] = self.unit_cost[i] / 10
            state[i, self.user_num + 1:self.user_num + 2] = self.remain_energy[i] / 50
            state[i, self.user_num + 2:self.user_num + 3] = self.R / 10
        # reward = np.mean(reward, keepdims=True)

        done = False

        return torch.from_numpy(state).float().to(self.device), torch.from_numpy(reward).float(), done
