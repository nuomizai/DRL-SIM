from distributions import MultiHeadCategorical
import torch
from utils import init
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, state_dim, action_dim, device, trainable=True, hidsize=128):
        super(Model, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        # feature extract
        self.base = nn.Sequential(
            init_(nn.Linear(state_dim, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, hidsize)),
            nn.ReLU()
        ).to(device)

        # actor
        self.dist = MultiHeadCategorical(hidsize, 1, action_dim, device)
        # # critic
        # self.critic = nn.Sequential(
        #     init_(nn.Linear(hidsize, 1))
        # ).to(device)
        # critic
        self.q_network = nn.Sequential(
            init_(nn.Linear(hidsize, action_dim)),
        ).to(device)
        self.device = device
        self.identity = torch.eye(action_dim).to(device)
        if trainable:
            self.train()
        else:
            self.eval()

    # @torchsnooper.snoop()
    def act(self, inputs):
        with torch.no_grad():
            obs_feature = self.base(inputs)

            # value = self.critic(obs_feature)
            self.dist(obs_feature)
            action = self.dist.sample()
            action_log_probs = self.dist.log_probs(action)
            action_log_probs = action_log_probs.mean(-1, keepdim=True)

            q_value = self.q_network(obs_feature)
            # mean
            value = torch.sum(self.dist.probs * q_value, -1, keepdim=True)
        return value, action.squeeze(), action_log_probs

    def get_value(self, inputs):
        obs_feature = self.base(inputs)
        # value = self.critic(obs_feature)
        self.dist(obs_feature)
        q_value = self.q_network(obs_feature)
        value = torch.sum(self.dist.probs * q_value, -1, keepdim=True)
        return value

    def evaluate_actions(self, inputs, action):
        obs_features = self.base(inputs)
        # value = self.critic(obs_features)
        q_value = self.q_network(obs_features)
        index = self.identity[action.squeeze(-1)]
        value = torch.sum(q_value * index, -1).unsqueeze(-1)

        self.dist(obs_features)

        action_log_probs = self.dist.log_probs(action).mean(-1, keepdim=True)

        dist_entropy = self.dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    def print_grad(self):
        for name, p in self.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)


