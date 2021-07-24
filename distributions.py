import torch.nn as nn
import torch
from utils import init
from torch.utils.data import WeightedRandomSampler
from torch.distributions import Categorical


class _Categorical(Categorical):
    """
    a son class inherit from class torch.distributions.Categorical
    it adds a gumbel softmax sample method, for gumbel softmax sample
    and a mode method for argmax sample
    """

    def __init__(self, _logits):
        super(_Categorical, self).__init__(logits=_logits)
        self._logits = self.logits
        self.weighted_sampler = WeightedRandomSampler

    def gumbel_softmax_sample(self, tau, device):
        dist = F.gumbel_softmax(self._logits, tau=tau, hard=False)
        action = torch.tensor(list(self.weighted_sampler(dist, 1, replacement=False))).to(device)
        return action.squeeze(-1)

    def mode(self):
        return torch.argmax(self._logits, dim=-1, keepdim=False)


class MultiHeadCategorical(nn.Module):
    """
    define a multi-head Categorical for multi-label classification
    --init:
    num_inputs: input feature dim
    dim_vec: a list for dim of each action space, e.g. [2,3,5], 2-dim for action1, 3-dim for action2, 5-dim for action3
    device: running device
    --forward:
    inputs: flatten input feature
    """

    # @torchsnooper.snoop()
    def __init__(self, num_inputs, action_num, action_dim, device):
        super(MultiHeadCategorical, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)
        self.action_num = action_dim
        self.linear_list = torch.nn.ModuleList(
            [init_(nn.Linear(num_inputs, action_dim).to(device)) for _ in range(action_num)])
        self.action_num = action_num
        self.logits_head = []
        self.weight_sample = WeightedRandomSampler
        self.device = device
        self.categorical_list = []
        self.train()

    def forward(self, inputs):
        self.categorical_list = [_Categorical(linear(inputs)) for linear in self.linear_list]

    def gumbel_softmax_sample(self, tau):
        action = torch.cat([p.gumbel_softmax_sample(tau, self.device) for p in self.categorical_list])
        return action

    @property
    def probs(self):
        if self.action_num == 1:
            return self.categorical_list[0].probs
        else:
            return torch.cat([p.probs.unsqueeze(-1) for p in self.categorical_list], dim=-1)

    def log_probs(self, action):
        if self.action_num == 1:
            return self.categorical_list[0].log_prob(action)
        else:
            return torch.cat([p.log_prob(a).unsqueeze(-1) for a, p in zip(action, self.categorical_list)], dim=-1)

    def mode(self):
        if self.action_num == 1:
            return self.categorical_list[0].mode()
        else:
            return torch.cat([p.mode() for p in self.categorical_list])

    def sample(self):
        if self.action_num == 1:
            return self.categorical_list[0].sample()
        else:
            return torch.cat([p.sample() for p in self.categorical_list])

    def entropy(self):
        if self.action_num == 1:
            return self.categorical_list[0].entropy()
        else:
            return torch.cat([p.entropy() for p in self.categorical_list])
