from typing import Iterable
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha,
                 saved=None):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here
        self.nn = MLPClassifier(input_features=state_dims, output_features=num_actions)
        if saved is not None:
            self.nn.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'pi.th'), map_location='cpu'))
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=alpha)

    def __call__(self,s, valid_filter) -> int:
        # Sample an action according to the policy
        logits = self.nn(torch.FloatTensor(s))
        logit_fiter = logits * valid_filter
        return sample_categorical(logits=logit_fiter)
       

    def update(self, s, a, gamma_t, delta, valid_filter):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.optim.zero_grad()
        p = Categorical(self.nn(torch.FloatTensor(s)) * valid_filter)
        l = p.log_prob(torch.tensor(a))
        (-l*delta*gamma_t).backward()
        self.optim.step()

    def save_pi_model():
        from torch import save
        from os import path
        return save(self.nn.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'pi.th'))

def sample_categorical(out):
    return Categorical(probs=out).sample().item()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_features, output_features=1, layers=None, log=False, log_dir='./log'):
        super().__init__()
        if not layers:
            layers = [input_features * 2, input_features * 4]

        L = []
        c = input_features
        for l in layers:
            L.append(torch.nn.Linear(c, l))
            L.append(torch.nn.ReLU())
            c = l
        L.append(torch.nn.Linear(c, output_features))

        self.network = torch.nn.Sequential(*L)

        self.global_step = 0

        self.log = log
        if self.log:
            self.logger = tb.SummaryWriter(path.join(log_dir, 'train'), flush_secs=1)


    def forward(self, x):
        return self.network(x)

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha,
                 saved=None):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.nn = MLPClassifier(n_input_features=state_dims)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=alpha)
        if saved is not None:
            self.nn.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'baseline.th'), map_location='cpu'))

    def __call__(self,s):
        return self.nn(torch.FloatTensor(s)).detach().item()

    def update(self, s, G):
        val = self.nn(torch.FloatTensor(s))
        returns = torch.tensor([G])
        self.optimizer.zero_grad()
        (.5 * F.mse_loss(val, returns)).backward()
        self.optimizer.step()

    def save_base_model():
        from torch import save
        from os import path
        return save(self.nn.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'baseline.th'))
