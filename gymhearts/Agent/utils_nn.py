import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb

from os import path
from torch.distributions import Categorical
from torch import save, load
from .utils_env import *

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

# Perform an update to a mlp classifier
def mlp_classifier_update(nn, optimizer, device, G, features):
    nn.train()
    val = nn(features)
    ret = torch.Tensor([G]).to(device).float()
    optimizer.zero_grad()
    loss = F.mse_loss(val, ret)
    loss.backward()
    optimizer.step()

    if nn.log and nn.global_step % 1000 == 0:
        nn.logger.add_scalar('loss', loss, nn.global_step)
    nn.global_step += 1

class PiApproximationWithNN():
    def __init__(self, input_features, output_features, params=dict()):

        self.nn = MLPClassifier(input_features=input_features, output_features=output_features)
        self.ALPHA = params.get('alpha', 3e-6)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=alpha, weight_decay=1e-4)

    def __call__(self,s, valid_filter) -> int:
        # Sample an action according to the policy
        out = F.softmax(self.nn(torch.FloatTensor(s)), dim=0)*valid_filter
        # return first card we can, wierd edge case
        if math.isnan(out.sum().item()) or out.sum() == 0:
            print("WARNING: Numeric instability")
            for idx in range(len(valid_filter)):
                if valid_filter[idx] == 1:
                    return idx
        probs = out / out.sum()
        return sample_categorical(probs)
       

    def update(self, s, a, gamma_t, delta, valid_filter):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.optim.zero_grad()
        out = F.softmax(self.nn(s), dim=0)*valid_filter
        if math.isnan(out.sum().item()) or out.sum() == 0:
            print("WARNING: Numeric instability")
            return
        else: 
            probs = out / out.sum()
            p = Categorical(probs=probs)
            l = p.log_prob(torch.tensor(a))
        (-l*delta*gamma_t).backward()
        # torch.nn.utils.clip_grad_norm_(self.nn.parameters(),.5)
        self.optim.step()

    # Save the policy model to predetermined location
    def save_pi_model(self):
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


class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha,
                 saved=False):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.nn = MLPClassifier(state_dims).float()
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=alpha, weight_decay=1e-4)
        if saved:
            self.nn.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'baseline.th'), map_location='cpu'))
        self.nn = self.nn.train()
        self.device = torch.device('cpu')

    def __call__(self,s):
        return self.nn(torch.FloatTensor(s)).detach().item()

    def update(self, s, G):
        mc_update(self.nn, self.optim, self.device, G, get_features(s))


# ----------------- MODEL UTILS ------------------

def save_model(value_model, model_name, model_type, pi_model=None):
    from torch import save
    from os import path
    if model_type == "mlp":
        save(value_model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'{model_name}.th'))
    elif model_type == 'simple':
        filename = path.join(path.dirname(path.abspath(__file__)), f'{model_name}.th')
        with open(filename, 'wb') as file:
            pickle.dump(value_model, file)
    elif model_type == 'reinforce':
        save(value_model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'{model_name}_v.th'))
        save(pi_model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'{model_name}_pi.th'))
    else:
        if model_type == '':
            raise ValueError(f"model_type is blank!")
        else:
            raise ValueError(f"model_type '{model_type}' is not supported!")

def load_model(model_name, model_type, feature_list=None):
    from torch import load
    from os import path
    if model_type == "mlp":
        model = MLPClassifier(input_features=feature_length(feature_list))
        model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), f'{model_name}.th'), map_location='cpu'))
        return model
    elif model_type == 'simple':
        filename = path.join(path.dirname(path.abspath(__file__)), f'{model_name}.th')
        weights = []
        with open(filename, 'rb') as file:
            weights = pickle.load(file)
        return weights
    elif model_type == 'reinforce':
        pi = []
        return None
    else:
        if model_type == '':
            raise ValueError(f"model_type is blank!")
        else:
            raise ValueError(f"model_type '{model_type}' is not supported!")
