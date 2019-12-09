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
    def __init__(self, input_features, output_features=52, params=dict()):

        self.nn = MLPClassifier(input_features, output_features)
        self.ALPHA = params.get('alpha', 3e-6)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=self.ALPHA, weight_decay=1e-4)

    def __call__(self, s, valid_filter) -> int:
        # Sample an action according to the policy
        out = F.softmax(self.nn(torch.Tensor(s).float()), dim=0) * valid_filter

        # return first card we can, weird edge case
        if math.isnan(out.sum().item()) or out.sum() == 0:
            print("WARNING: Numeric instability")
            for i in range(len(valid_filter)):
                if valid_filter[i] == 1:
                    return i

        probs = out / out.sum()
        return self.sample_categorical(probs)

    def reinforce_update(self, s, a, gamma_t, delta, valid_filter):
        self.optim.zero_grad()
        out = F.softmax(self.nn(s), dim=0)*valid_filter
        if math.isnan(out.sum().item()) or out.sum() == 0:
            print("WARNING: Numeric instability")
            return
        else: 
            probs = out / out.sum()
            p = Categorical(probs=probs)
            l = p.log_prob(torch.Tensor(a))
        (-l*delta*gamma_t).backward()
        # torch.nn.utils.clip_grad_norm_(self.nn.parameters(),.5)
        self.optim.step()

    def sample_categorical(self, out):
        return Categorical(probs=out).sample().item()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self, b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass


class VApproximationWithNN(Baseline):
    def __init__(self, input_features, output_features=1):
        self.nn = MLPClassifier(input_features, output_features).float()
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=alpha, weight_decay=1e-4)
        self.device = torch.device('cpu')

    def __call__(self, s):
        self.nn.eval()
        return self.nn(torch.Tensor(s).float()).detach().item()

    def update(self, s, G):
        mlp_classifier_update(self.nn, self.optim, self.device, G, get_features(s))


# ----------------- MODEL UTILS ------------------

def save_model(value_model, model_name, model_type, pi_model=None):
    from torch import save
    from os import path
    if model_type == "mlp":
        save(value_model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}.th'))
    elif model_type == 'simple':
        filename = path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}.th')
        with open(filename, 'wb') as file:
            pickle.dump(value_model, file)
    elif model_type == 'reinforce':
        save(value_model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}_v.th'))
        save(pi_model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}_pi.th'))
    else:
        if model_type == '':
            raise ValueError(f"model_type is blank!")
        else:
            raise ValueError(f"model_type '{model_type}' is not supported!")

def load_model(model_name, model_type, feature_list=None):
    from torch import load
    from os import path
    if model_type == "mc_nn":
        model = MLPClassifier(input_features=feature_length(feature_list))
        model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}.th'), map_location='cpu'))
        return model
    elif model_type == 'mc_simple':
        filename = path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}.th')
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
