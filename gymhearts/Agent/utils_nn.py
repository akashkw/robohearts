import math
import pickle
import numpy as np
import torch
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
    loss = torch.nn.MSELoss()(val, ret)
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
        self.softmax = torch.nn.Softmax()

    def __call__(self, state_features, valid_features) -> int:
        probs = self.action_probs(state_features, valid_features)
        # Randomly select according to probs
        return self.sample_categorical(probs)
    
    def action_probs(self, state_features, valid_features):
        self.nn.eval()
        # Find preferences of all actions
        prefs = self.nn(torch.Tensor(state_features).float())
        # Only consider preferences of valid actions
        filtered_prefs = prefs * valid_features
        # Softmax to get probabilites of selection
        probs = self.softmax(filtered_prefs)

        # return first card we can, weird edge case
        if math.isnan(probs.sum().item()) or probs.sum() == 0:
            print("WARNING: Numeric instability")
            for i in range(len(valid_features)):
                if valid_features[i] == 1:
                    return i
        return probs

    def reinforce_update(self, state_features, action, gamma_t, delta, valid_features):
        probs = self.action_probs(state_features, valid_features)
        self.nn.train() 
        if math.isnan(probs.sum().item()) or probs.sum() == 0:
            print("WARNING: Numeric instability")
            return
        else: 
            p = Categorical(probs=probs)
            log_prob = p.log_prob(torch.Tensor([action]))
        loss = -(gamma_t*delta*log_prob)
        self.optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.nn.parameters(),.5)
        self.optim.step()

    def sample_categorical(self, probs):
        return Categorical(probs=probs).sample().item()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self, b):
        self.b = b

    def __call__(self, s) -> float:
        return self.b

    def update(self, s, G):
        pass


class VApproximationWithNN(Baseline):
    def __init__(self, input_features, output_features=1, params=dict()):
        self.nn = MLPClassifier(input_features, output_features)
        self.ALPHA = params.get('alpha', 3e-6)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=self.ALPHA, weight_decay=1e-4)
        self.device = torch.device('cpu')

    def __call__(self, state_features):
        self.nn.eval()
        return self.nn(torch.Tensor(state_features).float()).detach().item()

    def update(self, state_features, G):
        mlp_classifier_update(self.nn, self.optim, self.device, G, state_features)


# ----------------- MODEL UTILS ------------------

def save_model(value_model, model_name, model_type, pi_model=None):
    from torch import save
    from os import path
    if model_type == "mc_nn":
        save(value_model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}.th'))
    elif model_type == 'mc_simple':
        filename = path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}.th')
        with open(filename, 'wb') as file:
            pickle.dump(value_model, file)
    elif model_type == 'reinforce':
        save(pi_model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}_pi.th'))
        save(value_model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}_v.th'))
    else:
        if model_type == '':
            raise ValueError(f"model_type is blank!")
        else:
            raise ValueError(f"model_type '{model_type}' is not supported!")

def load_model(model_name, model_type, feature_list=None):
    from torch import load
    from os import path
    if model_type == "mc_nn":
        model = MLPClassifier(feature_length(feature_list))
        model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}.th'), map_location='cpu'))
        return model
    elif model_type == 'mc_simple':
        filename = path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}.th')
        weights = []
        with open(filename, 'rb') as file:
            weights = pickle.load(file)
        return weights
    elif model_type == 'reinforce':
        pi = PiApproximationWithNN(feature_length(feature_list))
        v = VApproximationWithNN(feature_length(feature_list))
        pi.nn.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}_pi.th'), map_location='cpu'))
        v.nn.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), f'models/{model_name}_v.th'), map_location='cpu'))
        return pi, v
    else:
        if model_type == '':
            raise ValueError(f"model_type is blank!")
        else:
            raise ValueError(f"model_type '{model_type}' is not supported!")
