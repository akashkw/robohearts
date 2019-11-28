import torch
import numpy as np
import torch.nn.functional as F
from .agent_utils import *
import torch.utils.tensorboard as tb
from os import path


class MLPClassifier(torch.nn.Module):
    def __init__(self, n_input_features=52, hidden_nodes=256, n_output_Features=1, n_layers=2, log=False):
        super().__init__()

        """
        Your code here
        """
        if n_layers != 2:
            print("More or less than 2 layers is not supported, so using 2")


        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_input_features, hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes, 1),
        )

        LOG_DIR = '/content/robohearts/log'
        self.logger = tb.SummaryWriter(path.join(LOG_DIR, 'train'), flush_secs=1)
        self.log = log
        self.global_step = 0

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B, n_input_features))
        @return: torch.Tensor((B, 1))
        """
        return self.network(x)

def update(nn, optimizer, alpha, G, hand, device):
    val = nn(torch.tensor(inhand_features(hand)).to(device).double())
    returns = torch.tensor([G]).to(device).double()
    optimizer.zero_grad()
    loss = F.mse_loss(val, returns)
    (alpha * .5 * loss).backward()
    optimizer.step()

    if nn.log and nn.global_step % 1000 == 0:
        nn.logger.add_scalar('loss', loss, nn.global_step)
    nn.global_step += 1


model_factory = {
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory['mlp']()
    if not model and model is not "":
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r


#-------------- FEATURE CALCULATIONS --------------
def inhand_features(hand):
    deck = deck_reference()
    feature_vec = np.zeros(52)
    for card in hand:
        feature_vec[deck[card]] = 1
    return feature_vec 

def inplay_features(play_cards):
    deck = deck_reference()
    feature_vec = np.zeros(52)
    for card in play_cards:
        feature_vec[deck[card]] = 1
    return feature_vec

def played_features(played_cards):
    deck = deck_reference()
    feature_vec = np.zeros(52)
    for card in played_cards:
        feature_vec[deck[card]] = 1
    return feature_vec

def won_features(player_won):
    winnable_pts = pts_reference()
    feature_vec = np.zeros(4, 13)
    for i, player_cards_won in enumerate(player_won):
        for card in player_cards_won:
            feature_vec[i][winnable_pts[card]] = 1
    return feature_vec.flatten()

def get_score_feature(scores):
    feature_vec = np.zeros(4)
    for i, score in enumerate(scores):
        feature_vec[i] = score
    return feature_vec


def get_features(observation, feature_list=[]):
    features = np.inhand_features()
    if 'in_play' in feature_list:
        features.stack(inplay_features(observation['play']))
    if 'played_cards' in feature_list:
        features.stack(played_features(observation['play']))
    if 'cards_won' in feature_list:
        features.stack(won_features(observation['won']))
    if 'scores' in feature_list:
        features.stack(get_score_feature(observation['scores']))
    return features
