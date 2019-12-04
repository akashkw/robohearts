import torch
import numpy as np
import torch.nn.functional as F
from .agent_utils import *
import torch.utils.tensorboard as tb
from os import path


class MLPClassifier(torch.nn.Module):
    def __init__(self, n_input_features=108, hidden_nodes=256, n_output_Features=1, n_layers=2, log=False, log_dir='./log'):
        super().__init__()

        if n_layers != 2:
            print("More or less than 2 layers is not supported, so using 2")


        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_input_features, hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes, 1),
        )

        self.logger = tb.SummaryWriter(path.join(log_dir, 'train'), flush_secs=1)
        self.log = log
        self.global_step = 0

    def forward(self, x):
        return self.network(x)

def update(nn, optimizer, device, G, features):
    val = nn(features)
    ret = torch.tensor([G]).to(device).double()
    optimizer.zero_grad()
    loss = F.mse_loss(val, ret)
    loss.backward()
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
    if model is not '':
        print("loaded from " + str(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model)))
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r


#-------------- FEATURE CALCULATIONS --------------
def cards_to_bin_features(cards):
    deck = deck_reference()
    feature_vec = np.zeros(52)
    for card in hand:
        feature_vec[deck[card]] = 1
    return feature_vec 

def in_hand_features(observation):
    return cards_to_bin_features(observation['data']['hand'])

def in_play_features(play_cards):
    return cards_to_bin_features(observation['data']['currentTrick'])

def played_cards_features(played_cards):
    return cards_to_bin_features(played_cards)

def won_cards_features(won_cards):
    point_cards = pts_reference()
    feature_vec = np.zeros((4, 14))
    for player, won_card in enumerate(won_cards):
        for card in won_card:
            if card in point_cards:
                feature_vec[player][point_cards[card]] = 1
    return feature_vec.flatten()

def scores_features(scores):
    return np.array(scores)

'''
 Need data for feature construction - played cards and won cards can be built from TrickEnd event
 in the primary program driver (note, played_cards could be a list, won_cards could be list of lists
 or a dictionary (lists of lists/2d array seems easier).  Probably cleanest sol is both are np
 arrays we update at the end of tricks in the MC agent (ie MC keeps the state).  Scores just stored
 in a list easily. 
'''
def get_features(observation, feature_list=['in_hand'], played_cards=None, won_cards=None, scores=None):
    features = np.array([0])

    if 'in_hand' in feature_list:
        features = np.concatenate([features, in_hand_features(observation)])
    if 'in_play' in feature_list:
        features = np.concatenate([features, in_play_features(observation)])
    if 'played_cards' in feature_list:
        features = np.concatenate([features, played_cards_features(played_cards)])
    if 'won_cards' in feature_list:
        features = np.concatenate([features, won_cards_features(won_cards)])
    if 'scores' in feature_list:
        features = np.concatenate([features, scores_features(scores)])

    return features
