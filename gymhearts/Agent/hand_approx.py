import torch
import numpy as np
import torch.nn.functional as F
from agent_utils import *

class MLPClassifier(torch.nn.Module):
    def __init__(self, n_input_features=52, n_output_Features=1, n_layers=2):
        super().__init__()

        """
        Your code here
        """
        if n_layers != 2:
            print("More or less than 2 layers is not supported, so using 2")

        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_input_features, 52),
            torch.nn.ReLU(),
            torch.nn.Linear(52, 1),
        )

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B, n_input_features))
        @return: torch.Tensor((B, 1))
        """
        return self.network(x)


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
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r

# Return the features corresponding to a hand
def inhand_features(hand):
    deck = deck_reference()
    feature_vec = np.zeros(52)
    for card in hand:
        feature_vec[deck[card]] = 1
    return feature_vec 


# TODO:  Add function to build:
# - Cards in play this trick -- 52 length vector, only up to 4 active
# - Cards previously played in this round --- gives history of cards, 52 length vector
# - hearts + q spade each player has won --- length 14 vector for each player, 56 length vector
# - score for each player (may inform strategy?) -- 4 length vector
# - function to append the approx for each one of these features (4 *52 + 4 + 4 for:
 #   cards in hand, cards played, hearts won by each player,  cards this trick, + Qs each player + score per player)