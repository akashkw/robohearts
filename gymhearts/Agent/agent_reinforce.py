import numpy as np
import random
import torch

from .utils_env import *
from .utils_nn import *

class REINFORCE_Agent:
    def __init__(self, name, params=dict()):
        # Game Params
        self.name = name
        self.print_info = params.get('print_info', False)

        # Agent Params
        self.EPSILON = params.get('epsilon', .01)
        self.GAMMA = params.get('gamma', .95)
        # Note - needed low Alpha to get this working!
        self.ALPHA = params.get('alpha', 3e-6)

        # fn approx items:
        self.FT_LIST = params.get('feature_list', ['in_hand'])
        self.p_idx = []
        self.players = []
        # Scores of each player
        self.scores = [0 for i in range(4)]
        self.played_cards=[]
        # Keeps track of the cards won by each of the four players
        self.won_cards=[list() for i in range(4)]

        # NN params
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Overwrite -> comment out if not needed
        self.device = torch.device('cpu')

        # Load model if desired
        model_name = params.get('load_model', '')
        if model_name:
            self.nn = load_model(model_name, 'mc_nn', self.FT_LIST).to(self.device)
            self.pi, self.baseline = load_model(model_name, 'reinforce', self.FT_LIST)
        else:
            self.pi = PiApproximationWithNN(feature_length(self.FT_LIST), params)
            self.baseline = VApproximationWithNN(feature_length(self.FT_LIST), params)

        self.deck = create_deck()
        self.deck_reference = deck_reference()

    def Do_Action(self, observation):

        if observation['event_name'] == 'GameStart':
            # Create a list of player names
            self.players = [entry['playerName'] for entry in observation['data']['players']]

        elif observation['event_name'] == 'PassCards':
            if self.print_info:
                print(handle_event(observation))
            passCards = random.sample(observation['data']['hand'],3)
            
            if self.print_info:
                print(self.name, 'is passing ::', " ".join([pretty_card(card) for card in passCards]))
                
            return {
                    "event_name" : "PassCards_Action",
                    "data" : {
                        'playerName': self.name,
                        'action': {'passCards': passCards}
                    }
                }

        elif observation['event_name'] == 'PlayTrick':
            if self.print_info:
                print(handle_event(observation))

            hand = observation['data']['hand']
            if '2c' in hand:
                choose_card = '2c'
            else:
                action = self.epsilon_greedy_selection(observation)
                choose_card = filter_valid_moves(observation)[action]
                if self.print_info:
                    print(self.name, 'chose card ::', pretty_card(choose_card))

            return {
                    "event_name" : "PlayTrick_Action",
                    "data" : {
                        'playerName': self.name,
                        'action': {'card': choose_card}
                    }
                }

        elif observation['event_name'] == 'NewRound':
            # Init scores for all players
            self.scores = [entry['score'] for entry in observation['data']['players']]

        elif observation['event_name'] == 'ShowTrickEnd':
            # Record the cards won in a trick, add to played_cards history
            winner = self.players.index(observation['data']['trickWinner'])
            for card in observation['data']['cards']:
                self.won_cards[winner].append(card)
                self.played_cards.append(card)

        elif observation['event_name'] == 'RoundEnd':
            # Reset for the next round
            self.played_cards = []
            self.won_cards = [list() for i in range(4)]

    # Update using the REINFORCE algorithm, with episodes in batches
    def update(self, batch):
        G_list = []
        for episode_history, G in batch:
            for i, history in enumerate(reversed(episode_history)):
                t = len(history) - i - 1
                state_features, valid_features, action = history
                state_features = torch.Tensor(state_features).float().to(self.device)
                delta = G - self.baseline(state_features)
                self.baseline.update(state_features, G)
                self.pi.reinforce_update(state_features, action, (self.GAMMA**t), delta, valid_features)
                G *= self.GAMMA
            G_list.append(G)
        return np.array(G_list).mean()

    # Select an action using epsilon-greedy action selection
    def epsilon_greedy_selection(self, observation):
        rand = np.random.uniform(0,1)
        if rand < self.EPSILON:
            return np.random.randint(0, len(filter_valid_moves(observation)))
        else:
            return self.greedy_action(observation)

    # Select an action using epsilon-greedy action selection
    def greedy_action(self, observation):
        state_features, valid_features = self.generate_features(observation)
        action = self.pi(state_features, valid_features)
        return filter_valid_moves(observation).index(self.deck[action])

    def generate_features(self, observation):
        state_features = get_features(observation, feature_list=self.FT_LIST, 
            played_cards=self.played_cards, won_cards=self.won_cards, scores=self.scores)
        hand = observation['data']['hand']
        valid_features = cards_to_valid_bin_features(hand)
        return state_features, valid_features