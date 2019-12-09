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

        self.pi = PiApproximationWithNN(feature_length(self.FT_LIST), 52, self.ALPHA, pi_path)
        self.baseline = VApproximationWithNN(feature_length(self.FT_LIST), self.ALPHA, baseline_path)

        self.deck = create_deck()

    def Do_Action(self, observation):

        if observation['event_name'] == 'Game   Start':
            players = observation['data']['players']
            for player in players:
                self.p_idx.append(player['playerName'])

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
            players = observation['data']['players']
            for i, player in enumerate(players):
                self.scores[i] = player['score']

        elif observation['event_name'] == 'ShowTrickEnd':
            winner = self.p_idx.index(observation['data']['trickWinner'])
            for card in observation['data']['cards']:
                self.won_cards[winner].append(card)
                self.played_cards.append(card)

        elif observation['event_name'] == 'RoundEnd':
            self.played_cards = []
            self.won_cards = []
            for i in range(4):
                self.won_cards.append([])

    # Update using the REINFORCE algorithm, with episodes in patches
    def update(self, batch):
        gs = []
        for episode_info in batch:
            G = 0
            for i, result in enumerate(reversed(episode_info)):
                t = len(episode_info) - i - 1
                state, valid_filter, action, reward = result
                state =  torch.tensor(state).float().to(self.device)
                action = self.deck_ref[action['data']['action']['card']]

                G = G*self.GAMMA + reward
                delta = G - self.baseline(state)
                self.baseline.update(state, G)
                self.pi.update(state, action, (self.GAMMA**t), delta, valid_filter)
            gs.append(G)

        avg = 0
        for g in gs:
            avg += g / len(gs)
        return avg

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