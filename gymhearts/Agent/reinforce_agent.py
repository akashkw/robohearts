import numpy as np
import random
import torch

from .agent_utils import *
from .reinforce_utils import *

class REINFORCE_Agent:
    def __init__(self, name, params=dict()):
        # Game Params
        self.name = name
        self.print_info = params.get('print_info', False)

        # Agent Params
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
        pi_path = params.get('pi_saved', False)
        baseline_path = params.get('base_saved', False)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.pi = PiApproximationWithNN(feature_length(self.FT_LIST), 52, self.ALPHA, pi_path)
        self.baseline = VApproximationWithNN(feature_length(self.FT_LIST), self.ALPHA, baseline_path)
        self.deck_idx = {v: k for k, v in deck_reference().items()}
        self.deck_ref = deck_reference()

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
                action = self.action_select(observation)
                choose_card = self.deck_idx[action]
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
    def action_select(self, observation):
        ft =  torch.tensor(self.features_state(observation)).float().to(self.device)
        val_filter = self.get_filter(observation)
        return self.pi(ft, val_filter)

    def features_state(self, observation):
        return get_features(observation, feature_list=self.FT_LIST, 
            played_cards=self.played_cards, won_cards=self.won_cards, scores=self.scores)

    def get_filter(self, observation):
        val_moves = filter_valid_moves(observation)
        deck_ref = deck_reference()
        indices_valid = [deck_ref[card] for card in val_moves]
        val_filter = torch.zeros(52).float()
        for index in indices_valid:
            val_filter[index] = 1
        return val_filter

    def get_state_info(self, observation):
        return self.features_state(observation), self.get_filter(observation)