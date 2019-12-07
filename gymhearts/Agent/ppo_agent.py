import random
import numpy as np
from datetime import datetime
import torch
import copy

from .agent_utils import *
from .reinforce import *

class PPO_Agent:
    def __init__(self, name, params=dict()):
        # Game Params
        self.name = name
        self.print_info = params.get('print_info', False)

        # Agent Params
        self.GAMMA = params.get('gamma', .95)
        self.ALPHA = params.get('alpha', 3e-4)

        # NN params
        pi_path = params.get('pi_saved', False)
        baseline_path = params.get('base_saved', False)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.pi = PiApproximationWithNN(52, 52, self.ALPHA, pi_path)
        self.baseline = VApproximationWithNN(52, self.ALPHA, baseline_path)

        # optimizer params
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=self.ALPHA)

        # fn approx items:
        self.FT_LIST = params.get('feature_list', ['in_hand'])
        self.p_idx = []
        self.players = []
        # Scores of each player
        self.scores = [0 for i in range(4)]
        self.played_cards=[]
        # Keeps track of the cards won by each of the four players
        self.won_cards=[list() for i in range(4)]


    def Do_Action(self, observation):

        if observation['event_name'] == 'GameStart':
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
                choose_card = deck_reference()[action]
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

    def update_weights(self, batch_info):
        G = 0
        for i, result in enumerate(reversed(episode_info)):
            t = len(episode_info) - i - 1
            state, filter, action, reward = result
            G = G*self.GAMMA + reward
            delta = G - self.baseline(state)
            self.baseline.update(state, G)
            self.pi.update(state, action, (self.GAMMA**t), delta, filter)

    # Select an action using epsilon-greedy action selection
    def action_select(self, observation):
        ft =  torch.tensor(features_state(observation)).to(self.device)
        filter = self.get_filter(observation)
        return self.pi(features, filter).detach().item()

    def features_state(self, observation):
        return get_features(observation, feature_list=self.FT_LIST, 
            played_cards=self.played_cards, won_cards=self.won_cards, scores=self.scores)

    def get_filter(self, observation):
        val_moves = filter_valid_moves(observation)
        deck_ref = deck_reference()
        indices_valid = [deck_ref.index(card) for card in val_moves]
        filter = torch.zeros(52)
        for index in indices_valid:
            filter[index] = 1
        return filter

    def get_state_info(self, observation):
        return self.features_state(observation), self.get_filter(observation)