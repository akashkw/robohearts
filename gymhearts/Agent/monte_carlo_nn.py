import random
import numpy as np
from datetime import datetime
import torch
import copy

from .agent_utils import *
from .hand_approx import *

class MonteCarloNN:
    def __init__(self, name, params=dict()):
        # Game Params
        self.name = name
        self.print_info = params.get('print_info', False)

        # Agent Params
        self.EPSILON = params.get('epsilon', .05)
        self.GAMMA = params.get('gamma', .95)
        self.ALPHA = params.get('alpha', .1)

        # NN params
        path = params.get('nn_path', '')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.nn = load_model(path).double().to(self.device)

        # optimizer params
        lr = params.get('lr', 1e-3)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=lr)

        # fn approx items:
        self.FT_LIST = []
        self.p_idx = []
        self.scores = [0]*4
        self.played_cards=[]
        self.won_cards=[]
        for i in range(4):
            self.won_cards.append([])

        if params.get('in_hand', True):
            self.FT_LIST.append('in_hand')
        if params.get('in_play', False):
            self.FT_LIST.append('in_play')
        if params.get('played_cards', False):
            self.FT_LIST.append('played_cards')
        if params.get('cards_won', False):    
            self.FT_LIST.append('cards_won')
        if params.get('scores', False):    
            self.FT_LIST.append('scores')

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


    def update_weights(self, history, ret):
        for observation, played_cards, won_cards in reversed(history):

            ft = get_features(observation, feature_list=self.FT_LIST, 
                played_cards=played_cards, won_cards=won_cards, scores=self.scores)
            features = torch.tensor(ft).to(self.device)

            update(self.nn, self.optim, self.device, self.ALPHA, ret, features)
            ret *= self.GAMMA
        return

    # Select an action using epsilon-greedy action selection
    def epsilon_greedy_selection(self, observation):
        rand = np.random.uniform(0,1)
        if rand < self.EPSILON:
            return np.random.randint(0, len(filter_valid_moves(observation)))
        else:
            return self.greedy_action(observation)

    # Return the value of a hand
    def value(self, observation):
        ft = get_features(observation, feature_list=self.FT_LIST, 
            played_cards=self.played_cards, won_cards=self.won_cards, scores=self.scores)

        features = torch.tensor(ft).to(self.device)
        return self.nn(features).detach().item()

    # Perform a one-step lookahead and select the action that has the best expected value
    def greedy_action(self, observation):
        hand = observation['data']['hand']
        obs_prime = copy.deepcopy(observation)

        valid_moves = filter_valid_moves(observation)
        best_move, best_succ_val = None, float('-inf')
        for move, card in enumerate(valid_moves):

            # set up observation for next state, after play
            succ_hand = [c for c in hand if c != card]
            obs_prime['data']['hand'] = succ_hand
            obs_prime['data']['currentTrick'][0]['card'] = card
            succ_val = self.value(obs_prime)
            if succ_val > best_succ_val:
                best_move, best_succ_val = move, succ_val
        return best_move