import random
import numpy as np
from datetime import datetime
import torch
import copy

from .agent_utils import *

class MonteCarloNN:
    def __init__(self, name, params=dict()):
        # Game Params
        self.name = name
        self.print_info = params.get('print_info', False)
        self.log = params.get('log', False)
        self.log_dir = params.get('log_dir', '/content/robohearts/log')

        # Agent Params
        self.EPSILON = params.get('epsilon', .05)
        self.GAMMA = params.get('gamma', .95)
        self.ALPHA = params.get('alpha', 1e-3)

        # fn approx items:

        # in_hand, in_play, played_cards, won_cards, scores
        self.FT_LIST = params.get('feature_list', ['in_hand'])
        # List of player names
        self.players = []
        # Scores of each player
        self.scores = [0 for i in range(4)]
        self.played_cards=[]
        # Keeps track of the cards won by each of the four players
        self.won_cards=[list() for i in range(4)]

        # NN params
        model_name = params.get('load_model', '')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Overwrite -> comment out if not needed
        self.device = torch.device('cpu')

        # Layers of the Neural Network
        self.layers = params.get('layers', [feature_length(self.FT_LIST)*2, feature_length(self.FT_LIST)*4])

        if model_name:
            self.nn = load_model(model_name, self.FT_LIST).to(self.device)
        else:
            self.nn = MLPClassifier(input_features=feature_length(self.FT_LIST), layers=self.layers, log=self.log).to(self.device)

        # optimizer params
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=self.ALPHA)

    def Do_Action(self, observation):

        if observation['event_name'] == 'GameStart':
            # Create a list of player names
            self.players = [entry['playerName'] for entry in observation['data']['players']]

        elif observation['event_name'] == 'PassCards':
            if self.print_info:
                print(handle_event(observation))
            # Randomly choose a card to pass
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


    def update_weights(self, history, ret):
        for observation, played_cards, won_cards in reversed(history):

            ft = get_features(observation, feature_list=self.FT_LIST, 
                played_cards=played_cards, won_cards=won_cards, scores=self.scores)
            features = torch.tensor(ft).to(self.device).float()
            update(self.nn, self.optim, self.device, ret, features)
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
        features = torch.tensor(ft).to(self.device).float()
        self.nn.eval()
        return self.nn(features).detach().item()

    # Perform a one-step lookahead and select the action that has the best expected value
    def greedy_action(self, observation):
        hand = observation['data']['hand']
        valid_moves = filter_valid_moves(observation)
        best_move, best_succ_val = None, float('-inf')
        for move, card in enumerate(valid_moves):
            # set up observation for next state, after play
            succ_hand = [c for c in hand if c != card]
            obs_prime = copy.deepcopy(observation)
            obs_prime['data']['hand'] = succ_hand
            obs_prime['data']['currentTrick'].append({'playerName' : self.name, 'card' : card})
            succ_val = self.value(obs_prime)
            if succ_val >= best_succ_val:
                best_move, best_succ_val = move, succ_val
        return best_move