import random
import numpy as np
from datetime import datetime
import torch

from .utils_env import *
from .utils_nn import *

class MonteCarlo:
    def __init__(self, name, params=dict()):
        # Game Params
        self.name = name
        self.print_info = params.get('print_info', False)

        # Agent Params
        self.EPSILON = params.get('epsilon', .01)
        self.GAMMA = params.get('gamma', .95)
        self.ALPHA = params.get('alpha', .1)
        
        # Load model if desired
        model_name = params.get('load_model', None)
        if model_name:
            self.weight_vec = load_model(model_name, 'mc_simple')
        else:
            self.weight_vec = params.get('weight_vec', np.zeros(52))


    def Do_Action(self, observation):
        if observation['event_name'] == 'PassCards':
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

    def update_weights(self, history, ret):
        errors = []
        for observation in reversed(history):
            hand = observation['data']['hand']
            value = self.value(hand)
            error = ret - value
            features = cards_to_bin_features(hand)
            self.weight_vec += self.ALPHA * error * features
            ret *= self.GAMMA
            errors.append(error)
        return errors

    # Select an action using epsilon-greedy action selection
    def epsilon_greedy_selection(self, observation):
        rand = np.random.uniform(0,1)
        if rand < self.EPSILON:
            return np.random.randint(0, len(filter_valid_moves(observation)))
        else:
            return self.greedy_action(observation)

    # Return the value of a hand
    def value(self, hand):
        return np.dot(cards_to_bin_features(hand), self.weight_vec)

    # Perform a one-step lookahead and select the action that has the best expected value
    def greedy_action(self, observation):
        hand = observation['data']['hand']
        valid_moves = filter_valid_moves(observation)
        best_move, best_succ_val = None, float('-inf')
        for move, card in enumerate(valid_moves):
            succ_hand = [c for c in hand if c != card]
            succ_val = self.value(succ_hand)
            if succ_val >= best_succ_val:
                best_move, best_succ_val = move, succ_val
        return best_move