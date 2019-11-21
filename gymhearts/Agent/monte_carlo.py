import random
import numpy as np
from datetime import datetime

from .agent_utils import *

class MonteCarlo:
    def __init__(self, name, params=dict()):
        # Game Params
        self.name = name
        self.print_info = params.get('print_info', False)

        # Agent Params
        self.EPSILON = params.get('epsilon', .05)
        self.GAMMA = params.get('gamma', .95)
        self.ALPHA = params.get('alpha', .1)

        # Value function params
        self.weight_vec = params.get('weight_vec', np.zeros(52))
        self.whole_deck = whole_deck()

    def Do_Action(self, observation):
        if observation['event_name'] == 'PassCards':
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
                card_idx = self.epsilon_action(observation)
                choose_card = filter_valid_moves(observation)[card_idx]
                if self.print_info:
                    print(self.name, 'chose card ::', pretty_card(choose_card))

            return {
                    "event_name" : "PlayTrick_Action",
                    "data" : {
                        'playerName': self.name,
                        'action': {'card': choose_card}
                    }
                }

    def update_reward_fn(self, history, reward):
        returns = reward
        for observation in reversed(history):
            reward = 0 if reward == None else reward
            returns = reward + self.GAMMA*returns
            value_fn = self.get_value_fn(observation['data']['hand'])
            value = value_fn.sum()
            self.weight_vec += self.ALPHA * (reward - value)*self.get_feature_vec(observation['data']['hand'])

    def epsilon_action(self, observation):
        arr = observation['data']['hand']
        value_fn = self.get_value_fn(arr)
        return self.q_argmax(value_fn, observation) if np.random.uniform(0, 1) > self.EPSILON else np.random.randint(0, len(filter_valid_moves(observation)))

    def get_value_fn(self, arr):
        return self.get_feature_vec(arr) * self.weight_vec

    def get_feature_vec(self, arr):
        feature_vec = np.zeros(52)
        for card in arr:
            feature_vec[self.all_cards.index(card)] = 1
        return feature_vec 

    def q_argmax(self, value_fn, observation):
        value = value_fn.sum()
        i_max = 0
        value_p = float('-inf')
        valid = filter_valid_moves(observation)
        for i in range(len(valid)):
            idx = self.all_cards.index(valid[i])
            if(value - value_fn[idx] > value_p):
                i_max = i
                value_p = value - value_fn[idx]
        return i_max