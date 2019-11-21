import random
from datetime import datetime
import numpy as np
from gymhearts.Hearts import Card
from gymhearts.Agent.agent_utils import filter_valid_moves

class MC:
    def __init__(self, name, epsilon=.05, gamma=.95, alpha=.1, params=dict()):
        self.name = name
        self.EPSILON = epsilon
        self.GAMMA = gamma
        self.ALPHA = alpha

        # simple weight vector, 1 weight for each card in hand
        self.weight_vec = params.get('weights', np.zeros(52))

        #used for feature vector encoding
        self.all_cards = []
        for suit in range(0,4):
            for rank in range(2,15):    
                self.all_cards.append(str(Card(rank, suit)))

        self.print_info = params.get('print_info', False)

    def Do_Action(self, observation):
        if observation['event_name'] == 'PassCards':
            if self.print_info:
                print(observation)
            # Random passing of a card 
            passCards = random.sample(observation['data']['hand'], 3 )
            
            if self.print_info:
                print(self.name, ' pass cards: ', passCards)
            return {
                    "event_name" : "PassCards_Action",
                    "data" : {
                        'playerName': self.name,
                        'action': {'passCards': passCards}
                    }
                }
        elif observation['event_name'] == 'PlayTrick':
            if self.print_info:
                print(observation)

            hand = observation['data']['hand']
            if '2c' in hand:
                choose_card = '2c'
            else:
                card_idx = self.epsilon_action(observation)
                choose_card = hand[card_idx]
                if self.print_info:
                    print(self.name, ' choose card: ', choose_card)

            return {
                    "event_name" : "PlayTrick_Action",
                    "data" : {
                        'playerName': self.name,
                        'action': {'card': choose_card}
                    }
                }
        else:
            if self.print_info:
                print(observation)

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
        return self.q_argmax(value_fn, observation) if np.random.uniform(0, 1) > self.EPSILON else np.random.randint(0, len(arr))

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