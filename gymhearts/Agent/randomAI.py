import random
from datetime import datetime
from .agent_utils import *

class RandomAI:
    def __init__(self, name, params=dict()):
        random.seed(datetime.now())
        self.name = name
        self.print_info = params.get('print_info', False) 
    
    def Do_Action(self, observation):
        if observation['event_name'] == 'GameStart':
            if self.print_info:
                #print(handle_event(observation))
                pass
        elif observation['event_name'] == 'NewRound':
            if self.print_info:
                #print(handle_event(observation))
                pass
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
        
        elif observation['event_name'] == 'ShowPlayerHand':
            if self.print_info:
                #print(handle_event(observation))
                pass

        elif observation['event_name'] == 'PlayTrick':
            if self.print_info:
                print(handle_event(observation))

            hand = observation['data']['hand']
            if '2c' in hand:
                choose_card = '2c'
            else:
                #choose_card = random.choice(observation['data']['hand'])
                choose_card = random.choice(filter_valid_moves(observation))
                if self.print_info:
                    print(self.name, 'chose card ::', pretty_card(choose_card))

            return {
                    "event_name" : "PlayTrick_Action",
                    "data" : {
                        'playerName': self.name,
                        'action': {'card': choose_card}
                    }
                }
        elif observation['event_name'] == 'ShowTrickAction':
            if self.print_info:
                #print(handle_event(observation))
                pass
        elif observation['event_name'] == 'ShowTrickEnd':
            if self.print_info:
                #print(handle_event(observation))
                pass
        elif observation['event_name'] == 'RoundEnd':
            if self.print_info:
                #print(handle_event(observation))
                pass
        elif observation['event_name'] == 'GameOver':
            if self.print_info:
                #print(handle_event(observation))       
                pass