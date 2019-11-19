from .agent_utils import *


class Human:
    def __init__(self, name, params=None):
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
            passCards = []
            english_ref = {1 : 'first', 2 : 'second', 3 : 'third'}
            for i in range(3):
                passCards.append(input('{0}, please select your {1} card to pass :: '.format(self.name, english_ref[i+1])))
            
            # Handle the case with inputting 10
            for i in range(len(passCards)):
                card = passCards[i]
                if len(card) == 3 and card[0] == '1' and card[1] == '0':
                    passCards[i] = f'T{card[2]}'
                    
            print(f'\n{self.name} is passing : {" ".join([pretty_card(card) for card in passCards])}')

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
                choose_card = input('{0}, please choose a card :: '.format(self.name))

            return {
                    "event_name" : "PlayTrick_Action",
                    "data" : {
                        'playerName': self.name,
                        'action': {'card': choose_card}
                    }
                }
        elif observation['event_name'] == 'ShowTrickAction':
            if self.print_info:
                print(handle_event(observation))
        elif observation['event_name'] == 'ShowTrickEnd':
            if self.print_info:
                print(handle_event(observation))
        elif observation['event_name'] == 'RoundEnd':
            if self.print_info:
                print(handle_event(observation))
        elif observation['event_name'] == 'GameOver':
            if self.print_info:
                print(handle_event(observation))         