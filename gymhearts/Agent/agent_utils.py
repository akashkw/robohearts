'''
Utilities to be implemented in all agents
'''

def pretty_card(card):
    rank = card[0].upper() if card[0] != 'T' else '10'
    suit = card[1]
    suit_lookup = {'c':'♣', 'd':'♦', 's':'♠', 'h':'♥'}
    return f'[{rank}{suit_lookup[suit]}]'

def filter_valid_moves(observation):
    data = observation['data']
    hand = data['hand']
    trick_num = data['trickNum']
    trick_suit = data['trickSuit']
    no_suit = trick_suit == 'Unset'
    hearts_broken = data['IsHeartsBroken']

    suit_in_hand = True
    if not no_suit:
        suit_in_hand = False
        for card in hand:
            if trick_suit in card:
                suit_in_hand = True
                break
    
    valid_cards = []
    # First move, only 2c
    if trick_num == 1 and no_suit:
        if '2c' in hand:
            valid_cards.append('2c')
    # Starting trick, hearts broken, all cards valid
    elif hearts_broken and no_suit:
        valid_cards = hand
    # Starting trick, hearts not broken, all non-heart cards valid
    elif no_suit:
        for card in hand:
            if 'h' not in card:
                valid_cards.append(card)
        # Nothing but cards in hand
        if not valid_cards:
            valid_cards = hand
    # Not starting trick, valid suit in hand, only cards of suit valid
    elif suit_in_hand:
        for card in hand:
            if trick_suit in card:
                valid_cards.append(card)
    # Not starting trick, valid suit not in hand, all cards valid
    else:
        valid_cards = hand
        # Can't play queen of spades or hearts in first trick
        if trick_num == 1 and len(valid_cards) > 1:
            valid_cards = [card for card in valid_cards if card != 'Qs' and 'h' not in card]
    return valid_cards

def handle_event(observation):
    event = observation['event_name']
    if event == 'PassCards':
        hand = observation['data']['hand']
        phand = [pretty_card(card) for card in hand]
        retstring = f"\n{observation['data']['playerName']}'s Hand\n{' '.join(phand)}\n"
        return retstring
    elif event == 'PlayTrick':
        hand = observation['data']['hand']
        phand = [pretty_card(card) for card in hand]
        filtered_hand = filter_valid_moves(observation)
        filtered_phand = [pretty_card(card) for card in filtered_hand]
        retstring = f"\n{observation['data']['playerName']}'s Hand\n{' '.join(phand)}\n"
        retstring += f"\nValid Moves\n{' '.join(filtered_phand)}\n"
        return retstring
    else:
        return observation
