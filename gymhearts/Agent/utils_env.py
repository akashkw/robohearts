import numpy as np

'''
Utilities to be implemented in all agents
'''
# -------------- CARD / SCORE UTILS --------------

suits = ["c", "d", "s", "h"]
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

# List of all valid cards
def create_deck():
    deck = list()
    for suit in suits:
        for rank in ranks:
            deck.append(f'{rank}{suit}')
    return deck

def create_points():
    pts = list()
    for rank in ranks:
        suit = 'h'
        pts.append(f'{rank}{suit}')
    pts.append('Qs')
    return pts

# Reference to get index for each card
def deck_reference():
    deck = create_deck()
    return {card : i for i, card in enumerate(deck)}

# Reference to get index for each point bearing card
def pts_reference():
    pts = create_points()
    return {card : i for i, card in enumerate(pts)}

# Return the number of features associated with each feature group
def feature_length(feature_list):
    count = 0
    if 'in_hand' in feature_list:
        count += 52
    if 'in_play' in feature_list:
        count += 52
    if 'played_cards' in feature_list:
        count += 52
    if 'won_cards' in feature_list:
        count += 4 * 14
    if 'scores' in feature_list:
        count += 4
    return count

# Format cards from Hearts.Card format to pretty format
def pretty_card(card):
    rank = card[0].upper() if card[0] != 'T' else '10'
    suit = card[1]
    suit_lookup = {'c':'♣', 'd':'♦', 's':'♠', 'h':'♥'}
    return f'[{rank}{suit_lookup[suit]}]'

# -------------- GAMEPLAY UTILS --------------

# Return a list of all valid moves in Hearts.Card format
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

# Handle specific observations by presenting human friendly prompts
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


# ------------- FEATURE UTILS --------------

def cards_to_bin_features(cards):
    deck = deck_reference()
    feature_vec = np.zeros(52)
    for card in cards:
        feature_vec[deck[card]] = 1
    return feature_vec 

def cards_to_valid_bin_features(cards):
    valid_cards = filter_valid_moves(cards)
    return cards_to_bin_features(valid_cards)

def in_hand_features(observation):
    return cards_to_bin_features(observation['data']['hand'])

def in_play_features(observation):
    in_play_cards = [entry['card'] for entry in observation['data']['currentTrick']]
    return cards_to_bin_features(in_play_cards)

def played_cards_features(played_cards):
    return cards_to_bin_features(played_cards)

def won_cards_features(won_cards):
    point_cards = pts_reference()
    feature_vec = np.zeros((4, 14))
    for player, won_card in enumerate(won_cards):
        for card in won_card:
            if card in point_cards:
                feature_vec[player][point_cards[card]] = 1
    return feature_vec.flatten()

def scores_features(scores):
    return np.array(scores)

# Generate a list of features based on the feature list
def get_features(observation, feature_list=['in_hand'], played_cards=None, won_cards=None, scores=None):
    features = np.array([])

    if 'in_hand' in feature_list:
        features = np.concatenate([features, in_hand_features(observation)])
    if 'in_play' in feature_list:
        features = np.concatenate([features, in_play_features(observation)])
    if 'played_cards' in feature_list:
        features = np.concatenate([features, played_cards_features(played_cards)])
    if 'won_cards' in feature_list:
        features = np.concatenate([features, won_cards_features(won_cards)])
    if 'scores' in feature_list:
        features = np.concatenate([features, scores_features(scores)])
    return features