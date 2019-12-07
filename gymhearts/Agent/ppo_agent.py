import random
import numpy as np
from datetime import datetime
import torch
import copy

from .agent_utils import *
from .policy_grad import *

class PPO_Agent:
    def __init__(self, name, params=dict()):
        # Game Params
        self.name = name
        self.print_info = params.get('print_info', False)

        # Agent Params
        self.GAMMA = params.get('gamma', .95)
        self.ALPHA = params.get('alpha', 3e-4)

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

        # NN params
        pi_path = params.get('pi_saved', False)
        baseline_path = params.get('base_saved', False)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.pi = PiApproximationWithNN(input_features=feature_length(self.FT_LIST), 52, self.ALPHA, pi_path)
        self.baseline = VApproximationWithNN(input_features=feature_length(self.FT_LIST), self.ALPHA, baseline_path)


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

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    # Select an action using epsilon-greedy action selection
    def action_select(self, observation):
        ft =  torch.tensor(features_state(observation)).to(self.device)
        val_filter = self.get_filter(observation)
        return self.pi(features, val_filter).detach().item()

    def features_state(self, observation):
        return get_features(observation, feature_list=self.FT_LIST, 
            played_cards=self.played_cards, won_cards=self.won_cards, scores=self.scores)

    def get_filter(self, observation):
        val_moves = filter_valid_moves(observation)
        deck_ref = deck_reference()
        indices_valid = [deck_ref.index(card) for card in val_moves]
        val_filter = torch.zeros(52)
        for index in indices_valid:
            val_filter[index] = 1
        return val_filter

    def get_state_info(self, observation):
        return self.features_state(observation), self.get_filter(observation)