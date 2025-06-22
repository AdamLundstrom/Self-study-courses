# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:38:24 2024

@author: eamadlu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 11:28:54 2023

@author: eamadlu
"""

import numpy as np
import tensorflow as tf
import utils as u
import random
import os
import matplotlib.pyplot as plt

import time
import random
import heapq as hp
import numpy as np
from keras import backend as K
import keras
from collections import deque
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
import tensorflow_probability as tfp
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
from keras.optimizers import  Adam
from scipy import signal
import tensorflow as tf
import evaluate as e

import model as m
# config =tf.ConfigProto(
#         device_count={'GPU':0}
# )
# sess=tf.Session(config=config)


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.9):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, logprobability, action, value, reward):
        
        
        # print(f"Cards {cards}")
        # print(f"Reward {reward}")
        # print(f"Value {value}")
        # print(f"Action {action}")
        
        # raise ValueError
        # print(logprobability)
        # raise
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)
        # print(rewards)
        # print(values)
        
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]
        # print(deltas)
        # print(self.advantage_buffer[path_slice])
        # print(self.return_buffer[path_slice])
        # raise
        self.trajectory_start_index = self.pointer

    def discounted_cumulative_sums(self, x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]    

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )




def train_rl(number_rounds,batch_size,path_playing,previous_path_playing,path_trading,previous_path_trading,num_players,models_type,trading=False):
    # m1_main = change_model()
    # m1_temp = change_model()
    # m1_main.load_weights(f"{path}\\m1_main.ckpt").expect_partial()
    # m1_temp.load_weights(f"{path}\\m1_temp.ckpt").expect_partial()

    model_state = "train"

    target_kl = 0.5

    state_size = 94
    # buffer_size = 10000
    buffer_size = 1024
    # buffer1 = Buffer(state_size, buffer_size)
    # buffer2 = Buffer(state_size, buffer_size)
    # buffer3 = Buffer(state_size, buffer_size)
    train_iterations = 5
    save_best_lim = buffer_size*10
    save_best_count = 0

    action_size = 5
    mini_batch_size_2 = buffer_size
    mini_batch_size_1 = buffer_size
    # play_agent1 = m.Agent(6)
    # play_agent3 = m.Agent(6)
    # chicago_agent1 = m.Agent(2)
    # chicago_agent2 = m.Agent(2)
    # chicago_agent3 = m.Agent(2)

    # m2_main = play_model()
    # m2_temp = play_model()


    # num_players = 4
    global_best_model_play = 0
    global_best_model_trade = 0
    winner_count = np.zeros(num_players)
    global_average_points_play = []
    global_average_reward_play = []
    global_average_points_trade = []
    global_average_reward_trade = []
    global_cumulative_wins = []
    global_critic_loss_trade = []
    
    
    best_score_model = np.zeros(num_players)
    
    # players = [play_agent1,play_agent2,play_agent3]
    # chicago_determiners = [chicago_agent1,chicago_agent2,chicago_agent3]
    # buffer_players = [buffer1,buffer2,buffer3]
    # buffer_chicago_determiners = [buffer1,buffer2,buffer3]

    kl_values = []
    
    loss_critic = []
    loss_actor = []
    
    players = []
    buffer_players = []
    for i in range(num_players):
        if models_type[i] == "random":
            players.append(m.RandomAgent(5))
            buffer_players.append(None)
        else:
            players.append(m.Agent(5))
            buffer_players.append(Buffer(state_size, buffer_size))


    # previous_path_play = rf"saved_models/v15/best"        
    if previous_path_playing is not None:
        for p in players:
            p.load(f"{previous_path_playing}/m2_actor.ckpt",f"{previous_path_playing}/m2_critic.ckpt")


    # previous_path_play = rf"saved_models/v15/best"        
    # players[2].load(f"{previous_path_playing}/m2_actor.ckpt",f"{previous_path_playing}/m2_critic.ckpt")
 
    
    traders = []
    if trading:
        state_size_trade = 84
        buffer_size_trade = buffer_size
        
        buffer_traders = []
        for i in range(num_players):
            if models_type[i] == "random":
                traders.append(m.RandomAgent(32))
                buffer_players.append(None)
            else:
                traders.append(m.Agent(32))
                buffer_traders.append(Buffer(state_size_trade, buffer_size_trade))
            
        if previous_path_trading is not None:
            for t in traders:
                t.load(f"{previous_path_trading}/m1_actor.ckpt",f"{previous_path_trading}/m1_critic.ckpt")        

    award_per_player = np.zeros(num_players) 
    points_per_player = np.zeros(num_players)
    av_points_per_player = []
    av_award_per_player = []
    av_award_per_player_test = []
    played_rounds_count = 0
    for round_ in range(number_rounds):  

        number_runs = []

        chicago_count = np.zeros(num_players)
        
        bad_trade = []
        avoided_bad_trade = []
        overall_winner = []

        missed_points = []
        retreived_points = []
        bad_play = 0
        avoided_bad_play = 0
        bad_play_control = 0
        avoided_bad_play_control = 0
        
        
        
        buffer_count = 0
        print(f"Round {round_}")

        #Not really used currently
        for batch in range(batch_size):
            # print(f"Batch {batch}")
            previous_cards = np.zeros([num_players,15])
            points_counter = np.zeros(num_players)
            
             
            points_per_trader = np.zeros(num_players)
            award_per_trader = np.zeros(num_players)  
            
            
            runs = 0
            
            break_flag = False
            
            traded_rounds_count = 0

            
            while np.max(points_counter) < 52:

                ################################
                ###INIT EACH GAME
                ################################             
                runs += 1
                deck = (np.arange(52)+2).astype(float)
                deck = u.shuffle(deck)
                deck, players_cards = u.deal_cards(deck,num_players)
                local_reward_m2 = []
                local_saved_data_m2 = []     
                local_reward_m1 = []
                local_saved_data_m1 = []  
                
                cards_storage = []
                
                for i in range(num_players):
                    players_cards[i] = -np.sort(-players_cards[i])

                previous_cards_traid = []
                for i in range(num_players):
                    
                    previous_cards_traid.append(np.zeros(15))
                    local_saved_data_m2.append([])
                    local_reward_m2.append(0)
                    local_saved_data_m1.append([])
                    local_reward_m1.append(0)
                    
                    cards_storage.append([])
                    
                    
                # print(players_cards)
                ###############################
                ##TRADING
                ###############################
                if not trading:
                    previous_cards_traid = u.assign_random_previous_cards(previous_cards_traid,deck,num_players-1)
                    # previous_cards_traid = np.zeros()
                    
                if trading:
                    
                    best_points = 0
                    for r in range(3):     

                        for i in range(num_players):
                            # print(i)
                            if points_counter[i] < 42:
                                points_init, _ = u.get_points(players_cards[i])
                                    
                           
                                # if i > 0:
                                #     cards_storage[i-1].append(players_cards[i].copy())       
                                # print(cards_storage[i-1])
                                # print(players_cards[i])
                                # num_players, i, r, local_reward_m1, local_saved_data_m1, players_cards, previous_cards_traid, traders, buffer_traders, mini_batch_size_1, points_init, points_counter, best_points, compared_player_flag=False
                                                                                
                                action, local_reward_m1, local_saved_data_m1 = m.trade(model_state, num_players, traders, i, r, points_init, points_counter, best_points, players_cards, previous_cards, local_reward_m1, local_saved_data_m1,  buffer_traders, mini_batch_size_1)

                                for c in players_cards[i]:
                                    previous_cards_traid[i-1] = np.roll(previous_cards_traid[i-1],1)
                                    previous_cards_traid[i-1][0] = c
                           

                                
                                if action != 0:    
                                    if points_init > 1: 
                                        missed_points.append(1)
                                        
                                    deck, p = u.change_cards(deck,players_cards[i],action)
                                    players_cards[i] = p
                                    players_cards[i] = -np.sort(-players_cards[i])
                                    

                                if  action == 0 and points_init > 1:
                                    retreived_points.append(1)

                                points, _ = u.get_points(players_cards[i])
                                

                                local_reward_m1[i-1] -= points_init                       
                                # if i > 0:
                                #     local_reward_m1[i-1] += (float(points))
                                # award_per_trader[i] += (float(points))
                                if points_init < 42 and points == 42:
                                    if i > 0:
                                        local_reward_m1[i-1] -= 3.
                                    award_per_trader[i] -= 3.
                                elif points_init < 42 and points == 43:
                                    if i > 0:
                                        local_reward_m1[i-1] -= 2.
                                    award_per_trader[i] -= 2.
                                elif points_init < 42 and points == 44:
                                    if i > 0:
                                        local_reward_m1[i-1] -= 1.
                                    award_per_trader[i] -= 1.
                                
                                # average_points_player.append(points)

                                # if i == 1: 
                                #     print("----")
                                #     print(local_reward_m1)
                                #     print(points_init)
                                #     print(points)
                                #     print(action)
                                #     print(f"Change: {r}")
                                #     print(u.get_color_values(players_cards[i]))
                                #     print(u.get_card_values(players_cards[i]))  
                                #     print("------")
     
                                
                        traded_rounds_count += 1
                        max_player, max_points = u.determine_point_winner(players_cards)
                        best_points = max_points
        
                        if max_player is not None:
                            points_counter[max_player] += max_points
                            award_per_trader[max_player] += max_points
                        
                        for j in range(num_players):
                            if j == max_player and points_counter[j] < 42:
                                local_reward_m1[j-1] += float(max_points)+1
                            elif j != max_player and points_counter[j] < 42:
                                local_reward_m1[j-1] += -1
                            
                            if j != max_player and points_counter[j] < 42:
                                award_per_trader[j] += -1               
                        

                         
                        # print("--round eval--")
                        # print(max_player)
                        # print(max_points)
                        # print(local_reward_m1)
                        # print("------")       
                        # print("------------------------------------")                         

                            
                            
                        buffer_size = []
                        for i in range(len(buffer_traders)):
                            buffer_size.append(buffer_traders[i].pointer)                        
                        if r + 1 != 3:
                            if np.max(buffer_size) < mini_batch_size_1-2:
                                for i in range(num_players):
                                    if points_counter[i] < 42: 
                                        local_saved_data_m1[i].append(local_reward_m1[i])
    
                                        try:
                                            if models_type[i] != "random":
                                                buffer_traders[i].store(local_saved_data_m1[i][0],local_saved_data_m1[i][1],local_saved_data_m1[i][2],local_saved_data_m1[i][3],local_saved_data_m1[i][4])

        
                                        except:
                                            print("BUFFER FAULT")
                                            print(np.max(buffer_size))
                                            print()
                                            print(buffer_count)
                                            print(len(local_saved_data_m1[i]))
                                            print(local_saved_data_m1[i])
                                            raise ValueError
                    
 
                for i in range(len(local_reward_m1)):
                    local_reward_m1[i] = 0
                    
                # print(local_reward_m1[1])                
                ################################
                ###CHICAGO EVAL
                ################################
                previous_cards_played = np.zeros([6,5])
                chicago_flag = 0
                players_cards_play = players_cards.tolist()   
                weights = np.zeros(5)+1
                init = np.zeros(num_players)+1
                starter = random.randint(0,num_players-1)
                order_play = []
                order_play.append(starter)
                for j in range(num_players):
                    if j > starter: 
                        order_play.append(j)
                for j in range(num_players):
                    if j < starter: 
                        order_play.append(j)
                chicago_player = None
                
                # for i in order_play:

                #     index = 0
                #     left = num_players-1
                #     controling_card = 0
                    
                #     selected, local_reward_m2, local_saved_data_m2 = m.play(model_state, models_type[i],players, num_players, chicago_flag, i, 
                #                                                            left, index ,controling_card,
                #                                                            players_cards_play,previous_cards_played, 
                #                                                               previous_cards_traid, init, local_reward_m2, local_saved_data_m2, weights, 
                #                                                               buffer_players, mini_batch_size_2)

                #     init[i] = 0
                #     # if selected == 5:
                #     #     chicago_flag = 1
                #     #     chicago_player = i
                #     #     # if u.posibility_of_certain_chicago(players_cards[i]):
                #     #     #     local_reward_m2[i] += 1   
                #     #     #     award_per_player[i] += 1                            
                #     #     break    

                #     # if u.posibility_of_certain_chicago(players_cards[i]):
                #     #     local_reward_m2[i] -= 1   
                #     #     award_per_player[i] -= 1
                        
                if chicago_flag:
                    starter = chicago_player
                    chicago_count[chicago_player] += 1
                

                ######################################
                
                ######################################
                ##PLAYING
                ######################################
                break_flag = False 
                # weights[-1] = 0
                # print("0")
                # print("new")
                for i in range(5):
                    # print(i)
                    
                    # print("--!---")
                    # print(i)
                    # print("---!--")
                    controling_card = 0.0

                    played_cards = np.zeros(num_players)
                    
                    if i == 4:
                        selected = np.where(np.array(players_cards_play)[starter] > 0)[0][0]
                    else:

                        weights_play = u.get_acceptable_play(controling_card, players_cards_play[starter])
                     
                        
                        # weights_play = weights*np.insert(weights_play,len(weights_play),1) 
                        weights_play = weights*weights_play 
                        
                        # weights_play = weights_play
                        # print("--------------")
                        
                        # print("start")
                        
                        # print(starter)
         
                        # print(weights_play)
                        
                        
                        selected, local_reward_m2, local_saved_data_m2 = m.play(model_state, models_type[starter], players, num_players, chicago_flag, starter, num_players-1, 
                                                                                i ,controling_card,
                                                                               players_cards_play,previous_cards_played, 
                                                                                previous_cards_traid, init, local_reward_m2, local_saved_data_m2, weights_play, 
                                                                                                   buffer_players, mini_batch_size_2)
                        # print(selected) 
                        # print(players_cards_play[starter])
                        # print("------")
                        init[starter] = 0
                        
                    controling_card = players_cards_play[starter][selected]
                    
                    init_bad_play = True
                    if controling_card == 0:
                        raise ValueError
                    
                    # del players_cards_play[starter][selected]
                    previous_cards_played[starter][i] = controling_card
                    players_cards_play[starter][selected] = 0.0
                    played_cards[starter] = controling_card
                    if break_flag:
                        break
                    
                    order_play = []
                    for j in range(num_players):
                        if j > starter: 
                            order_play.append(j)
                    for j in range(num_players):
                        if j < starter: 
                            order_play.append(j)                        
                        
                    left_count = 0
                    
                    # print("Starter")
                    # print(starter)
                    # print(played_cards[starter])
                    # print(players_cards_play)
                    # print("----")
                    
                    for j in order_play:
                            
                        init_bad_play = False
                        init_bad_play2 = False
                        
                        if j != starter:
                            
                            if i == 4:
                                selected = np.where(np.array(players_cards_play)[j] > 0)[0][0]
                            else:
                                a_play = False
                                while not a_play:
     
                                    if not init_bad_play:
                                        
                                        init_bad_play = True
                                        
                                        weights_play = u.get_acceptable_play(controling_card, players_cards_play[j])

                                        # weights_play = weights*np.insert(weights_play,len(weights_play),1)
                                        weights_play = weights*weights_play 
                                        # weights_play = weights
                                        # print("_----")
                                        # print(players_cards_play[j])
                                        # print(j)
                                        # print(weights_play)
                        
                                        left = len(order_play)-left_count-1
                                        
                                        selected,local_reward_m2, local_saved_data_m2 = m.play(model_state, models_type[j], players, num_players, chicago_flag, j, 
                                                                                               left, i ,controling_card,
                                                                                              players_cards_play,previous_cards_played, 
                                                                                                previous_cards_traid, init, local_reward_m2, local_saved_data_m2, weights_play, 
                                                                                                buffer_players, mini_batch_size_2)                                        

                                        # print(selected)
                                        # print("------")
                                        init[j] = 0                                
    
                                    elif init_bad_play and not init_bad_play2:
                                        print("---Bad play")
                                        print(len(local_saved_data_m2[j]))
                                        print(local_saved_data_m2[j])
                                        
                                        print(j)
                                        print(selected)
                                        print(controling_card)
                                        print(starter)
                                        print(players_cards_play)
                                        print(u.get_color_values(players_cards_play[j]))
                                        
                                        print(weights_play)
                                       
                                        raise ValueError
                                        bad_play += 1
                                        local_reward_m2[j] -= 1.
                                        init_bad_play2 = True
                                        break_flag = True
                                        break

                                    
                                    # controling_card = controling_card.numpy()
                               
                                    a_play = u.acceptable_play(players_cards_play[j][selected], players_cards_play[j],controling_card)                

                                if not init_bad_play2:
                                    avoided_bad_play += 1
                                    # local_reward_m2[-1] += 1.
                            
                            played_cards[j] = players_cards_play[j][selected]
                            previous_cards_played[j][i] = played_cards[j]
                            players_cards_play[j][selected] = 0.0
                            # print("Player")
                            # print(j)
                            # print(played_cards[j])
                            # print(players_cards_play)
                            # print("----") 

                            # print(previous_cards_played)
                            left_count += 1
                            if break_flag:
                                break                            

                    # weights[5-i-1] = 0
                    played_rounds_count += 1
                    if break_flag:
                        break     
                    
                    starter = u.local_winner(played_cards, controling_card)
                    
                    buffer_count += 1
                    
                    if chicago_flag and starter != chicago_player:
                        break
                # raise ValueError
                #############################################
                ##DONE PLAYING
                ################################################
                # print("AFTER PLAYING")
                # print(local_reward_m1[1])
                # print(f"CHICAGO PLAYER {chicago_player}")
                
                #################################
                ##EVAL POINTS
                ###################################
                winner = starter
                if chicago_flag and chicago_player != winner:
                    award_per_player[chicago_player] -= 3
                    points_counter[winner] -= 0.
                    points_per_player[winner] -= 0.
                    award_per_trader[chicago_player] -= 15
                    local_reward_m2[winner] += 0.5
                    local_reward_m2[chicago_player] -= 3
                    if trading and points_counter[chicago_player] < 42:
                        local_reward_m1[chicago_player] -= 15. 
                        
                elif chicago_flag and chicago_player == winner:
                    award_per_player[chicago_player] += 3.
                    points_counter[winner] += 15.   
                    points_per_player[winner] += 15.
                    award_per_trader[chicago_player] += 15
                    local_reward_m2[chicago_player] += 3.
                    if trading and points_counter[chicago_player] < 42:
                        local_reward_m1[chicago_player] += 15. 
       
                else:    
                    award_per_player[winner] += 1
                    points_counter[winner] += 5.
                    points_per_player[winner] += 5.
                    award_per_trader[winner] += 5
                    local_reward_m2[winner] += 1

                # for i in range(num_players):
                #     if i != winner:
                #         award_per_player[i] -= 5
                #         local_reward_m2[i] -= 5

                
                max_player, max_points = u.determine_point_winner(players_cards)
                if max_player is not None:
                    points_counter[max_player] += max_points
                    award_per_trader[max_player] += max_points
                for j in range(num_players):
                    if j == max_player  and points_counter[j] < 42:
                        local_reward_m1[j] += float(max_points)
                    elif j != max_player and points_counter[j] < 42:
                        local_reward_m1[j] += -1
                    if j != max_player and points_counter[j] < 42:
                        award_per_trader[j] += -1    
                    
                # print(local_reward_m1[0])
                # for i in range(num_players):
                #     if winner - 1 == i:
                #         # for j in range(3):
                #         #     buffer_players[i].reward_buffer[buffer_players[i].pointer-j] += 1
                #         local_reward_m2[i] += 1. 
                #         if trading and points_counter[i] < 42:
                #             local_reward_m1[i] += 5. 
                #         best_score_model[i] += 1.
                #     else:
                #         # for j in range(3):
                #         #     buffer_players[i].reward_buffer[buffer_players[i].pointer-j] -= 1                        
                #         local_reward_m2[i] -= 1  
                #         best_score_model[i] -= 1.5
                #         if trading and points_counter[i] < 42:
                #             local_reward_m1[i] -= 1.

                buffer_size = []
                for i in range(num_players):
                    if models_type[i] != "random":
                        buffer_size.append(buffer_players[i].pointer)

                if np.max(buffer_size) < mini_batch_size_2-1:

                    for i in range(len(buffer_players)):
                        local_saved_data_m2[i].append(local_reward_m2[i])
                        # print("-----")
                        # print(local_reward_m2[i])
                        # print(winner)
                        # print(chicago_player)
                        try:
                            # print(local_saved_data_m2[i])
                            if models_type[i] != "random":
                                buffer_players[i].store(local_saved_data_m2[i][0],local_saved_data_m2[i][1],local_saved_data_m2[i][2],local_saved_data_m2[i][3],local_saved_data_m2[i][4])
                            # print(buffer_players[i].reward_buffer[:20])
                        except:
                            print("BUFFER FAULT")
                            
                            print(buffer_count)
                            raise ValueError
 
                if trading:
                    buffer_size = []
                    for i in range(num_players):
                        buffer_size.append(buffer_traders[i].pointer)
                    if np.max(buffer_size) < mini_batch_size_1-1:
                        for i in range(num_players):
                            if points_counter[i] < 42: 
                                local_saved_data_m1[i].append(local_reward_m1[i])
                                try:
                                    if models_type[i] != "random":
                                        buffer_traders[i].store(local_saved_data_m1[i][0],local_saved_data_m1[i][1],local_saved_data_m1[i][2],local_saved_data_m1[i][3],local_saved_data_m1[i][4])
        
                                except:
                                    print("BUFFER FAULT")
                                    
                                    print(buffer_count)
                                    raise ValueError
                                    
                # print(f"Winner {winner}")
                # print(u.get_color_values(players_cards[1]))
                # print(u.get_card_values(players_cards[1])) 
                # print(max_player)
                # print(max_points)

                # print(local_reward_m1[0])
                # print("_--")
                # for i in range(5):
                #     print(buffer_traders[0].reward_buffer[i])
                
                # raise ValueError
                last_value = 0
                for i in range(num_players):
                    if models_type[i] != "random": 
                        buffer_players[i].finish_trajectory(last_value)
                    if trading and points_counter[i] < 42:
                        if models_type[i] != "random":
                            buffer_traders[i].finish_trajectory(last_value)
   
                if save_best_count >= save_best_lim:
                    
                    
                    loss_actor_temp = []
                    loss_critic_temp = []
                    for i in range(len(loss_actor)):
                        if i < 5:
                            loss_actor_temp.append(loss_actor[i])
                            loss_critic_temp.append(loss_critic[i])
                        else:
                            loss_actor_temp.append(np.mean(loss_actor[i-5:i]))
                            loss_critic_temp.append(np.mean(loss_critic[i-5:i]))
                    
                    
                    scores_play, scores_trade = e.evaluate(400,players,num_players,models_type,traders,trading=False,print_out=True)
                    
                    
                    fig, ax = plt.subplots(1,1)
                    ax.plot(np.arange(len(loss_actor_temp)),loss_actor_temp)
                    ax.set_title("Loss Actor")
                    ax.set_xlabel("Round")
                    plt.show()    
                    fig, ax = plt.subplots(1,1)
                    ax.plot(np.arange(len(loss_critic_temp)),loss_critic_temp)
                    ax.set_title("Loss Critic")
                    ax.set_xlabel("Round")
                    plt.show()    
                    av_award_per_player.append(award_per_player/played_rounds_count+0.00001)
                    av_points_per_player.append(points_per_player/played_rounds_count+0.00001)
                    av_award_per_player_test.append(scores_play[1])
                    
                    fig, ax = plt.subplots(1,1)
                    ax.plot(np.arange(len(np.array(av_award_per_player))),np.array(av_award_per_player)[:,1])
                    ax.set_title("Award/player train")
                    ax.set_xlabel("Round")
                    plt.show()
                    fig, ax = plt.subplots(1,1)
                    ax.plot(np.arange(len(np.array(av_award_per_player_test))),np.array(av_award_per_player_test))
                    ax.set_title("Award/player test")
                    ax.set_xlabel("Round")
                    plt.show()                        
                    # fig, ax = plt.subplots(1,1)
                    # ax.plot(np.arange(len(np.array(av_points_per_player))),np.array(av_points_per_player)[:,1])
                    # ax.set_title("Points/player")
                    # ax.set_xlabel("Round")
                    # plt.show()
                    
                    
                    
                    
                    weights = np.ones(num_players)
                    for i in range(num_players):
                        if models_type[i] == "random":
                            weights[i] = 0
                            
                    ind_play = np.argmax(scores_play*weights)
                    global_best_model_play = ind_play
                    
                    
                    # ind_trade = np.argmax(scores_trade*weights)
                    # global_best_model_trade = ind_trade
                    print("-------------------------------")
                    print("Choosing best model to continue")
                    print(f"AVERAGE SCORE {award_per_player} ")
                    print(f"AVERAGE POINTS {points_per_player} ")
                    print("-------------------------------")
                    if ind_play == 1:
                        # if award_per_player[1] > 0.071:
                        print("here")
                        # for i in range(num_players):
                        #     if i != ind_play and models_type[i] != "random":
                        # players[2].actor.set_weights(players[ind_play].actor.get_weights())
                        # players[2].critic.set_weights(players[ind_play].critic.get_weights())
                    
                        print("here2")
                        players[1].save(f"{path_playing}/{round_}/m2_actor.ckpt",f"{path_playing}/{round_}/m2_critic.ckpt") 
                    if trading:
                        pass
                            # for i in range(num_players):
                            #     if i != ind_trade and models_type[i] != "random":
                            #         traders[i].actor.set_weights(traders[ind_trade].actor.get_weights())
                            #         traders[i].critic.set_weights(traders[ind_trade].critic.get_weights())
                    save_best_count = 0
                    best_score_model = np.zeros(num_players)

                    award_per_player = np.zeros(num_players) 
                    points_per_player = np.zeros(num_players)
                    played_rounds_count = 0
                      
                else:
                    save_best_count += 1

                buffer_size = []
                if trading:
                    for i in range(len(buffer_traders)):
                        if models_type != "random":
                            buffer_size.append(buffer_traders[i].pointer)
                            
                if trading and np.max(buffer_size) >= mini_batch_size_1-1:
                    
                    buffer_count = 0

                    print("Train trading")
                    for i in range(num_players):
                        if models_type[i] != "random":
                            
                            (
                                observation_buffer,
                                action_buffer,
                                advantage_buffer,
                                return_buffer,
                                logprobability_buffer,
                            ) = buffer_traders[i].get()
                            kl_temp = []
                            for _ in range(train_iterations):
                            # a_loss, c_loss = agent.train(saved_data_m2,gamma=0.99)
                                kl = traders[i].train_policy(
                                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                                )
                                
                                kl_temp.append(kl)
                                # players[i].train_value_function(observation_buffer, return_buffer)
                           
                                # if kl > 1.5 * target_kl:
                                #     print(f"break {i}")
                                #     break                    
                                
                                # print(f"Actor loss: {a_loss}, Critic loss: {c_loss}")
                            
                            kl_values.append(np.mean(kl_temp))
                            loss = []
                            for _ in range(train_iterations):    
                                traders[i].train_value_function(observation_buffer, return_buffer)
                            
                

                buffer_size = []
    
                for i in range(num_players):
                    if models_type[i] != "random":
                        buffer_size.append(buffer_players[i].pointer)
                        
                if np.max(buffer_size) >= mini_batch_size_2-1:
                    
                    print("Train playing")

                    # weights = np.ones(num_players)
                    # for i in range(num_players):
                    #     if models_type[i] == "random":
                    #         weights[i] = 0
                            
                    # ind_play = np.argmax(np.mean(global_average_reward_play,axis=0)*weights)
                    # # print(np.array(global_average_reward_play.shape)
                    # if ind_play == 1:
                    #     print("Updated based on player 1")
                    #     for i in range(num_players):
                    #         if i != ind_play and models_type[i] != "random":
                    #             players[i].actor.set_weights(players[ind_play].actor.get_weights())
                    #             players[i].critic.set_weights(players[ind_play].critic.get_weights())


                    #     players[1].save(f"{path_playing}/{round_}/m2_actor.ckpt",f"{path_playing}/{round_}/m2_critic.ckpt")    

                    for i in range(num_players):

                        if models_type[i] != "random":

                            (
                                observation_buffer,
                                action_buffer,
                                advantage_buffer,
                                return_buffer,
                                logprobability_buffer,
                            ) = buffer_players[i].get()
                            kl_temp = []
                            
                            if i == 1:
                                # print("here2")
                                loss_list = []
                                for _ in range(train_iterations):
                                # a_loss, c_loss = agent.train(saved_data_m2,gamma=0.99)
                                    kl, loss, e_loss = players[i].train_policy(
                                        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                                    )
                                    loss_actor.append(loss)
                                    kl_temp.append(kl)
                                    # players[i].train_value_function(observation_buffer, return_buffer)
                                    # print(kl)
                                    if kl > 1.5 * target_kl:
                                        print(f"break  KL {i}")
                                        break                    
                                # print(loss_list)
                                # print(np.min(loss_list))
                                    # print(f"Actor loss: {a_loss}, Critic loss: {c_loss}")
                                # print(kl_temp)
                                # print(np.mean(kl_temp))
                                kl_values.append(np.mean(kl_temp))
                                loss_list = []
                                for _ in range(train_iterations):    
                                    loss = players[i].train_value_function(observation_buffer, return_buffer)
                                    loss_critic.append(loss)
                                # print(np.min(loss_list))
                                # print(loss_list)
                    # players[global_best_model_play].save(f"{path_playing}_{round_}/m2_actor.ckpt",f"{path_playing}_{round_}/m2_critic.ckpt")        
      
                
                if np.max(points_counter) >= 52:
                    break_flag = True
                    # print(points_counter)
                    winner = np.argmax(points_counter)
                    
                    overall_winner.append(winner)
                    winner_count[winner] += 1
                    break
                    
            
            
            
            np_global_cumulative_wins = np.array(global_cumulative_wins)
            temp = []
            for i in range(num_players):
                if len(global_cumulative_wins) == 0:
                    if i == winner:
                        temp.append(1)
                    else:
                        temp.append(0)
                else:
                    if i == winner:
                        
                        temp.append(1+np_global_cumulative_wins[-1,i])
                    else:
                        temp.append(0+np_global_cumulative_wins[-1,i])                   
            
            global_cumulative_wins.append(temp)
                
            number_runs.append(runs)
                
                # average
            # for o in range(len((reward_m1))):
            #     average_reward.append(reward_m1[o]/runs)
               
    
            # average_reward = np.mean(reward_m1)

            average_points_player_mean = points_per_player/played_rounds_count+0.00001
            global_average_points_play.append(average_points_player_mean)
            global_average_reward_play.append(award_per_player/played_rounds_count+0.00001)  
            if trading:
                global_average_reward_trade.append(award_per_trader/traded_rounds_count+0.00001)   
            

        
        # print(f"Average max points: {np.mean(average_max_points)}")


        # print(f"BAD PLAY: {bad_play}, AVOIDED BAD PLAY: {avoided_bad_play}, BAD PLAY CONTROL: {bad_play_control}, AVOIDED BAD PLAY CONTROL: {avoided_bad_play_control}")
        print(f"Round {round_}, Winner: {np.unique(overall_winner,return_counts=True)[1]}")
        print(f"Player model: Number of chicagos: {chicago_count}, A nr rounds: {np.mean(number_runs)}, Average reward: {np.mean(global_average_reward_play,axis=0)}, Retrieved points: {np.sum(retreived_points)}")
        # print(f"Trader model: Retrieved points: {np.sum(retreived_points)}, lost points: {np.sum(missed_points)}, Average reward: {np.mean(global_average_reward_trade,axis=0)}")
        # print(f"REtrieved points: {np.sum(retreived_points)}, lost points: {np.sum(missed_points)}")            
        # if i % 50:
        # m1_main.save_weights(f"{path}\\m1_main.ckpt")
        # m1_temp.save_weights(f"{path}\\m1_temp.ckpt")
   

   
    
    # m2_main.save_weights(f"{path}\\m2_main.ckpt")
    # m2_temp.save_weights(f"{path}\\m2_temp.ckpt")

    
    scores_play, scores_trade = e.evaluate(400,players,num_players,models_type,traders,trading)
    weights = np.ones(num_players)
    for i in range(num_players):
        if models_type[i] == "random":
            weights[i] = 0
            
    ind_play = np.argmax(scores_play*weights)
    global_best_model_play = ind_play
    ind_trade = np.argmax(scores_trade*weights)
    global_best_model_trade = ind_trade
    print("-------------------------------")
    print("Choosing best model to continue")
    print(f"AVERAGE SCORE {scores_play} {scores_trade}")
    print("------------------------------")
    for i in range(len(players)):
        if i != ind_play and models_type != "random":
            players[i].actor.set_weights(players[ind_play].actor.get_weights())
            players[i].critic.set_weights(players[ind_play].critic.get_weights())
    if trading:
        for i in range(len(traders)):
            if i != ind_trade and models_type != "random":
                traders[i].actor.set_weights(traders[ind_trade].actor.get_weights())
                traders[i].critic.set_weights(traders[ind_trade].critic.get_weights())
    save_best_count = 0
    best_score_model = np.zeros(num_players)
    players[global_best_model_play].save(f"{path_playing}/best/m2_actor.ckpt",f"{path_playing}/best/m2_critic.ckpt")
    if trading:
        traders[global_best_model_trade].save(f"{path_trading}/best/m1_actor.ckpt",f"{path_trading}/best/m1_critic.ckpt")
    # try:
    # global_average_reward = np.stack(global_average_reward)
    # print(global_average_reward)

        # try:
        #     np.concatenate(new_global_average_reward,global_average_reward[i],
        # except:
            
        #     raise ValueError
    # print("---")
    # print(global_average_points)
    new_global_average_reward = []
    size = 75
    global_average_reward = np.stack(global_average_reward_play)
    for i in range(size,len(global_average_reward),size):

        new_global_average_reward.append(np.mean(global_average_reward[i-size:i],axis=0))
            

    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(len(new_global_average_reward)),new_global_average_reward,label=np.arange(num_players))
    ax.set_title("Reward play")
    ax.set_ylabel("Average reward")
    ax.set_xlabel("Round")
    ax.legend() 
    plt.show()

    if trading:
        new_global_average_reward = []
        size = 75
        global_average_reward = np.stack(global_average_reward_trade)
        for i in range(size,len(global_average_reward),size):
            new_global_average_reward.append(np.mean(global_average_reward[i-size:i],axis=0))
            
        fig, ax = plt.subplots(1,1)
        ax.plot(np.arange(len(new_global_average_reward)),new_global_average_reward,label=np.arange(num_players))
        ax.set_title("Reward trade")
        ax.set_ylabel("Average reward")
        ax.set_xlabel("Round")
        ax.legend() 
        plt.show()        

    # fig, ax = plt.subplots(1,1)
    # ax.plot(np.arange(len(global_average_points_play)),global_average_points_play)
    # ax.set_title("Points play")
    # ax.set_ylabel("Average points")
    # ax.set_xlabel("Round")
    # ax.legend() 
    # plt.show() 
    
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(len(kl_values)),kl_values)
    ax.plot(np.arange(len(kl_values)),np.zeros(len(kl_values))+(1.5 * target_kl))
    ax.set_title("KL")
    ax.set_ylabel("KL")
    ax.set_xlabel("X")
    ax.legend() 
    plt.show() 

    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(len(global_average_points_play)),global_cumulative_wins,label=np.arange(num_players))
    ax.set_title("Cum sum")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Wins")
    ax.legend() 
    plt.show()  
    # except:
    #     pass         

    
    print(winner_count)
    print(f"Average value play {np.mean(global_average_reward_play)}")
    print(f"Average value trade {np.mean(global_average_reward_trade)}")
    print(f"Average points {np.mean(global_average_points_play)}")
    
def train_util(identifier_play,model_type,train,past_identifier_play, trading = False, identifier_trade = None, past_identifier_trade= None):

    path_play = rf"saved_models/{identifier_play}/"
    
    if identifier_trade is not None:
        path_trade = rf"saved_models/{identifier_trade}"
    else:
        path_trade = None
    
    if past_identifier_play is not None:
        previous_path_play = rf"saved_models/{past_identifier_play}"
    else:
        previous_path_play = None
    if past_identifier_trade is not None:
        previous_path_trade = rf"saved_models/{past_identifier_trade}"
    else:
        previous_path_trade = None
    # if model_type == "RL":
    #     parh = rf"{path}\\RL"
    #     index = identifier_play
    # if past_identifier_trade is not None:
    #     previous_path_trade = rf"C:/Users/eamadlu/OneDrive - SCA Forest Products AB/Documents/Python/Chicago/saved_models/{past_identifier_trade}"
    # else:
    #     previous_path_play = None
    # if model_type == "RL":
    #     parh = rf"{path_trade}\\RL"
    #     index = identifier_trade


    # else:
    #     raise ValueError("Specified model is not available")

    if not os.path.exists(path_play):
        os.makedirs(path_play)

    
    if train:
        init = True
        models_type = ["random","ppo","random"]
        train_rl(1000,100,path_play, previous_path_play,  path_trade, previous_path_trade, 3, models_type, trading=trading)
        # for i in range(6):
        #     for j in range(2,6):
        #         if init:
        #             if j == 2:
        #                 train_rl(5,40,path, previous_path, j)
        #                 # raise ValueError
        #             else:
        #                 train_rl(int(30/j),40,path, previous_path, j)
        #             init = False
        #         else:
        #             if j == 2:
        #                 train_rl(5,40,path, path, j)
        #             else:
        #                 train_rl(int(30/j),40,path, path, j)
        
    else:
        return ""
        
    
    return 
