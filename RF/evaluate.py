# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 07:44:58 2024

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
from scipy.special import softmax
import model as m
# config =tf.ConfigProto(
#         device_count={'GPU':0}
# )
# sess=tf.Session(config=config)


  
# def reward(points):
#     return points/52.0

def evaluate(number_rounds,models_play,num_players, models_type, models_trade=None, trading=False, print_out=False):

    model_state = "evaluate"    

    actions_taken = [np.zeros(32),np.zeros(32),np.zeros(32)]    
    
    play_actions_taken = [np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6)]
    play_actions_taken = [np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5)]

    state_size = 94
    action_size = 5
    update_after = 10

    winner_count = np.zeros(num_players)
    global_average_points_play = []
    global_average_reward_play = []
    global_cumulative_wins = []
    
    global_average_reward_trade = []

    overall_winner = []

    sucessful_chicago_count = np.zeros(num_players)
    chicago_count = np.zeros(num_players)
    
    players = []
    
    
    for i in range(len(models_play)):
        players.append(models_play[i])
        
    traders = []
    if trading:
        for i in range(len(models_trade)):
            traders.append(models_trade[i])
            
    for round_ in range(number_rounds):  

        number_runs = []
        
        missed_points = []
        retreived_points = []

        points_per_player = np.zeros(num_players)
        reward_per_player = np.zeros(num_players)

        reward_per_trader = np.zeros(num_players)
        
        points_counter = np.zeros(num_players)
        
        runs = 0
        
        break_flag = False
        played_rounds_count = 0
        traded_rounds_count = 0
                    
        while np.max(points_counter) < 52:
            ##############################
            #INIT VALUES FOR EACH GAME
            ##########################
            runs += 1
            deck = (np.arange(52)+2).astype(float)
            deck = u.shuffle(deck)
            deck, players_cards = u.deal_cards(deck,num_players)
            for i in range(num_players):
                players_cards[i] = -np.sort(-players_cards[i])
            previous_cards_traid = []
            # print(f"PLayers card init: {players_cards[i]}")
            # print(u.get_color_values(players_cards[1]))
            # print(u.get_card_values(players_cards[1]))            
            for i in range(num_players):
                previous_cards_traid.append(np.zeros(15))

            if not trading:
                previous_cards_traid = u.assign_random_previous_cards(previous_cards_traid,deck,num_players)
 
            if trading:
                
                best_points = 0
                for r in range(3):     

                    for i in range(num_players):
                        # print(i)
                        if points_counter[i] < 42:
                            points_init, _ = u.get_points(players_cards[i])
                                

                            action = m.trade(model_state, num_players, traders, i, r, points_init, points_counter, best_points, players_cards, previous_cards_traid)

                            for c in players_cards[i]:
                                previous_cards_traid[i-1] = np.roll(previous_cards_traid[i-1],1)
                                previous_cards_traid[i-1][0] = c


                            # if np.max(action) != 0:

                            if i == 1:
                                actions_taken[r][action] += 1                                

                            # if i > 0 and points_init > 3:
                            #     print("----")
                            #     print("Missed opertunity")
                            #     print(action)
                            #     print(i)
                            #     print(f"Change: {r}")
                            #     print(u.get_color_values(players_cards[i]))
                            #     print(u.get_card_values(players_cards[i]))  
                            #     print(points_init)
                            #     raise ValueError                                

                            # if i > 0 and action == 0:
                            #     print("----")
                            #     print("Catched oppertunity")
                            #     print(action)
                            #     print(i)
                            #     print(f"Change: {r}")
                            #     print(u.get_color_values(players_cards[i]))
                            #     print(u.get_card_values(players_cards[i]))  
                            #     print(points_init)
                            #     raise ValueError
                                

                            if action != 0:    
                                if i == 1 and points_init > 1: 
                                    missed_points.append(1)
                                    
                                deck, p = u.change_cards(deck,players_cards[i],action)
                                players_cards[i] = p
                                players_cards[i] = -np.sort(-players_cards[i])
                            
                            if action == 0 and points_init > 1:
                                retreived_points.append(1)
                            
                            points, _ = u.get_points(players_cards[i])
                            # if i == 1:
                            #     print("----")
                            #     print(f"Change: {r}")
                            #     print(f"ACTION: {action}")
                            #     print(u.get_color_values(players_cards[1]))
                            #     print(u.get_card_values(players_cards[1]))  
                            #     print(points)                            

                            if points_init < 42 and points == 42:
                                reward_per_trader[i] -= 3.
                            elif points_init < 42 and points == 43:
                                reward_per_trader[i] -= 2.
                            elif points_init < 42 and points == 44:
                                reward_per_trader[i] -= 1.

                    traded_rounds_count += 1
                    max_player, max_points = u.determine_point_winner(players_cards)
                    best_points = max_points
    
                    if max_player is not None:
                        points_counter[max_player] += max_points
                        reward_per_trader[max_player] += max_points
                    for j in range(num_players):
                        if j != max_player and points_counter[j] < 42:
                            reward_per_trader[j] += -1 
            
            # raise ValueError
            previous_cards_played = np.zeros([6,5])
            
            chicago_flag = 0

            players_cards_play = players_cards.tolist()   

            weights = np.zeros(5)+1
        
            #######################
            #CHICAGO EVALUATE
            #######################
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
            #     # print(i)

            #     index = 0
            #     left = num_players-1
            #     controling_card = 0
            #     # print(weights)
                
            #     selected = m.play(model_state, models_type[i], players, num_players, chicago_flag, i, left, index ,controling_card,
            #                       players_cards_play,previous_cards_played, previous_cards_traid,weights=weights)
                
            #     # print(selected)
            #     # raise ValueError
            #     # print(selected)
            #     if i == 1:
            #         play_actions_taken[0][selected] += 1  


            #     if selected == 5 and models_type[i] != "random":
            #         chicago_flag = 1
            #         chicago_player = i
            #         break      
                
                
            #     # raise ValueError


            #     # cards_temp = players_cards[i]
            #     # other_cards_temp = []
            #     # for j in range(len(players_cards)):
            #     #     if j != i:
            #     #         other_cards_temp.append(players_cards[i])
            #     # if u.posibility_of_chicago(cards_temp, other_cards_temp):
            #     #     print("---")
            #     #     print(players_cards_play[1])
            #     #     print(u.get_color_values(players_cards_play[1]))
            #     #     print(u.get_card_values(players_cards_play[1]))
            #     #     raise ValueError
                    
            #     #     reward_per_player[j] -= 1 
                              
            if chicago_flag:
                starter = chicago_player
                chicago_count[starter] += 1

            
            ##############################
            ##PLAY
            #############################
            break_flag = False 
            # weights[-1] = 0
            
            for i in range(5):
                # print("---")
                # print(players_cards_play[1])
                # print(u.get_color_values(players_cards_play[1]))
                # print(u.get_card_values(players_cards_play[1]))
                controling_card = 0.0

                played_cards = np.zeros(num_players)
                
                if i == 4:
                    selected = np.where(np.array(players_cards_play)[starter] > 0)[0][0]
                else:

                    weights_play = u.get_acceptable_play(controling_card, players_cards_play[starter])
                    # weights_play = weights*np.insert(weights_play,len(weights_play),1)        
                    weights_play = weights*weights_play 
                    selected = m.play(model_state, models_type[starter], players, num_players, chicago_flag, starter, num_players-1, i ,controling_card,
                                      players_cards_play,previous_cards_played, previous_cards_traid,weights=weights_play)

                    if starter == 1:
                        play_actions_taken[i][selected] += 1  
                # print(starter)
                # print(selected)
                # print(players_cards_play)
                # print(previous_cards_played)
                
                # print(previous_cards_traid)
                # print("-------")
                # print(starter)
                
                controling_card = players_cards_play[starter][selected]
                
                # if starter == 1:
                #     print(f"Started with card: {selected}")
                
                init_bad_play = True
                if controling_card == 0:
                    print(f"PLAYER {starter}")
                    print(weights_play)
                    print(controling_card)
                    print(selected)
                    print(players_cards_play)
                    print(players_cards)
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
                                    left = len(order_play)-left_count-1
                                    selected = m.play(model_state, models_type[j], players, num_players, chicago_flag, j, left, i ,controling_card,
                                                      players_cards_play,previous_cards_played, previous_cards_traid,weights=weights_play)
                                    # if j == 1:
                                    #     print(f"Selected card: {selected}")

                                elif init_bad_play and not init_bad_play2:
                                    print("---Bad play")
                                    print(f"PLAYER {j}")
                                    print(weights_play)
                                    print(controling_card)
                                    print(selected)
                                    print(players_cards_play)
                                    print(players_cards)
                                    
                                    raise ValueError
                                    
                          
                                a_play = u.acceptable_play(players_cards_play[j][selected], players_cards_play[j],controling_card)                

                                if j == 1:
                                    play_actions_taken[i+1][selected] += 1  

                        played_cards[j] = players_cards_play[j][selected]
                        previous_cards_played[j][i] = played_cards[j]
                        players_cards_play[j][selected] = 0.0

                        # print(previous_cards_played)
                        left_count += 1
                        if break_flag:
                            break                            

                # weights[5-i-1] = 0
                
                if break_flag:
                    break                     
                starter = u.local_winner(played_cards, controling_card)
                
                if chicago_flag and starter != chicago_player:
                    break
 
            played_rounds_count += 1
            winner = starter
            # print(f"Winner: {winner}")
            # raise ValueError

            if chicago_flag and chicago_player != winner:
                points_counter[winner] -= 0
                points_per_player[winner] -= 0
                reward_per_player[winner] += 0.5
                points_counter[chicago_player] -= 15
                points_per_player[chicago_player] -= 15            
                reward_per_player[chicago_player] -= 3
                reward_per_trader[chicago_player] -= 3 
        
            elif chicago_flag and chicago_player == winner:
                sucessful_chicago_count[winner] += 1
                points_counter[winner] += 15   
                points_per_player[winner] += 15
                reward_per_player[chicago_player] += 3
                reward_per_trader[chicago_player] += 1 
   
            else:    
                points_counter[winner] += 5
                points_per_player[winner] += 5
                reward_per_player[winner] += 1
                reward_per_trader[winner] += 5
                
            # for i in range(num_players):
            #     if i != winner:
            #         reward_per_player[i] -= 5      

            max_player, max_points = u.determine_point_winner(players_cards)

            if max_player is not None:
                points_counter[max_player] += max_points
                reward_per_trader[max_player] += max_points
            for j in range(num_players):
                if j != max_player and points_counter[j] < 42:
                    reward_per_trader[j] -= 1 
                        
                        
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
        # print(points_per_player)
        average_points_player_mean = points_per_player/played_rounds_count
        global_average_points_play.append(average_points_player_mean)
        
        average_reward_player_mean = reward_per_player/played_rounds_count
        global_average_reward_play.append(average_reward_player_mean)
        

        # global_average_reward_trade.append(reward_per_trader/traded_rounds_count)
            # global_average_reward.append(average_reward)         
                    
            

    # fig, ax = plt.subplots(1,1)
    # ax.plot(np.arange(len(global_average_points)),global_cumulative_wins,label=np.arange(num_players))
    # ax.set_title("Cum sum")
    # ax.set_xlabel("Rounds")
    # ax.set_ylabel("Wins")
    # ax.legend() 
    # plt.show()  

    # for i in range(3):
    #     fig, ax = plt.subplots(1,1)
    #     ax.bar(np.arange(32),actions_taken[i],label=f"Round {i}")
    #     ax.set_title("Trade Actions/round")
    #     ax.set_xlabel("Action")
    #     ax.set_ylabel("Amount")
    #     ax.legend() 
    #     plt.show() 

    for i in range(5):
        fig, ax = plt.subplots(1,1)
        ax.bar(np.arange(5),play_actions_taken[i],label=f"Round {i}")
        ax.set_title("Play Actions/round")
        ax.set_xlabel("Action")
        ax.set_ylabel("Amount")
        plt.show()               

    if print_out:
        print(f"Winner count: {winner_count}")
        print(f"Chicago count: {chicago_count}")
        print(f"Sucessful chicago count: {sucessful_chicago_count}")
        print(f"Average points per player: {np.mean(np.array(global_average_points_play),axis=0)}")
        print(f"Average reward per player: {np.mean(np.array(global_average_reward_play),axis=0)}")
        # print(f"Average reward per trader: {np.mean(np.array(global_average_reward_trade),axis=0)}")
        
    # print(f"Average value {np.mean(global_average_reward)}")
    # print(f"Average points {np.mean(global_average_points)}")
    
    return np.mean(np.array(global_average_reward_play),axis=0), np.mean(np.array(global_average_reward_trade),axis=0)
    
