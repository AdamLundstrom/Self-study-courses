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
from scipy.special import softmax

# config =tf.ConfigProto(
#         device_count={'GPU':0}
# )
# sess=tf.Session(config=config)




class critic(tf.keras.Model):
  def __init__(self,activation="relu"):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(64,activation=activation)
    self.d2 = tf.keras.layers.Dense(64,activation='relu')
    # self.d3 = tf.keras.layers.Dense(64,activation='relu')
    self.v = tf.keras.layers.Dense(1, activation = None)

  def call(self, input_data):
    x = self.d1(input_data)
    x = self.d2(x)
    # x = self.d3(x)
    v = self.v(x)
    return v
    

class actor(tf.keras.Model):
  def __init__(self,actor_desicions,activation="relu"):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(64,activation=activation)
    self.d2 = tf.keras.layers.Dense(64,activation='relu')
    # self.d3 = tf.keras.layers.Dense(64,activation='relu')
    self.a = tf.keras.layers.Dense(5,activation='softmax')
    # self.a2 = tf.keras.layers.Dense(actor_desicions,activation='softmax')
    # self.a3 = tf.keras.layers.Dense(actor_desicions,activation='softmax')
    # self.a4 = tf.keras.layers.Dense(actor_desicions,activation='softmax')
    # self.a = tf.keras.layers.Dense(actor_desicions,activation=None)

  def call(self, input_data):

    x = self.d1(input_data)
    x = self.d2(x)
    # x = self.d3(x)
    x = self.a(x)
    return x



class RandomAgent():
    def __init__(self,actor_desicions):
        self.decision_space = actor_desicions

    def load(self, name1,name2):
        return self

    def save(self, name1,name2):
        pass

    # Sample action from actor
    def act(self,observation,weights=None):
        
        logits = np.random.dirichlet(np.ones(self.decision_space),size=1)
        #Means that it is the trader
        if self.decision_space > 6:
            if weights is not None:
                logits = weights*logits
            action = np.argmax(logits)
        else:
            if weights is not None:
                # weights[-1]*=0.5
                logits = weights*logits
                
            action = np.argmax(logits)

                
        return logits, action
    
    def pred(self,observation,weights=None):
        logits, action = self.act(observation,weights)
        return logits
    
    
class Agent():
    def __init__(self,actor_desicions,activation="relu"):
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
        # self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # self.c_opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
        self.actor = actor(actor_desicions,activation)
        self.critic = critic(activation)
        self.clip_ratio = 0.2
        self.actor_desicions = actor_desicions

    def load(self, name1,name2):
        self.actor.load_weights(name1).expect_partial()
        self.critic.load_weights(name2).expect_partial()

    def save(self, name1,name2):
        self.actor.save_weights(name1)
        self.critic.save_weights(name2)
          
    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)

        # logprobabilities_all = tf.nn.log_softmax(logits)
        # logits = tf.nn.softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.actor_desicions) * logits, axis=1
        )
        
        return logprobability+ 1e-10
    
    
    # Sample action from actor
    # @tf.function
    def act(self,observation,weights=None):
        # print(observation)
        # print("-action--")
        logits = self.actor(observation)+1.0e-20
        # logits = tf.nn.softmax(logits)+1.0e-20
        # +1.0e-20
        # tf.random.categorical
        # +1.0e-20
        # print(logits)
        # print(logits)
        # print(weights)
        # weights[0][5] = 0
        # ind = np.where(weights[0] == 0)
        # weights[0][ind] = -100
        if weights is not None:
            logits *= tf.cast(weights,dtype=tf.float32)
        
        
        ind = tf.where(logits[0] != 0 )
        ind = tf.reshape(ind,[tf.size(ind)])
        
        
        action = ind[int(tf.squeeze(tf.random.categorical(tf.gather(tf.math.log(logits) , ind, axis=1), 1), axis=1))]
        # action = logits = tf.nn.softmax(logits)
        # print(logits)
        # action = int(tf.squeeze(tf.random.categorical(logits, 1), axis=1))
        # print(action)
        # print(weights)
        # print(logits)
        # print(action)
        # print("äääää''")
        # raise ValueError
        # try:

        #     action = ind[int(tf.squeeze(tf.random.categorical(tf.gather(logits , ind, axis=1), 1), axis=1))]
        # except:
        #     print(logits)
        #     print(weights)
        #     print(action)
        #     # print(weights)
        #     # print(logits)
        #     # raise ValueError
            
        #     print(tf.where(weights[0] != 0 ))
        #     action = tf.where(weights != 0 )[0]
        #     raise ValueError
            
            
        # print(action)
        # print("----")
        return logits, action
    
    
    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(self,
        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):
        tf.config.run_functions_eagerly(False)
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            policy = self.actor(observation_buffer)
            ratio = tf.exp(
                self.logprobabilities(policy, action_buffer)
                - logprobability_buffer
            )
            
            min_advantage = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            # min_advantage = tf.where(
            #     advantage_buffer > 0,
            #     (1 + self.clip_ratio) ,
            #     (1 - self.clip_ratio) * advantage_buffer,
            # )
            
            # print(min_advantage)
            # policy_loss = -tf.reduce_mean(
            #     tf.minimum(ratio * advantage_buffer, min_advantage)
            # )            
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage* advantage_buffer)
            )
            entropy_bonus = tf.reduce_mean(policy * tf.math.log(policy + 1e-10))
            policy_loss -= 0.01 * entropy_bonus 
            # print(policy_loss)
            
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.a_opt.apply_gradients(zip(policy_grads, self.actor.trainable_variables))
    
        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        # print(kl)
        # print(tf.reduce_sum(kl))
        # raise ValueError
        # kl = tf.reduce_sum(kl)

        return kl, policy_loss, entropy_bonus
    
    
    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
            # entropy_bonus = tf.reduce_mean(policy * tf.math.log(policy + 1e-10))
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.c_opt.apply_gradients(zip(value_grads, self.critic.trainable_variables))
        return value_loss
    
    def pred(self, observation,weights=None):
        logits = self.actor(observation).numpy()[0]+1.0e-20
        # logits = tf.nn.softmax(logits)
        # weights[5] = 0
        if weights is not None:
            logits *= weights
        return logits


def normalize_card_value(value):
    return value/53.0

def normalize_card_value_seperated(value):
    return value/14.0

def normalize_color_value_seperated(value):
    return value/4.0

def normalize_color_one_hot_color(value,card):
    out = np.zeros(4)
    if value > 0:
        out[int(value)-1] = normalize_card_value_seperated(card)
    return out

def normalize_color_one_hot_all(cards):
    out = np.zeros(52)
    for i in range(len(cards)):
        if cards[i] > 0:
            out[int(cards[i])-2] = 1
    return out

def normalize_points2(value):
    return value/8.0
    
def normalize_points(points):
    return points/52.0

def normalize_turn(turn):
    return turn/3.0

def normalize_one_hot(cards):
    out = np.zeros(52)
    for i in range(len(cards)):
        
        if cards[i] >= 2:
            out[int(cards[i])-2] = 1
    return out
    

def change_input_data(num_players,turn,cards,previous_cards,turn_points,all_points, best_points):

    data = []
    out = np.zeros(6)
    out[num_players-1] = 1
    for o in out:
        data.append(o)

    out = np.zeros(3)
    out[turn] = 1
    for o in out:
        data.append(o)

    ordered_cards = np.zeros(5)
    ind = np.where(np.array(cards) > 0.0)[0]
    
    for i in range(len(ind)):
        ordered_cards[i] = cards[ind[i]]
            
    for i in range(len(cards)):
        out = normalize_color_one_hot_color(u.get_color_values([ordered_cards[i]]),u.get_card_values([ordered_cards[i]]))
        for o in out:
            
            data.append(o)
    #     print("-----")
    #     print(out)
    # raise ValueError
    cards = np.array(previous_cards).flatten()
    out = normalize_color_one_hot_all(cards)
    for o in out:
        data.append(o)

    data.append(float(normalize_points(turn_points)))
    data.append(float(normalize_points(all_points)))
    data.append(float(normalize_points(best_points)))

    data = np.array(data).flatten()

    return data.reshape([1,-1])    


#ADD ALL  POINTS
def play_input_data(num_players,chicago,left,index,controling_card,cards,previous_cards,previous_cards_traid):
    data = []
  
    data.append(chicago)

    out = np.zeros(6)
    out[num_players-1] = 1
    for o in out:
        data.append(o)

    out = np.zeros(5)
    out[index] = 1
    for o in out:
        data.append(o)

    out = np.zeros(6)
    out[left] = 1
    for o in out:
        data.append(o)

    out = normalize_color_one_hot_color(u.get_color_values([controling_card]),u.get_card_values([controling_card]))
    for o in out:
        data.append(o)
    

    ordered_cards = np.zeros(5)
    ind = np.where(np.array(cards) > 0.0)[0]
    
    for i in range(len(ind)):
        ordered_cards[i] = cards[ind[i]]

    colors_temp = u.get_color_values(ordered_cards)
            

    for i in range(len(cards)):
        out = normalize_color_one_hot_color(u.get_color_values([ordered_cards[i]]),u.get_card_values([ordered_cards[i]]))
        for o in out:
            data.append(o)

    
    cards = np.array(previous_cards).flatten()
    cards2 = np.array(previous_cards_traid).flatten()
    np.hstack([cards,cards2])
    out = normalize_color_one_hot_all(cards)
    for o in out:
        data.append(o)


    data = np.array(data).flatten()


    return data.reshape([1,-1])    

def chicago_input_data(num_players,cards,previous_cards_traid):
    data = []

    out = np.zeros(6)
    out[num_players-1] = 1
    for o in out:
        data.append(o)
    
    ordered_cards = np.zeros(5)
    ind = np.where(np.array(cards) > 0.0)[0]
    for i in range(len(ind)):
        ordered_cards[i] = cards[ind[i]]
    for i in range(len(cards)):
        out = normalize_color_one_hot_color(u.get_color_values([ordered_cards[i]]),u.get_card_values([ordered_cards[i]]))
        for o in out:
 
            data.append(o)

    cards = np.array(previous_cards_traid).flatten()

    out = normalize_color_one_hot_all(cards)
    for o in out:
        data.append(o)

    data = np.array(data).flatten()


    return data.reshape([1,-1]) 

def trade(model_state, num_players, traders, player, turn, turn_points, all_points, best_points, players_cards, previous_cards, local_reward_m1=None, local_saved_data_m1=None,  buffers=None, mini_batch_size_1=None):

    if model_state == "train": 
        actual_player = player

        data = change_input_data(num_players,turn,players_cards[actual_player],previous_cards[player],turn_points,all_points[actual_player],best_points)
    
        
        local_saved_data_m1[player] = []
        local_reward_m1[player] = 0
        
        local_saved_data_m1[player].append(data.squeeze())
        logits, action = traders[player].act(data)
        action = np.argmax(action)
    
        local_saved_data_m1[player].append(traders[player].logprobabilities(logits,action))
        local_saved_data_m1[player].append(action) 
        local_saved_data_m1[player].append(traders[player].critic(data))
        
        return action, local_reward_m1, local_saved_data_m1

    else:
        local_reward_m1 = None
        local_saved_data_m1 = None
        
    
        actual_player = player

    
        data = change_input_data(num_players,turn,players_cards[actual_player],previous_cards[player],turn_points,all_points[actual_player],best_points)
    
    
        pred = traders[player].pred(data)
      
        # if player == 0:
        #     print(f"ACTOR PRED FOR NOT TRADE: {pred[0]}")
        #     print(f"CRITC PRED: {models[player].critic(data).numpy()[0]}")
        action = np.argmax(pred)
        
    return action


def play(model_state, model_type, players, num_players, chicago, player, left, index, controling_card, players_cards_play, previous_cards_played, previous_cards_traid, init=None, local_reward_m2=None, local_saved_data_m2=None, weights=None, buffers=None, mini_batch_size_2=None):

    if model_state == "train":
        if np.max(weights) == 0:
            raise ValueError
            
        actual_player = player
       
        data = play_input_data(num_players, chicago, left, index, controling_card, players_cards_play[actual_player],previous_cards_played,previous_cards_traid[player])
    
        # print(init)
        # print(player)
        if not init[player]:
    
            local_saved_data_m2[player].append(local_reward_m2[player])
            local_reward_m2[player] = 0
            # print(local_saved_data_m2[player])
            # raise ValueError
            if model_type != "random":
                if buffers[player].pointer < mini_batch_size_2-2:
                    try:
                        buffers[player]
                        local_saved_data_m2[player][4]
                        local_saved_data_m2[player][3]
                        local_saved_data_m2[player][2]
                        local_saved_data_m2[player][1]
                        buffers[player].store(local_saved_data_m2[player][0],local_saved_data_m2[player][1],local_saved_data_m2[player][2],local_saved_data_m2[player][3],local_saved_data_m2[player][4])
                    
                    except:
                        print("BUFFER FAULT MODEL")
                        print(player)
                        print(buffers[player].pointer)
                        raise ValueError
                    
            local_saved_data_m2[player] = []
        else:
            local_reward_m2[player] = 0
        
        local_saved_data_m2[player].append(data.squeeze())
        # local_saved_data_m2.append(np.array(weights).copy())
        weights = np.expand_dims(np.array(weights),0)
        logits, action = players[player].act(data,weights)
        action = int(action)
        if model_type == "random":
            local_saved_data_m2[player].append(0)
            local_saved_data_m2[player].append(0) 
            local_saved_data_m2[player].append(0)
        else:
            local_saved_data_m2[player].append(players[player].logprobabilities(logits,action))
            local_saved_data_m2[player].append(action) 
            local_saved_data_m2[player].append(players[player].critic(data))
    
        if action == 5:
            # print(logits)
            # raise ValueError
            return action, local_reward_m2, local_saved_data_m2
        
    
        # ind = np.where(np.array(players_cards_play[actual_player]) != 0)[0]
        # for i in range(len(ind)):
        #     if i == action:
        #         action = ind[i] 
        #         break
        
        return action, local_reward_m2, local_saved_data_m2
    
    else:
        actual_player = player

        data = play_input_data(num_players, chicago, left, index, controling_card, players_cards_play[actual_player],previous_cards_played,previous_cards_traid)
        # print("----")
        # print(weights)
        pred = players[player].pred(data,weights)

        # print(pred[-1])
        action = np.argmax(pred)
        
        # print(action)

        # ind = np.where(np.array(players_cards_play[actual_player]) != 0)[0]
        # for i in range(len(ind)):
        #     if i == action:
        #         action = ind[i] 
        #         break
        # print(action)
        # print("----")
        return action

