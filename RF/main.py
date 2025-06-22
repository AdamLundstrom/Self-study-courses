# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:51:50 2023

@author: eamadlu
"""
import numpy as np
import tensorflow as tf



import random
import train as t
import time
import model as m
import evaluate as e
import utils as u

train = False
model_type = "RL"
# identifier_play = "v51/1"
# identifier_play = "v51/10"
# identifier_play = "v51/100"
# identifier_play = "v51/997"
identifier_play = "v55/998"

past_identifier_play = "v54/926"
# past_identifier_play = "random"

identifier_trade = "v2"
past_identifier_trade = "v1"
trading = False
# past_identifier = None
start = time.time()
####################################
#TRAIN
###################################
if train:
    t.train_util(identifier_play,model_type,train,past_identifier_play, trading, identifier_trade, past_identifier_trade)

####################################
#TEST
####################################
# PLayers
models_type = ["random","ppo","random"]
num_players = 3
path_play = rf"saved_models/{identifier_play}"
players = []
for i in range(num_players):
    if models_type[i] == "random":
        players.append(m.RandomAgent(5))
    else:
        if i == 0:
            players.append(m.Agent(5,activation="relu"))
        else:
            players.append(m.Agent(5,activation="relu"))
        players[-1].load(f"{path_play}/m2_actor.ckpt",f"{path_play}/m2_critic.ckpt")

# path_play = rf"saved_models/{identifier_play}"
# for p in players:
#     p.load(f"{path_play}/m2_actor.ckpt",f"{path_play}/m2_critic.ckpt")


# raise ValueError
# previous_path_play = rf"./saved_models/{past_identifier_play}"
# players[2].load(f"{previous_path_play}/m2_actor.ckpt",f"{previous_path_play}/m2_critic.ckpt")
# previous_path_play = rf"./saved_models/{past_identifier_play}"
# players[0].load(f"{previous_path_play}/m2_actor.ckpt",f"{previous_path_play}/m2_critic.ckpt")
    
# #Traders
traders = []

for i in range(num_players):
    if models_type[i] == "random":
        traders.append(m.RandomAgent(32))
    else:
        traders.append(m.Agent(32))


path_trade = rf"saved_models/{identifier_trade}"
for tr in traders:
    tr.load(f"{path_trade}/m1_actor.ckpt",f"{path_trade}/m1_critic.ckpt")

# if past_identifier_play == "random":
#     previous_player = "random"
    
# else:
#     previous_path_trade = rf"saved_models/{past_identifier_play}"

#     previous_trade = m.Agent(6)
#     previous_trade.load(f"{previous_path_trade}/m1_actor.ckpt",f"{previous_path_trade}/m1_critic.ckpt")

e.evaluate(1000,players,num_players,models_type,traders,trading=False,print_out = True)    

deck = (np.arange(52)+2).astype(float)
deck = u.shuffle(deck)
deck, players_cards = u.deal_cards(deck,num_players)

previous_cards_traid = []
for i in range(num_players-1):
    previous_cards_traid.append(np.zeros(15))
previous_cards_traid = u.assign_random_previous_cards(previous_cards_traid,deck,num_players-1)
# data = m.play_input_data(4,0,3,0,0,[14,13,53,27,40],np.zeros([6,5]),previous_cards_traid)
data = m.play_input_data(4,0,3,0,0,[53,52,40,27,14],np.zeros([6,5]),previous_cards_traid)
value_pred_test_critic1 = players[1].critic.predict(data,verbose=0)[0][0]
value_pred_test_actor1, _ = players[1].act(data)
# value_pred_test_actor1 = tf.nn.softmax(value_pred_test_actor1).numpy()
value_pred_test_actor1 = value_pred_test_actor1.numpy()

previous_cards = np.array([
    [2,8,15,0,0],
    [3,5,17,13,0],
    [25,26,27,39,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
    ])

# previous_cards = np.array([
#     [0,0,0,0,0],
#     [25,0,0,0,0],
#     [20,0,0,0,0],
#     [0,0,0,0,0],
#     [0,0,0,0,0],
#     [0,0,0,0,0]
#     ])
# data = m.play_input_data(4,0,3,3,13,[11,14,0,0,0],prev)

num_players = 3
chicago = 0
left = 0
index = 1
controling_card = 25
controling_card = 39
# cards = [14,13,0,0,0]
cards = [40,38,0,0,0]
# cards = [27,26,4,3,2]
# cards = [0,0,0,12,2]
print(u.get_card_values([40,38]))
print(u.get_color_values([23,38]))
weights = [0,0,0,0,1]
weights = [1,1,0,0,0]
data = m.play_input_data(num_players,chicago,left,index,controling_card,cards,previous_cards,previous_cards_traid)
value_pred_test_actor2, _ = players[1].act(data)
value_pred_test_actor2 = (value_pred_test_actor2).numpy()
value_pred_test_critic2 = players[1].critic.predict(data,verbose=0)[0][0]
print(players[1].pred(data,weights))
# print(f"Value of perfect cards1: {value_pred_test_critic1}, Value of perfect cards2: {value_pred_test_critic2}")
# print(f"Agent: {value_pred_test_actor1} Agent: {value_pred_test_actor2}")     

print(f"Value of perfect cards2: {value_pred_test_critic2}")
print(f"Agent: {value_pred_test_actor2}")
# end = time.time()
# print(end - start)
