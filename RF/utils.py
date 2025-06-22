# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:51:50 2023

@author: eamadlu
"""
import numpy as np
import tensorflow as tf
import random
import math
def shuffle(cards):    
    np.random.shuffle(cards)
    return cards


def get_card_values(cards):
    card_values = []
    for i in range(len(cards)):
        remainer = (cards[i]//15)
        subtractor = 13*remainer 
        card_values.append((cards[i]-subtractor))
        
    card_values = np.array(card_values)    
    card_values[card_values > 14] -= 13

    return card_values
# print(get_card_values([45., 28., 19., 17.,  4.]))
def get_color_values(cards):     
    color_values = np.copy(cards)
    color_values[(color_values > 0) & (color_values <= 14)] = 1
    color_values[(color_values > 14) & (color_values <= 27)] = 2
    color_values[(color_values > 27) & (color_values <= 40)] = 3
    color_values[(color_values > 30)] = 4

    return color_values    

def get_points(cards):
    points = 0
    color_values = get_color_values(cards)
    card_values = get_card_values(cards)
    order = -np.sort(-card_values)
    
    c = np.zeros(len(card_values))
    for i in range(len(card_values)):
        for j in range(len(card_values)):
            if card_values[i] == card_values[j] and i != j:
                c[i] += 1
                
    straight = True
    sorted_values = -np.sort(-card_values)
    for i in range(1,len(sorted_values)):
        # print(sorted_values[i-1])
        # print(sorted_values[i]+1)
        if sorted_values[i-1] != sorted_values[i]+1:
            straight = False

    # print(straight)
    # raise ValueError
    color = False
    
    if len(np.unique(color_values)) == 1:
        color = True

    if color and straight and np.max(card_values) == 14:
        points = 52.0
    elif color and straight:
        points = 8.0
    elif np.max(c) == 4:
        points = 7.0
    elif np.sum(c) == 8:
        points = 6.0
    elif color:
        points = 5.0
    elif straight:
        points = 4.0
    elif np.max(c)-5 == 3:
        points = 3.0
    elif np.sum(c) == 4:
        points = 2.0
    elif np.sum(c) == 2:
        points = 1.0
            
    return points, order

#Input 2d
# @tf.function
def determine_point_winner(cards):
    # print(cards)
    # raise ValueError
    point_per_player = []
    card_sorting = []
    player = None
    

    for i in range(len(cards)):

        a,b = get_points(cards[i])
        point_per_player.append(a)
        card_sorting.append(b)
        

    max_points = np.max(point_per_player)
    
    if max_points == 0:
        return player, 0
    else:
        max_player = np.argmax(point_per_player)
        saved = []
        for i in range(len(cards)):
            if point_per_player[i] == max_points:
                saved.append(i)
                
        if len(saved) == 0:
            return max_player, max_points
        
        else:

            for k in range(5):
                highest = -1
                count = 0
                for i in range(len(saved)):
                    if card_sorting[saved[i]][k] >= highest:
                        if card_sorting[saved[i]][k] == highest:
                            count += 1
                        else:
                            count = 0
                        highest = card_sorting[saved[i]][k]
                        max_player = saved[i]

                if count == 0:
                    return max_player, max_points
 
            #SHOULD BE DETERMINED BY COLOR
            return max_player, max_points

# @tf.function
def change_cards__(cards,current_cards,desicion):
    # print(len(current_cards))
    # print(number)
    # raise ValueError
    # new_cards = []
    
    # for i in range(len(cards)):
    #     new_cards.append(cards[i])
        
    # new_current_cards = []
    # for i in range(len(current_cards)):
    #     new_current_cards.append(current_cards[i])
        
    for i in range(5):

        if desicion[i] == 1:
            # print("YES")
            card_to_get = cards[-1]
            cards = np.delete(cards,-1)
            cards = np.insert(cards,0,current_cards[i])
            current_cards[i] = card_to_get
    
    
            # card_to_get = cards[-1]
            # del new_cards[-1]
            # new_cards.insert(0,current_cards)
            # # cards = cards.write(0,current_cards[i]).stack()
            # new_current_cards[i] = card_to_get
            
    # cards = tf.stack(cards)
    # new_current_cards = tf.stack(new_current_cards)
    # print(cards)
    # raise ValueError
    return cards, current_cards


def change_cards_(cards,current_cards,desicion):

    
    if desicion[0] == 1:
        pass
    elif desicion[1] == 1:         
        for i in range(5):
            card_to_get = cards[-1]
            cards = np.delete(cards,-1)
            cards = np.insert(cards,0,current_cards[i])
            current_cards[i] = card_to_get
    elif desicion[2] == 1:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[0])
        current_cards[0] = card_to_get
    elif desicion[3] == 1:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[1])
        current_cards[1] = card_to_get
    elif desicion[4] == 1:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[2])
        current_cards[2] = card_to_get  
    elif desicion[5] == 1:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[3])
        current_cards[3] = card_to_get  
    elif desicion[6] == 1:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[4])
        current_cards[4] = card_to_get 
    else: 

        if desicion[7] == 1:
            selected = [0,1]
        elif desicion[8] == 1:
            selected = [0,2]            
        elif desicion[9] == 1:
            selected = [0,3]    
        elif desicion[10] == 1:
            selected = [0,4] 
        elif desicion[11] == 1:
            selected = [1,2] 
        elif desicion[12] == 1:
            selected = [1,3] 
        elif desicion[13] == 1:
            selected = [1,4] 
        elif desicion[14] == 1:
            selected = [2,3] 
        elif desicion[15] == 1:
            selected = [2,4] 
        elif desicion[16] == 1:
            selected = [3,4] 
        elif desicion[17] == 1:
            selected = [0,1,2] 
        elif desicion[18] == 1:
            selected = [0,1,3] 
        elif desicion[19] == 1:
            selected = [0,1,4] 
        elif desicion[20] == 1:
            selected = [0,2,3] 
        elif desicion[21] == 1:
            selected = [0,2,4] 
        elif desicion[22] == 1:
            selected = [0,3,4] 
        elif desicion[23] == 1:
            selected = [1,2,3] 
        elif desicion[24] == 1:
            selected = [1,2,4] 
        elif desicion[25] == 1:
            selected = [1,3,4] 
        elif desicion[26] == 1:
            selected = [2,3,4] 
        elif desicion[27] == 1:
            selected = [0,1,2,3] 
        elif desicion[28] == 1:
            selected = [0,2,3,4] 
        elif desicion[29] == 1:
            selected = [0,1,2,4] 
        elif desicion[30] == 1:
            selected = [0,1,3,4] 
        elif desicion[31] == 1:
            selected = [1,2,3,4] 
        for i in selected:
            card_to_get = cards[-1]
            cards = np.delete(cards,-1)
            cards = np.insert(cards,0,current_cards[i])
            current_cards[i] = card_to_get
        
    return cards, current_cards

def change_cards(cards,current_cards,desicion):

    if desicion == 0:
        pass
    elif desicion == 1:         
        for i in range(5):
            card_to_get = cards[-1]
            cards = np.delete(cards,-1)
            cards = np.insert(cards,0,current_cards[i])
            current_cards[i] = card_to_get
    elif desicion == 2:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[0])
        current_cards[0] = card_to_get
    elif desicion == 3:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[1])
        current_cards[1] = card_to_get
    elif desicion == 4:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[2])
        current_cards[2] = card_to_get  
    elif desicion == 5:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[3])
        current_cards[3] = card_to_get  
    elif desicion == 6:
        card_to_get = cards[-1]
        cards = np.delete(cards,-1)
        cards = np.insert(cards,0,current_cards[4])
        current_cards[4] = card_to_get 
    else: 

        if desicion == 7:
            selected = [0,1]
        elif desicion == 8:
            selected = [0,2]            
        elif desicion == 9:
            selected = [0,3]    
        elif desicion == 10:
            selected = [0,4] 
        elif desicion == 11:
            selected = [1,2] 
        elif desicion == 12:
            selected = [1,3] 
        elif desicion == 13:
            selected = [1,4] 
        elif desicion == 14:
            selected = [2,3] 
        elif desicion == 15:
            selected = [2,4] 
        elif desicion == 16:
            selected = [3,4] 
        elif desicion == 17:
            selected = [0,1,2] 
        elif desicion == 18:
            selected = [0,1,3] 
        elif desicion == 19:
            selected = [0,1,4] 
        elif desicion == 20:
            selected = [0,2,3] 
        elif desicion == 21:
            selected = [0,2,4] 
        elif desicion == 22:
            selected = [0,3,4] 
        elif desicion == 23:
            selected = [1,2,3] 
        elif desicion == 24:
            selected = [1,2,4] 
        elif desicion == 25:
            selected = [1,3,4] 
        elif desicion == 26:
            selected = [2,3,4] 
        elif desicion == 27:
            selected = [0,1,2,3] 
        elif desicion == 28:
            selected = [0,2,3,4] 
        elif desicion == 29:
            selected = [0,1,2,4] 
        elif desicion == 30:
            selected = [0,1,3,4] 
        elif desicion == 31:
            selected = [1,2,3,4] 
            
 
        for i in selected:
            card_to_get = cards[-1]
            cards = np.delete(cards,-1)
            cards = np.insert(cards,0,current_cards[i])
            current_cards[i] = card_to_get
        
    return cards, current_cards


# @tf.function
def deal_cards(cards,num_players):
    players_cards = np.zeros([num_players,5])
    # print(cards)
    for i in range(num_players):
        for j in range(5):
            players_cards[i,j] = cards[-1]
            
            # print()
            cards = np.delete(cards,-1)        
    return cards, players_cards

def weighted_probability(deck, current_cards,action):
    #If possible and porbabilty is ok then do it
    pass

def local_winner(played_cards, controling_card):
    colors = get_color_values(played_cards)    
    values = get_card_values(played_cards)   
    controling_color = get_color_values([controling_card])[0]
    highest = 0
    winner = -1
    for i in range(len(played_cards)):
        if colors[i] == controling_color and values[i] > highest:
            highest = values[i]
            winner = i
    return winner

def acceptable_play(played_card, players_cards,controling_card):

    if played_card == 0.0:
        return False
    controling_color = get_color_values([controling_card])[0]
    played_color = get_color_values([played_card])[0] 

    if played_color == controling_color:
        return True

    else:
        colors = get_color_values(players_cards)
        if controling_color in colors:

            return False
        else:

            return True
                
def random_player_game(cards,controling_card=None):
    # return random.randint(0,num_cards)
    decisions = np.zeros(5)

    cards_copy = np.array(cards).copy()

    if controling_card is not None:

        controling_color = get_color_values([controling_card])[0]

        cards_color = get_color_values(cards_copy)

        for i in range(len(cards_copy)):

            if cards_color[i] != controling_color:
                if cards_copy[i] != 0:
                    cards_copy[i] = -1.0

    
    if np.max(cards_copy) > 0.0:
        for i in range(len(decisions)):
            
            if cards_copy[i] > 0.0:
                decisions[i] = np.random.uniform(0,1,1)[0]   
        
    else:
        for i in range(len(decisions)):
            if cards_copy[i] != 0.0:
                decisions[i] = np.random.uniform(0,1,1)[0] 

    return decisions

def get_acceptable_play(controling_card,cards):
    selectable = np.ones(5)

    for i in range(len(cards)):
        if cards[i] == 0:
            selectable[i] = 0
    if controling_card == 0:
        # for i in range(len(cards)):
        #     if cards[i] == 0:
        #         selectable[i] = 0
            
        return selectable

    # ind = np.where(np.array(cards) > 0.0)[0]
    # ordered_cards = np.zeros(len(cards))
    # for i in range(len(ind)):
    #     ordered_cards[i] = cards[ind[i]]

        
    controling_color = get_color_values([controling_card])[0]
    # print("åååå")
    # print(controling_color)
        
    

    colors = get_color_values(cards)
    # print(colors)
    # print("äääää")

    if controling_color not in colors:

        return selectable
    
    for i in range(len(colors)):
        if colors[i] != controling_color:
            selectable[i] = 0
            
    return selectable

def random_player_change_cards_():
    # amount = random.randint(0, 5)
    # return random.sample(range(5), amount)
    decisions = np.zeros(5)
    for i in range(len(decisions)):
        decisions[i] = np.random.uniform(0,1,1)[0]

    return decisions

def random_player_change_cards():
    decisions = random.randint(0,31)
    return decisions

def random_player_chicago():
    number = random.randint(0, 100)
    if number < 1:
        return True
    else:
        return False

def assign_random_previous_cards(previous_cards, deck, num_players):
    deck_temp = np.array(deck).copy()
    previous_cards = np.array(previous_cards)
    
    num_cards = random.randint(0,12)

    if num_cards*num_players > len(deck_temp)-1:
       num_cards = math.floor((len(deck_temp)-1)/num_players)


    for i in range(num_players):
        selected_cards = np.random.choice(deck_temp, num_cards, replace=False)

        for j in range(len(selected_cards)):
            selected_cards_indice = np.where(deck_temp == selected_cards[j])[0]
            deck_temp = np.delete(deck_temp,selected_cards_indice)            
            previous_cards[i][j] = selected_cards[j] 
            
    
    return previous_cards

# def chicago_winning(cards):
#     auto_winn = False
    
# def winning_hand(cards):
#     if     
    
# def check_possibility_of_winning():
#     pass

def posibility_of_chicago(cards,other_cards):
    values = get_card_values(cards)
    colors = get_color_values(cards)
    other_cards = np.array(other_cards)
    colors_other = np.zeros(other_cards.shape)
    values_other = np.zeros(other_cards.shape)
    possible = True
    for i in range(len(other_cards)):
        colors_other[i] = get_color_values(other_cards[i])
        values_other[i] = get_card_values(other_cards[i])
        
    for i in range(len(colors)):
        for j in range(len(other_cards)):
            if colors[i] == colors_other[j][i] and values[i] < values_other[j][i]:
                possible = False
                break
            
        if not possible:
            break
        

    return possible        

def posibility_of_certain_chicago(cards):
    values = get_card_values(cards)
    colors = get_color_values(cards)
    possible = False
    
    if len(np.unique(colors)) == 1:
        if values[0] == 14 and values[1] == 13 and values[2] > 9:
            return True
    if min(values) == 13:
        return True
        
    prev_color = 0
    init_color = False
    prev_value = 0
    # for i in range(len(cards)):
        
    #     if not init_color and values[i] != 14:
    #         # print(values[i])
            
    #         # print("here")
    #         return False
    
            
    #     elif not init_color and values[i] == 14:
    #         # print(cards[i])
    #         init_color = True
    #         prev_color = colors[i]
    #         prev_value = values[i]
            
    #     elif init_color and colors[i] == prev_color and values[i] != prev_value-1:
    #         return False
            
    #     elif init_color and colors[i] != prev_color:
    #         # print("here1")
    #         # print(cards[i])
    #         # print("----")
    #         init_color = True
        
    return possible 

def posibility_of_winning(cards,other_cards,starter):
    values = get_card_values(cards)
    colors = get_color_values(cards)
    other_cards = np.array(other_cards)
    colors_other = np.zeros(other_cards.shape)
    values_other = np.zeros(other_cards.shape)
    possible = True
    for i in range(len(other_cards)):
        colors_other[i] = get_color_values(other_cards[i])
        values_other[i] = get_card_values(other_cards[i])
        
    for i in range(len(other_cards)):
        for j in range(len(other_cards)):
            if colors[i] == colors_other[j][i] and values[i] < values_other[j][i]:
                possible = False
                break
            
        if not possible:
            break
        

    return possible    

def check_for_chicago(num_players):
    chicago_flag = False
    chicago_player = -1
    for i in range(num_players):
        if random_player_chicago():
            chicago_player = i
            chicago_flag = True
            return chicago_player, chicago_flag
        
    return -1, False

def descriptive_to_int(color, value):
    return (color*13) + value    

# players_cards = [[14., 19., 3.,  4., 53.],[15., 19., 3.,  4., 9.]]

# cards = [14,13,12,53,51]
# cards = -np.sort(-np.array(cards))
# values = get_card_values(cards)
# colors = get_color_values(cards)
# print(cards)
# print(values)
# print(colors)
# print(posibility_of_certain_chicago(cards))
# print(cards)
# print(get_card_values(cards))
# other_cards = [[5,15,25,35,52],[4,14,39,36,49]]
# for i in range(len(other_cards)):
#     other_cards[i] = -np.sort(-np.array(other_cards[i]))
# print(posibility_of_chicago(cards,other_cards))
# print(get_points(players_cards[0]))
# print(determine_point_winner(players_cards))

# deck = (np.arange(52)+2).astype(float)
# deck = shuffle(deck)
# deck, players_cards = deal_cards(deck,4)

# previous_cards = [np.zeros(15),np.zeros(15),np.zeros(15),np.zeros(15)]

# previous_cards = assign_random_previous_cards(previous_cards, deck, 4)


