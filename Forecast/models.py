# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:46:49 2022

@author: eamadlu
"""
import pandas as pd
import numpy as np
# additional modules



# import scipy

import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow as tf
import utils as gp
import matplotlib.pyplot as plt
import time
import  scipy
# import tensorflow_addons as tfa
from pathlib import Path
import tensorflow as tf
from spektral.layers import GCNConv, GatedGraphConv, GraphSageConv, GATConv
root_dir = Path(__file__).resolve().parent


class GNN(tf.keras.Model):
    def __init__(self, OUT_STEPS, num_features, adjacency_matrix):
        super(GNN, self).__init__()
        self.a = adjacency_matrix
        units = 64
        self.gnn_layer4 = GatedGraphConv(units,2)
        self.gnn_layer2 = GCNConv(units)
        self.gnn_layer3 = GraphSageConv(units)
        self.gnn_layer1 = GATConv(units,2,activation="relu")
        # tf.keras.Sequential([
        #     # tf.keras.layers.Reshape([19, 24]),
        #     GCNConv(units)
        # ])
        self.flatten = tf.keras.layers.Flatten()
        self.lstm = tf.keras.layers.LSTM(16)
        self.dense = tf.keras.Sequential([
            # tf.keras.layers.Reshape([24, units])
            # tf.keras.layers.LSTM(16)
            
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # tf.keras.layers.Flatten(),
            # Shape => [batch, out_steps*features].
            # tf.keras.layers.Dense(OUT_STEPS*num_features,
            #                     kernel_initializer=tf.initializers.zeros()),
            # # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])
       
    def call(self, input):
        x = input
        # print(x.shape)
        # print(self.a_train.shape)
        # print(x.shape[2])
        # # x = tf.keras.layers.Reshape([x.shape[2], x.shape[1]])
        x = tf.transpose(x,[0,2,1])
        # # print(x.shape)
        x = self.gnn_layer1([x,self.a])
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        xx = self.lstm(input)
        # print(xx.shape)
        # raise ValueError
        x = tf.concat([x,xx],-1)

        # x = self.gnn_layer2([x,self.a])
        # print(x.shape)
        # x = tf.transpose(x,[0,2,1])
        # print(x.shape)
        x = self.dense(x)
        # print(x.shape)
        # x = self.dense(x)
        return x

class Transformer(tf.keras.Model):
    def __init__(self, OUT_STEPS, num_features):
        super(Transformer, self).__init__()
        self.out_steps = OUT_STEPS
        self.num_features = num_features
        self.embedding = tf.keras.layers.Dense(OUT_STEPS)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=6, key_dim=OUT_STEPS)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(OUT_STEPS)
        ])


        # self.ffn = tf.keras.layers.Dense(64, activation="relu")
            # tf.keras.layers.Dense(OUT_STEPS)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.out_dense = tf.keras.Sequential([
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])      
        ])
        self.dropout = tf.keras.layers.Dropout(0.1)
        # self.dropout2 = Dropout(rate)

    def positional_encoding(self,):

        pos = np.arange(self.out_steps)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.out_steps , 2) * -(np.log(10000) / self.out_steps ))

        pe = np.zeros((self.out_steps, self.out_steps))
        pe[:, 0::2] = np.sin(pos * div_term) # index is even
        pe[:, 1::2] = np.cos(pos * div_term) # index is odd

        return tf.cast(tf.expand_dims(pe, axis=0), dtype=tf.float32)

    def call(self, x, training):
        embed = x
        embed = self.embedding(x)
        # print(embed.shape)
        pos = self.positional_encoding()
        embed = embed + pos 
        # print(embed.shape)
        attn_output = self.att(embed,embed)
        x = attn_output
        x = self.dropout(x, training=training)
        x = self.layernorm1(embed + x)
        # x = x[:,-1,:]
        x = self.ffn(x)
        x = self.dropout(x, training=training)
        x = self.layernorm2(embed + x)
        x = x[:,-1,:]
        x = self.out_dense(x)
        return x

class Dense(tf.keras.Model):
    def __init__(self,OUT_STEPS, num_features):
        super(Dense, self).__init__()

        self.multi_step_dense = tf.keras.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])
    
    def call(self, x):
        x = self.multi_step_dense(x)
        return x
    
class CNNLSTM(tf.keras.Model):
    def __init__(self,OUT_STEPS, num_features):
        super(CNNLSTM, self).__init__()

        self.conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                kernel_size=(3,),
                                activation='relu'),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.LSTM(16,return_sequences=False),
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])   
    
    def call(self, x):

        x = self.conv_model(x)

        return x
class CNN(tf.keras.Model):
    def __init__(self,OUT_STEPS, num_features):
        super(CNN, self).__init__()

        self.conv_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -3:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(3)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                    kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])   
    
    def call(self, x):

        x = self.conv_model(x)

        return x

class LSTM(tf.keras.Model):
    def __init__(self,OUT_STEPS, num_features):
        super(LSTM, self).__init__()

        self.multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(16,return_sequences=False),
            # tf.keras.layers.LSTM(8),
            # tf.keras.layers.Dense(16,activation="relu"),
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # tf.keras.layers.Flatten(),
            # Shape => [batch, out_steps*features].
            # tf.keras.layers.Dense(OUT_STEPS*num_features,
            #                     kernel_initializer=tf.initializers.zeros()),
            # # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])
    
    def call(self, x):

        x = self.multi_lstm_model(x)

        return x
    
class Bi_LSTM(tf.keras.Model):
    def __init__(self,OUT_STEPS, num_features):
        super(Bi_LSTM, self).__init__()

        self.multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            # tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12,return_sequences=False)),
            # tf.keras.layers.Dense(32,activation="relu"),
            # tf.keras.layers.Flatten(),
            # Shape => [batch, out_steps*features].
            # tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # tf.keras.layers.Dense(OUT_STEPS*num_features,
            #                     kernel_initializer=tf.initializers.zeros()),
            # # # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])
    
    def call(self, x):
        x = self.multi_lstm_model(x)
        return x


	
def model_util(data_name, model_type, window, version, new, num_features,adjacency_matrix=None):


    path = rf"{root_dir}/saved_models/{data_name}"


    if model_type == "Transformer":
        model_name = rf"{path}/Transformenr"
        if version == "latest":
            index = gp.general_get_latest_index(model_name)
        else:
            index = version
            model = Transformer(window.label_width, num_features)
 
    elif model_type == "CNNLSTM":
        model_name = rf"{path}/CNNLSTM"
        if version == "latest":
            index = gp.general_get_latest_index(model_name)
        else:
            index = version
        model = CNNLSTM(window.label_width, num_features) 
    elif model_type == "CNN":
        model_name = rf"{path}/CNN"
        if version == "latest":
            index = gp.general_get_latest_index(model_name)
        else:
            index = version
        model = CNN(window.label_width, num_features)  
    elif model_type == "LSTM":
        model_name = rf"{path}/LSTM"
        if version == "latest":
            index = gp.general_get_latest_index(model_name)
        else:
            index = version
        model = LSTM(window.label_width, num_features)  
    elif model_type == "Bi_LSTM":
        model_name = rf"{path}/Bi_LSTM"
        if version == "latest":
            index = gp.general_get_latest_index(model_name)
        else:
            index = version
        model = Bi_LSTM(window.label_width, num_features)    

    elif model_type == "Dense":
        model_name = rf"{path}/Dense"
        if version == "latest":
            index = gp.general_get_latest_index(model_name)
        else:
            index = version
        model = Dense(window.label_width, num_features)  

    elif model_type == "GNN":
        model_name = rf"{path}/GNN"
        if version == "latest":
            index = gp.general_get_latest_index(model_name)
        else:
            index = version
        model = GNN(window.label_width, num_features, adjacency_matrix)    
    else:
        raise ValueError("Specified model is not available")

  
    
    gp.create_if_not_exist(model_name)
    
    if not new:
        model_name = rf"{model_name}\\{index}"
        model.load_weights(f"{model_name}\\.ckpt").expect_partial()
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanSquaredError()])
    else:
    # if model_type == "CNN" or model_type == "CNN_Repeat" or model_type == "LSTM" or model_type == "Dense_Repeat" or model_type == "Transformer" or model_type == "Bi_LSTM"
            
            model_name = rf"{model_name}\\{index}"
            # if int(version) == 1:

            # model.load_weights(f"{model_name}\\.ckpt").expect_partial()
            model.compile(loss=tf.keras.losses.MeanSquaredError(),
                            optimizer=tf.keras.optimizers.Adam(),
                            metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanSquaredError()])
    
            MAX_EPOCHS = 50
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                patience=2,
                                                                mode='min')
            # if model_type != "GNN":
            #     train_data = window.train
            #     validation_data = window.val
            # else:
            #     train_data = [window.train,a_matrix_train]
            #     validation_data = [window.val,a_matrix_val]
            train_data = window.train
            validation_data = window.val
            history = model.fit(train_data, epochs=MAX_EPOCHS,
                                validation_data=validation_data,
                                verbose=1,
                                shuffle=True,
                                callbacks=[early_stopping])           

            model.save_weights(f"{model_name}\\.ckpt")
          
                
            #---------------------------------------
            
            # print(history.history.keys())
            # plt.plot(history.history['loss'])
            # plt.plot(history.history['val_loss'])
            # plt.title('loss RULModel')
            # plt.ylabel('loss')
            # plt.xlabel('epoch')
            # plt.legend(['train', 'test'], loc='upper left')
            # plt.show()
 
          #

    return model
