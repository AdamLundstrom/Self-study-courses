import os
import datetime

import IPython
import IPython.display

import numpy as np
import pandas as pd

import tensorflow as tf
import utils as utils
import models as model_utils
import random
import typing
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
# csv_path, _ = os.path.splitext(zip_path)
# df = pd.read_csv(csv_path)
# df.to_csv("data.csv",index=False)

def Random(seed,flag=False):
    random.seed(seed)
    np.random.seed(seed) 
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)

    # tf.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    print(f"Random seed set as {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)

Random(1)

df = pd.read_csv("data.csv")

# print(df.head())

# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame.
df['wv (m/s)'].min()

wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

timestamp_s = date_time.map(pd.Timestamp.timestamp)

day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

column_indices = {name: i for i, name in enumerate(df.columns)}

# print(df.columns)
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# corr_train = np.corrcoef(train_df.to_numpy().T)

np_data = train_df.to_numpy()
print(np_data.shape)
# raise
corr_train = np.zeros([np_data.shape[1],np_data.shape[1]])
for i in range(len(corr_train)):
    for j in range(len(corr_train)):

        A = np_data[:,i]
        B = np_data[:,j]
        dot_product = np.dot(A, B)
        magnitude_A = np.linalg.norm(A)
        magnitude_B = np.linalg.norm(B)
        corr_train[i,j] = dot_product / (magnitude_A * magnitude_B)

# corr_val = np.corrcoef(val_df.to_numpy().T)
# corr_test = np.corrcoef(test_df.to_numpy().T)
# print(corr_train.shape)

plt.figure(figsize=(8, 8))
plt.matshow(np.corrcoef(train_df.to_numpy().T), 0)
# plt.matshow(corr_train, 0)
plt.xlabel("road number")
plt.ylabel("road number")
plt.show()


class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes


sigma2 = 0.1
epsilon = 0.5
adjacency_matrix = utils.compute_adjacency_matrix(corr_train, sigma2, epsilon)

# plt.figure(figsize=(8, 8))
# print(adjacency_matrix)
# plt.matshow(adjacency_matrix, 0)
# plt.xlabel("road number")
# plt.ylabel("road number")
# plt.show()
# raise
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")
# raise

# corr_train = tf.data.Dataset.from_tensor_slices(corr_train)
# corr_val = tf.data.Dataset.from_tensor_slices(corr_val)


# LABEL_WIDTH = 24
# CONV_WIDTH = 3
# INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
# window = utils.WindowGenerator(
#     input_width=INPUT_WIDTH,
#     label_width=LABEL_WIDTH,
#     shift=1,train_df=train_df,
#     val_df=val_df,test_df=test_df,label_columns=['T (degC)'])

# multi_window = utils.WindowGenerator(
#     input_width=24, label_width=24, shift=24,train_df=train_df,
#     val_df=val_df,test_df=test_df,label_columns=['T (degC)'])

width = 36
# multi_window = utils.WindowGenerator(
#     input_width=width, label_width=width, shift=width,train_df=train_df,
#     val_df=val_df,test_df=test_df)

multi_window = utils.WindowGenerator(
    input_width=width, label_width=width, shift=width,train_df=train_df,
    val_df=val_df,test_df=test_df,label_columns=['T (degC)'])

if multi_window.label_columns is None:
    num_features = len(multi_window.column_indices)
else:
   num_features = len(multi_window.label_columns)


# print(wide_window.column_indices)
# print(len(wide_window.column_indices))
# print(wide_window.split_window(wide_window.train_df))
# print(multi_window.)
# print('Input shape:', multi_window.example[0].shape)
# print('Output shape:', multi_window.[0]).shape)
# raise

for example_inputs, example_labels in multi_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

adjacency_matrix = tf.sparse.from_dense(adjacency_matrix)
# print(adjacency_matrix)
# raise

model_types = ["GNN","Transformer","Dense","CNNLSTM","CNN","LSTM","Bi_LSTM"]
# model_types = ["GNN","Transformer"]
model_types = ["GNN"]
# model_types = ["Transformer"]
performance = {}
save = False
for model_type in model_types:
    print(model_type)
    new=True
    model = model_utils.model_util("multi_output", model_type, multi_window, 1, new, num_features, adjacency_matrix)
    # print('Input shape:', multi_window.example[0].shape)
    # print('Output shape:', model(multi_window.example[0]).shape)
    # raise
    # pred = model.predict(multi_window.test)
    performance[model_type] = model.evaluate(multi_window.test, verbose=0, return_dict=True)
    # print(model.evaluate(multi_window.test, verbose=0, return_dict=True))
    # print(model.summary())
    # multi_window.plot(model)

if multi_window.label_columns is None:
    lab = "All channels"
else:
    lab = 'Only T (degC)'
print(performance)
x = np.arange(len(performance))
width_ = 0.4
metric_name = 'mean_absolute_error'
test_mae = [v[metric_name] for v in performance.values()]
print(test_mae)
fig, ax = plt.subplots(layout='constrained')
ax.set_ylabel(f'MAE {lab}')
rects = ax.bar(x, test_mae, width_)
ax.bar_label(rects,fmt='%.3f')
ax.set_xticks(ticks=x, labels=performance.keys(),
           rotation=45)
# _ = plt.legend()
if save:
    file = rf"C:\Users\eamadlu\OneDrive - SCA\Documents\Skoldokument\Sensorkurs\MAE_{lab}_{width}"
    plt.savefig(f'{file}.png',dpi=600, bbox_inches='tight',transparent=True)
else:
    plt.show()

fig, ax = plt.subplots(layout='constrained')
metric_name = 'root_mean_squared_error'
test_mae = [v[metric_name] for v in performance.values()]
ax.set_ylabel(f'RMSE {lab}')
rects = ax.bar(x, test_mae, width_)
ax.bar_label(rects,fmt='%.3f')
ax.set_xticks(ticks=x, labels=performance.keys(),
           rotation=45)
# _ = plt.legend()
if save:
    file = rf"C:\Users\eamadlu\OneDrive - SCA\Documents\Skoldokument\Sensorkurs\RMSE_{lab}_{width}"
    plt.savefig(f'{file}.png',dpi=600, bbox_inches='tight',transparent=True)
else:
    plt.show()

fig, ax = plt.subplots(layout='constrained')
metric_name = 'mean_squared_error'
test_mae = [v[metric_name] for v in performance.values()]
ax.set_ylabel(f'MSE {lab}')
rects = ax.bar(x, test_mae, width_)
ax.bar_label(rects,fmt='%.3f')
ax.set_xticks(ticks=x, labels=performance.keys(),
           rotation=45)
# _ = plt.legend()
if save:
    file = rf"C:\Users\eamadlu\OneDrive - SCA\Documents\Skoldokument\Sensorkurs\MSE_{lab}_{width}"
    plt.savefig(f'{file}.png',dpi=600, bbox_inches='tight',transparent=True)
else:
    plt.show()
