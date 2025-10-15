#%%
from xmlrpc.client import DateTime

import pandas as pd
import numpy as np
import glob

from fontTools.misc.bezierTools import namedtuple
from torch.cuda import device

input_files = glob.glob('data/train/input*.csv')
output_files = glob.glob('data/train/output*.csv')

input_df = pd.concat((pd.read_csv(f) for f in input_files), ignore_index=True)
output_df = pd.concat((pd.read_csv(f) for f in output_files), ignore_index=True)

pd.set_option('display.max_columns', None)
#%%
## Redefining columns

one_hot_columsns = ["play_direction", "player_position", "player_side", "player_role"]
input_df = pd.get_dummies(input_df, columns=one_hot_columsns)
input_df = input_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
output_df = output_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
#%%
## Transforming input data



## Age

year = 2025
input_df["player_birth_date"] = pd.to_datetime(input_df["player_birth_date"])
input_df["age"] = 2025 - input_df["player_birth_date"].dt.year

## Height to meters

def foot_to_meters(x:str):
    x = x.replace("-",".")
    meters = float(x) * 0.3048
    return meters

input_df["player_height"] = input_df["player_height"].apply(foot_to_meters)
input_df["player_height"] = pd.to_numeric(input_df["player_height"])
#%%
## Scaling Inputs

from sklearn.preprocessing import StandardScaler

scaled_columns= ["absolute_yardline_number", "player_height", "player_weight", "age", "s", "a", "dir", "o", "ball_land_x", "ball_land_y"]
scaler = StandardScaler()
input_df[scaled_columns] = scaler.fit_transform(input_df[scaled_columns])
#%%
## Defining feature columns

feature_columns = []
for c in input_df.columns:
    for columns in one_hot_columsns:
        if c.startswith(columns) and c not in one_hot_columsns:
            feature_columns.append(c)
feature_columns.append("absolute_yardline_number")
feature_columns.append("player_height")
feature_columns.append("player_weight")
feature_columns.append("age")
feature_columns.append("x")
feature_columns.append("y")
feature_columns.append("s")
feature_columns.append("a")
feature_columns.append("dir")
feature_columns.append("o")
feature_columns.append("ball_land_x")
feature_columns.append("ball_land_y")

label_columns = ["x", "y"]
#%%
sequence_groups = ["game_id", "play_id", "nfl_id"]
groups_input = input_df[input_df["player_to_predict"]].groupby(sequence_groups)
groups_output = output_df.groupby(sequence_groups)
max_input_sequence = groups_input.size().max()
max_output_sequence = groups_output.size().max()
#%%
## Extracting sequences in shape of [total_sequences, len_sequences, vector_length]


feature_dim = len(feature_columns)
out_dim = len(label_columns)
input_sequences = []
output_sequences = []

i = 0
for (game_id, play_id, nfl_id), frame in groups_input:
    i += 1
    if i % 1000 == 0:
        print(i)
    input_sequence = frame[feature_columns].to_numpy(dtype=np.float32)
    group = groups_output.get_group((game_id, play_id, nfl_id))
    output_sequence = group[label_columns].to_numpy(dtype=np.float32)
    input_sequences.append(input_sequence)
    output_sequences.append(output_sequence)
#%%
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple
from typing import NamedTuple

class DataSetEntry(NamedTuple):
    input_sequence: any
    output_sequences: any

class TrainDataset(Dataset):
    def __init__(self, input_sequences, how_many):
        self.input_sequences = input_sequences
        self.how_many = how_many

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.how_many[idx]


class SequenceDataset(Dataset):
    def __init__(self, input_sequences,output_sequences):
        self.sequences = input_sequences

        self.labels = output_sequences

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return DataSetEntry(self.sequences[idx], self.labels[idx])
#%%
import torch

def collate_fn(batch):
    max_len_in_seq = max(len(e.input_sequence) for e in batch)
    max_len_out_seq = max(len(e.output_sequences) for e in batch)
    lengths_in = []
    lengths_out = []
    input_seq = []
    output_seq = []
    x,y = 31,32
    last_coordinates = []
    for entry in batch:
        inputs = entry.input_sequence
        last_coordinates.append([inputs[-1][x], inputs[-1][y]])
        outputs = entry.output_sequences
        lengths_in.append(len(inputs))
        lengths_out.append(len(outputs))
        padding_in = max_len_in_seq - len(inputs)
        if padding_in > 0:
            inputs = np.concatenate([inputs, np.zeros((padding_in, inputs.shape[1]))])
        input_seq.append(inputs)
        padding_out = max_len_out_seq- len(outputs)
        if padding_out > 0:
            outputs = np.concatenate([outputs, np.zeros((padding_out, outputs.shape[1]))])
        output_seq.append(outputs)

    input_seq = torch.tensor(np.array(input_seq), dtype=torch.float32)
    output_seq = torch.tensor(np.array(output_seq), dtype=torch.float32)
    lengths_in = torch.tensor(np.array(lengths_in), dtype=torch.long)
    lengths_out = torch.tensor(np.array(lengths_out), dtype=torch.long)
    last_coordinates = torch.tensor(np.array(last_coordinates), dtype=torch.float32)
    return input_seq, output_seq, lengths_in, lengths_out, last_coordinates
#%%
dataset = SequenceDataset(input_sequences, output_sequences)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
#%%
batch = next(iter(dataloader))
#%%
in_seq, out_seq, lengths_in, lengths_out, last_cords = batch
#%%
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Net(nn.Module):
    def __init__(self, input_dim, emedding_dim, hidden_dim):
        super().__init__()

        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, emedding_dim),
        )

        self.warm_gru = nn.GRU(emedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        self.predict_gru = nn.GRUCell(2, hidden_dim)

        self.predict_cords = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, batch, device):
        if len(batch) == 4:
            in_seq, lengths_in, lengths_out, last_cords = batch
        else:
            in_seq, out_seq, lengths_in, lengths_out, last_cords = batch
        in_seq = in_seq.to(device)
        lengths_out = lengths_out.to(device)
        last_cords = last_cords.to(device)

        x = self.embedder(in_seq)
        x = pack_padded_sequence(x, lengths_in, batch_first=True, enforce_sorted=False)
        out, hidden = self.warm_gru(x)
        hidden = hidden.squeeze(0)
        predictions = []
        time_steps = lengths_out.max().item()

        for t in range(time_steps):
            # Schritt 1: aktive Sequenzen bestimmen
            active_mask = (t < lengths_out).float().unsqueeze(1)  # (B, 1)

            # Schritt 2: GRU-Schritt NUR für aktive Einträge
            hidden = self.predict_gru(last_cords, hidden)
            last_cords = self.predict_cords(hidden)

            # Schritt 3: Tote Sequenzen einfrieren
            hidden = hidden * active_mask + hidden.detach() * (1 - active_mask)
            last_cords = last_cords * active_mask  # padding bleibt 0, wenn inactive

            predictions.append(last_cords)
        predictions = torch.stack(predictions).permute(1,0,2)  # (T, B, 2)

        return predictions

#%%
model = Net(39, 32, 128)
#%%
input_seq, output_seq, lengths_in, lengths_out, last_coordinates = batch
#%%
print(input_seq.shape)
print(output_seq.shape)
#%%
import torch.optim

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(100000):
    total_loss = 0
    for batch in dataloader:
        out = model(batch, device)
        _, label, _,_,_= batch

        label = label.to(device)
        loss = F.mse_loss(out, label)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch", epoch)
    print("loss", total_loss)
#%%
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
predict = model(batch, device)
#%%

predict = predict.cpu().detach().numpy()
predict_x = predict[:,:,0]
predict_y = predict[:,:,1]
pd.DataFrame(columns=[predict_x, predict_y])
#%%
_,l,_,_,_= batch
#%%
l[:,:,0]
#%%
import pandas as pd

test_output = pd.read_csv('data/test.csv')
test_input = pd.read_csv('data/test_input.csv')
#%%
from functions import *

test_input = transform_df(test_input)
#%%
test_input[scaled_columns] = scaler.transform(test_input[scaled_columns])
#%%
player_position_K = []
player_position_LB = []
player_position_P = []
player_position_T = []

for i in range(len(test_input)):
    player_position_T.append(False)
    player_position_LB.append(False)
    player_position_P.append(False)
    player_position_K.append(False)

test_input["player_position_K"] = player_position_K
test_input["player_position_LB"] = player_position_LB
test_input["player_position_P"] = player_position_P
test_input["player_position_T"] = player_position_T

#%%
import torch

def train_collate_fn(batch):
    input_sequences, lengths = zip(*batch)
    max_len_in_seq = max(len(input_sequence) for input_sequence in input_sequences)

    lengths_in = []

    input_seq = []

    x,y = 31,32
    last_coordinates = []
    for inputs in input_sequences:
        last_coordinates.append([inputs[-1][x], inputs[-1][y]])

        lengths_in.append(len(inputs))

        padding_in = max_len_in_seq - len(inputs)
        if padding_in > 0:
            inputs = np.concatenate([inputs, np.zeros((padding_in, inputs.shape[1]))])
        input_seq.append(inputs)


    input_seq = torch.tensor(np.array(input_seq), dtype=torch.float32)
    lengths_in = torch.tensor(np.array(lengths_in), dtype=torch.long)
    lengths_out = torch.tensor(np.array(lengths), dtype=torch.long).squeeze(1)
    last_coordinates = torch.tensor(np.array(last_coordinates), dtype=torch.float32)
    return input_seq, lengths_in, lengths_out, last_coordinates
#%%
test_output
#%%
def single_player_trajectory(input_df, group_in, output_df, feature_columns):
    groups_input = input_df[input_df["player_to_predict"]].groupby(group_in)
    groups_output = output_df.groupby(group_in)
    how_many = []

    input_sequences = []

    i = 0
    for (game_id, play_id, nfl_id), frame in groups_input:
        i += 1
        if i % 1000 == 0:
            print(i)
        input_sequence = frame[feature_columns].to_numpy(dtype=np.float32)
        input_sequences.append(input_sequence)
        how_many.append(frame["num_frames_output"].unique())

    ids = (
            output_df["game_id"].astype(str) + "_" +
            output_df["play_id"].astype(str) + "_" +
            output_df["nfl_id"].astype(str) + "_" +
            output_df["frame_id"].astype(str)
    ).tolist()
    output_df = pd.DataFrame({"id": ids})
    return input_sequences, output_df, how_many
#%%
input_sequences, result_df, how_many = single_player_trajectory(test_input, sequence_groups, test_output, feature_columns)

#%%
dataset = TrainDataset(input_sequences, how_many)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=train_collate_fn)
#%%
batch = next(iter(dataloader))
#%%
model = Net(39, 32, 128)
model.eval()
model.load_state_dict(torch.load('checkpoints/single_player_traj/model1'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#%%
result_x = []
result_y = []

with torch.no_grad():
    for batch in dataloader:
        input_seq, lengths_in, lengths_out, last_coordinates = batch
        out = model(batch, device)        # erwartet dein Model dieses Tuple? Falls nur x: model(input_seq, ...)
        # out: (B, T_max, 2)

        out = out.cpu().numpy()
        lengths_out = lengths_out.cpu().numpy()

        # Nur die ersten length_i Schritte je Sample nehmen
        for i in range(out.shape[0]):
            t = int(lengths_out[i])
            result_x.extend(out[i, :t, 0])
            result_y.extend(out[i, :t, 1])
#%%
result_df["x"] = result_x
result_df["y"] = result_y
#%%
test_input[(test_input["game_id"] == 2024120805)][:300]
#%%
result_df