#%%
from xmlrpc.client import DateTime

import pandas as pd
import numpy as np
import glob

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
input_df.head()
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
groups_input = input_df.groupby(sequence_groups)
groups_output = output_df.groupby(sequence_groups)
max_input_sequence = groups_input.size().max()
max_output_sequence = groups_output.size().max()
#%%
keys_in = set(groups_input.groups.keys())
keys_out = set(groups_output.groups.keys())

print("Nur in inputs:", len(keys_in - keys_out))
print("Nur in outputs:", len(keys_out - keys_in))
#%%

#%%
## Extracting sequences in shape of [total_sequences, len_sequences, vector_length]


feature_dim = len(feature_columns)
out_dim = len(label_columns)

input_sequences = []
output_sequences = []
last_entries = []
real_input_lengths = []
real_output_lengths = []

for i, (key, group) in enumerate(groups_input):
    if i > 10:
        break
    seq = group[feature_columns].to_numpy(dtype=np.float32)
    last_entries.append(seq[-1])
    in_seq_len = len(seq)
    real_input_lengths.append(in_seq_len)
    if in_seq_len <max_input_sequence:
        padding = np.zeros(shape=(max_input_sequence - len(seq), feature_dim), dtype=np.float32)
        seq = np.concatenate((seq, padding), axis=0)
    input_sequences.append(seq)

    out_group = groups_output.get_group(key)
    out_seq = out_group[label_columns].to_numpy(dtype=np.float32)
    out_seq_len = len(out_seq)
    if out_seq_len < max_output_sequence:
        padding = np.zeros(shape=(max_input_sequence - len(out_seq), out_dim), dtype=np.float32)
        out_seq = np.concatenate((out_seq, padding), axis=0)
    real_output_lengths.append(out_seq_len)
    output_sequences.append(out_seq)


#%%
output_df[output_df["game_id"] == 2023090700  &(output_df["nfl_id"] == 43290)]
#%%
output_df["nfl_id"].dtype
#%%
print(sequences[10].shape)
#%%
print(sequences[1].shape)