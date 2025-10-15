import pandas as pd
import numpy as np

def foot_to_meters(x: str):
    x = x.replace("-", ".")
    meters = float(x) * 0.3048
    return meters

def transform_df(dataframe):
    ## Redefining columns

    one_hot_columsns = ["play_direction", "player_position", "player_side", "player_role"]
    input_df = pd.get_dummies(dataframe, columns=one_hot_columsns)
    input_df = input_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    ## Transforming input data

    ## Age

    year = 2025
    input_df["player_birth_date"] = pd.to_datetime(input_df["player_birth_date"])
    input_df["age"] = year - input_df["player_birth_date"].dt.year

    ## Height to meters



    input_df["player_height"] = input_df["player_height"].apply(foot_to_meters)
    input_df["player_height"] = pd.to_numeric(input_df["player_height"])
    return input_df

def scale(frame, cols, mean, std):
    frame[cols] = frame[cols].apply(lambda x: (x - mean) / std)
    return frame

def single_player_trajectory(input_df, group_in, output_df, feature_columns):
    groups_input = input_df[input_df["player_to_predict"]].groupby(group_in)
    groups_output = output_df.groupby(group_in)
    how_many = []

    input_sequences = []
    output_df = pd.DataFrame(columns=["id"])

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