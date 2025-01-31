import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def load_csv(file_path, drop_columns=None):
    df = pd.read_csv(file_path)
    if drop_columns:
        drop_columns_from_dataframe(df, drop_columns)
    return df


def merge_dataframes(base, *args, **kwargs):
    if not args:
        return base
    return base.merge(merge_dataframes(*args), **kwargs)


def drop_columns_from_dataframe(df, columns):
    df.drop(columns=columns, inplace=True)


def combine_columns(df, new_column, base_columns):
    df[new_column] = df[base_columns].agg(' '.join, axis=1)


def load_drivers():
    drivers_df = load_csv('./data/drivers.csv', ['driverRef', 'number', 'code', 'dob',
                                                 'nationality', 'url'])
    combine_columns(drivers_df, 'full_name', ['forename', 'surname'])
    drop_columns_from_dataframe(drivers_df, ['forename', 'surname'])
    return drivers_df


def load_circuits():
    return load_csv('./data/circuits.csv', ['circuitRef', 'lat', 'lng', 'alt', 'url',
                                            'location', 'country'])


def load_races():
    df = load_csv('./data/races.csv', ['year', 'round', 'name', 'time', 'fp2_time',
                                       'fp3_time', 'quali_time', 'url', 'fp1_date',
                                       'fp1_time', 'fp2_date', 'fp3_date',
                                       'quali_date', 'sprint_date', 'sprint_time'])
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df.drop(columns=['date'], inplace=True)
    return df


def load_constructors():
    return load_csv('./data/constructors.csv', ['constructorRef', 'url', 'nationality'])


def load_qualifying():
    qualifying_df = load_csv('./data/qualifying.csv', ['qualifyId', 'number', 'q1', 'q2',
                                                       'q3', 'constructorId'])
    qualifying_df.rename(columns={'position': 'start_position'}, inplace=True)
    return qualifying_df


def load_results():
    return load_csv('./data/results.csv', ['grid', 'positionText', 'positionOrder',
                                           'points', 'laps', 'time', 'milliseconds',
                                           'fastestLap', 'rank', 'fastestLapTime',
                                           'fastestLapSpeed', 'statusId', 'resultId'])


def cleanup_data(df):
    drop_columns_from_dataframe(
        df, ['number', 'driverId', 'raceId', 'constructorId'])
    column_rename = {'name_x': 'circuit', 'name_y': 'constructor', 'full_name': 'name',
                     'position': 'result'}
    df.rename(columns=column_rename, inplace=True)
    df['start_position'].fillna(value=10, inplace=True)
    df.replace('\\N', np.nan, inplace=True)
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df['start_position'] = pd.to_numeric(df['start_position'], errors='coerce')
    df.dropna(subset=['result'], inplace=True)


def load_and_process_data():
    drivers_df = load_drivers()
    circuits_df = load_circuits()
    races_df = load_races()
    races_df = races_df.merge(
        circuits_df, on='circuitId').drop(columns=['circuitId'])
    constructors_df = load_constructors()
    qualifying_df = load_qualifying()
    results_df = load_results()

    dfs_to_merge = [(qualifying_df, ['raceId', 'driverId']), (races_df, 'raceId'),
                    (drivers_df, 'driverId'), (constructors_df, 'constructorId')]

    for df, on in dfs_to_merge:
        results_df = results_df.merge(df, on=on, how='left')

    cleanup_data(results_df)

    return results_df


def create_and_fit_encoder(df, columns):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[columns])
    return encoder


def apply_one_hot_encoding(encoder, df, columns):
    encoded_data = encoder.transform(df[columns])

    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out(columns))

    df_reset = df.reset_index(drop=True)
    encoded_df_reset = encoded_df.reset_index(drop=True)

    df_reset = df_reset.drop(columns, axis=1)
    df_combined = pd.concat([df_reset, encoded_df_reset], axis=1)

    return df_combined


def scale_numeric_features(data, columns):
    scalers = {}
    for col in columns:
        scaler = StandardScaler()
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        scalers[col] = scaler
    return scalers


def convert_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)


def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])
