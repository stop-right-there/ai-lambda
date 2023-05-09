import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

random.seed(42)

def scaleCoordinates(data):
    new_data = data.copy()
    new_data["center_latitude"] /= 10
    new_data["center_longitude"] /= 10
    return new_data

def dropSparseDatas(data):
    new_data = data.drop(["landfall_or_passage_indicator", "max_sustained_wind_speed", "central_pressure"], axis=1)

    idx = new_data["grade"]=="Just entering into the responsible area of RSMC Tokyo-Typhoon Center"
    new_data = new_data.drop(new_data[idx].index)
    return new_data

def vectorizeGrade(data):
    new_data = data.replace("Tropical Depression", "TD")
    new_data = new_data.replace("Severe Tropical Storm", "STS")
    new_data = new_data.replace("Tropical Storm", "TS")
    new_data = new_data.replace("Extra-tropical Cyclone", "L")
    new_data = new_data.replace("Typhoon", "TY")
    grade_map = {"TD":1, "TS":2, "STS":3, "TY":4, "L":1}  # I made grade map using the starndard for grade Tropical Depressions of WMO
    new_data["grade"] = new_data["grade"].map(grade_map)
    return new_data

def filterObservations(data, observation_interval=6):
    return data[data["hour"] % observation_interval == 0]

def concatDayAndMonth(data):
    new_data = data.copy()
    new_data["day"] = new_data["day"]/31
    new_data["month"] = new_data["month"]+new_data["day"]
    new_data = new_data.drop(["day"], axis=1)
    return new_data

def groupTyphoons(data):
    ids = data["intl_number_id"].unique() # Unique IDs of typhoons
    typhoons = list()
    for ID in ids:
        typhoon = data[data["intl_number_id"] == ID]
        typhoon = typhoon.drop(["intl_number_id"], axis=1)
        typhoons.append(typhoon)
    print(len(typhoons), "Typhoons Exists")
    return typhoons

def calculateUpdateTime(data):
    typhoons = data.copy()
    for typhoon in typhoons:
        typhoon["hours_after_latest_update"] = np.zeros(typhoon.shape[0])
        for i in range(1, typhoon.shape[0]):
            typhoon["hours_after_latest_update"].iloc[i] = typhoon["hour"].iloc[i] - typhoon["hour"].iloc[i-1]
            if typhoon["hours_after_latest_update"].iloc[i] < 0:
                typhoon["hours_after_latest_update"].iloc[i] += 24
    return typhoons

def makeTimeSeries(typhoons, n_sample, interval, augment_level):
    dataset = np.empty((0, n_sample, typhoons[0].shape[1]))
    test_typhoons = np.empty((0, n_sample, typhoons[0].shape[1]))  # Preserve Some typhoons for model evaluation

    for typhoon in tqdm(typhoons):
        nptyphoon = np.array(typhoon)
        for i in range(0, nptyphoon.shape[0]-n_sample, interval):
            if augment_level>0 and typhoon.iloc[0]["year"]!=2022 and random.random() < 0.5:  # Randomly Skip one row
                single_data_a = nptyphoon[i].reshape(1, -1)
                single_data_b = nptyphoon[i+2:i+1+n_sample]
                single_data = np.concatenate([single_data_a, single_data_b], axis=0)
                single_data = np.expand_dims(single_data, axis=0)
                dataset = np.append(dataset, single_data, axis=0)

            if augment_level>1 and typhoon.iloc[0]["year"]!=2022 and random.random() < 0.5:  # Randomly Skip one row from behind
                single_data_a = nptyphoon[i:i+n_sample-1]
                try:
                    single_data_b = nptyphoon[i+n_sample].reshape(1, -1)
                    single_data = np.concatenate([single_data_a, single_data_b], axis=0)
                    single_data = np.expand_dims(single_data, axis=0)
                    dataset = np.append(dataset, single_data, axis=0)
                except:
                    pass

            if augment_level>2 and typhoon.iloc[0]["year"]!=2022 and random.random() < 0.5:
                try:
                    single_data_a = nptyphoon[i].reshape(1, -1)
                    single_data_b = nptyphoon[i+2].reshape(1, -1)
                    single_data_c = nptyphoon[i+4].reshape(1, -1)
                    single_data = np.concatenate([single_data_a, single_data_b, single_data_c], axis=0)
                    single_data = np.expand_dims(single_data, axis=0)
                    dataset = np.append(dataset, single_data, axis=0)
                except:
                    pass
            
            single_data = nptyphoon[i:i+n_sample]
            single_data = np.expand_dims(single_data, axis=0)
            if typhoon.iloc[0]["year"]==2022:
                test_typhoons = np.append(test_typhoons, single_data, axis=0)
                continue
            dataset = np.append(dataset, single_data, axis=0)
    return dataset, test_typhoons

def split_x_y(dataset):
    x = dataset[:, :-1, :]
    y = dataset[:, -1, :]
    return x, y

def reorderColumns(data):
    idx = list([0, 1, 2, 3, 4, 5, -1])
    idx.extend(list(range(6, data[0].shape[1]-1)))
    new_data = data.copy()
    for i, d in enumerate(new_data):
        new_data[i] = d.iloc[:, idx]
    return new_data

def dropUnusedColumns(data):
    cols = ["apparent_temperature", "cloudcover_mid", "cloudcover_low"]
    idx = list()
    idx.extend(cols)
    for i in range(0, 316, 45):
        for c in cols:
            idx.append(c + "_" + str(i))
    return data.drop(idx, axis=1)

def loadData(observation_hour_interval=None, n_sample=3, data_sample_interval=1, path_data=False, reload=False, ignore_nan=False, augment=0):
    if reload == False and os.path.isfile("./pickle/path_data.pickle") and path_data:
        with open("./pickle/path_data_aug_"+str(augment)+".pickle", "rb") as fr:
            pick = pickle.load(fr)["data"]
        return pick["x_train"], pick["y_train"], pick["x_test"], pick["y_test"]
    elif reload == False and os.path.isfile("./pickle/storm_data.pickle"):
        with open("./pickle/storm_data_aug"+str(augment)+".pickle", "rb") as fr:
            pick = pickle.load(fr)["data"]
        return pick["x_train"], pick["y_train"], pick["x_test"], pick["y_test"]

        
    data = pd.read_csv("./data/meteoDataSet_final_fixed.csv", index_col=0, dtype={"landfall_or_passage_indicator":str})

    data = scaleCoordinates(data)
    data = dropSparseDatas(data)
    data = dropUnusedColumns(data)
    data = vectorizeGrade(data)
    data = concatDayAndMonth(data)

    if observation_hour_interval:
        data = filterObservations(data, observation_hour_interval)
    
    if path_data:  # filter out TD datas 
        data = data.drop(['dir_longest_radius_50kt_or_greater', 'longest_radius_50kt_or_greater', 'shortest_radius_50kt_or_greater', 'dir_longest_radius_30kt_or_greater', 'longest_radius_30kt_or_greater', 'shortest_radius_30kt_or_greater'], axis=1)
    else:
        idx = data["dir_longest_radius_50kt_or_greater"].isna()
        data = data.drop(data[idx].index)

    data = groupTyphoons(data)
    data = calculateUpdateTime(data)
    data = reorderColumns(data)

    train_data, test_data = makeTimeSeries(data, n_sample, data_sample_interval, augment)

    x_train, y_train = split_x_y(train_data)
    x_test, y_test = split_x_y(test_data)
    
    # x_train = x_train.reshape(-1, x_train.shape[2]*2)
    # x_test = x_test.reshape(-1, x_test.shape[2]*2)

    data = {"x_train": np.array(x_train), "y_train": np.array(y_train),
            "x_test": np.array(x_test), "y_test": np.array(y_test)}
    
    pick = {"data": data,
            "params": {
                "observation_hour_interval":observation_hour_interval,
                 "n_sample":n_sample,
                 "data_sample_interval":data_sample_interval,
                 "ignore_nan":ignore_nan
            }}
    if path_data:
        with open("./pickle/path_data_aug_"+str(augment)+".pickle", "wb") as fw:
            pickle.dump(pick, fw)
    else:
        with open("./pickle/storm_data_aug_"+str(augment)+".pickle", "wb") as fw:
            pickle.dump(pick, fw)
    
    if ignore_nan:
        x_train = np.nan_to_num(x_train, nan=np.nanmean(x_train))
        x_test = np.nan_to_num(x_test, nan=np.nanmean(x_test))

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = loadData(path_data=True, reload=True, augment=1)
    x_train, y_train, x_test, y_test = loadData(path_data=True, reload=True, augment=2)
    x_train, y_train, x_test, y_test = loadData(path_data=True, reload=True, augment=3)
    x_train, y_train, x_test, y_test = loadData(path_data=False, reload=True, augment=1)
    x_train, y_train, x_test, y_test = loadData(path_data=False, reload=True, augment=2)
    x_train, y_train, x_test, y_test = loadData(path_data=False, reload=True, augment=3)
    print("Train Data:", x_train.shape)
    print("Test Data:", x_test.shape)