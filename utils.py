import os
import random
from datetime import datetime as dt
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def setRandomSeed(seed):
    generator = torch.Generator()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    generator.manual_seed(seed)
    print("Random Seed: ", seed)

    return generator

def distance(lat1, lon1, lat2, lon2):
    """
    Calculates Distance between two coordinations.
    """
    # 지구 반지름 (km)
    R = 6373.0

    # 라디안으로 변환
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # 위도, 경도 차이 계산
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # 구면 삼각법을 이용한 거리 계산
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

def parseJSON(JSON, bef_hour=None):
    x = np.zeros(97)
    grade_map = {"TD":1, "TS":2, "STS":3, "TY":4, "STY":4, "L":1, "I": 0, "WV": 0, "LO": 1, "H1": 2, "H2": 2, "H3": 2, "H4": 2, "H5": 2}
    observation_date = JSON["observation_date"]
    observation_date = dt.strptime(observation_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    x[0] = int(observation_date.year)
    x[1] = int(observation_date.month) + int(observation_date.day)/31
    x[2] = int(observation_date.hour)
    x[3] = grade_map[JSON["grade_type"]]  # grade
    x[4] = float(JSON["central_latitude"])  # lat
    x[5] = float(JSON["central_longitude"])  #lon
    if bef_hour is None:
        x[6] = 0.  # observation time interval
    else:
        interval = x[2] - int(bef_hour)
        if interval < 0: interval += 24
        x[6] = interval
    era5 = JSON["around_weathers"]
    # coord_mapper = {315: 0, 0: 10, 45: 20, 270: 30, 90: 50, 225: 60, 180: 70, 135: 80}
    coord_mapper = {10:0, 0:10, 1:20, 9:30, 3:50, 7:60, 6:70, 4:80}

    for d in era5:
        point = int(d["point"])
        distance = int(d["distance"])
        if distance == 0 and point == 0:
            idx = 40
        elif distance == 750:
            if point == 0:
                idx = 10
            else:
                # idx = coord_mapper[(point-1)*45]
                try:
                    idx = coord_mapper[point]
                except KeyError:
                    continue
        else:
            continue
        idx += 7
        cols = ["temperature_2m", "relativehumidity_2m", "pressure_msl", "cloudcover", "direct_normal_irradiance",
                "windspeed_10m", "windspeed_100m", "winddirection_10m", "winddirection_100m", "windgusts_10m"]
        for i in range(len(cols)):
            x[idx+i] = float(d[cols[i]])

    return np.expand_dims(x, axis=0)
