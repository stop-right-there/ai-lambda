from datetime import datetime as dt
from datetime import timedelta

import torch
from flask import Flask, request
from flask_restx import Api, Resource
from model import PathPredictor
from utils import *

app = Flask(__name__)
api = Api(app)

model = torch.jit.load("./checkpoints/path_20230425_02164230_4096 0.001.pth", map_location=torch.device('cpu'))
# 자기소개서 요약 API
@api.route("/hello")
class Hello(Resource):
    def get(self):
        return {"content": "안녕하세용"}, 200

@api.route("/api/predict")
class Predictor(Resource):
    def post(self):
        try:
            query_hour = float(request.json["query_hour"])
        except Exception:
            query_hour = 6
        try:
            data = request.json["historical_details"] # Somehow get Data
            observation_date = data[0]["observation_date"]
            observation_date = dt.strptime(observation_date, "%Y-%m-%dT%H:%M:%S.%fZ")
            prediction_date = observation_date + timedelta(hours=query_hour)
            bef_data = parseJSON(data[1])
            cur_data = parseJSON(data[0], bef_data[0][2])
        except Exception as exp:
            return {"error_msg": "데이터를 파싱하는 과정에서 오류가 발생했습니다. \n오류 메시지:"+str(exp)}, 400
        data = np.concatenate([bef_data, cur_data], axis=0)
        data = np.expand_dims(data, axis=0)
        try:
            target_hour = torch.full((data.shape[0],), query_hour)
            predict = model.predict(data, target_hour=target_hour)  # Data should provided in shape of (1, 2, 97)
            print("prediction complete", predict, "query_hour:", query_hour)
        except Exception as exp:
            return {"error_msg": "예측 과정에서 오류가 발생했습니다. \n오류 메시지:"+str(exp)}, 400
        lat = float(predict[0][0])
        lon = float(predict[0][1])

        return {
            "timezone":"GMT",
            "units": {
                "grade": {
                    1: "TD",
                    2: "TS",
                    3: "STS",
                    4: "TY"
                },
                "central_pressure": "hPa",
                "maximum_wind_speed": "knot"
            },
            "prediction_date": prediction_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "central_latitude": lat,
            "central_longitude": lon,
            "grade": None,
            "central_pressure": None,
            "maximum_wind_speed": None
        }, 200

def handler(event, context):
    from serverless_wsgi import handle_request  # serverless-wsgi 모듈을 핸들러 함수 내에서 임포트합니다.
    return handle_request(app, event, context)
