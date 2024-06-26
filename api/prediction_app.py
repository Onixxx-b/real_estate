from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from keras.models import load_model
import numpy as np
import pickle


def get_prediction(region, room_count, total_sq, floor, subway, max_fl, liv_sq, kitchen_sq):
    num_networks = 5
    networks = []
    for i in range(num_networks):
        model = load_model(f"../saved_percent/model_{i}.h5")
        networks.append(model)

    regions = np.zeros(10)
    regions[region] = 1
    features = np.array([room_count, total_sq, floor, subway, max_fl, liv_sq, kitchen_sq, regions])
    X_new = np.column_stack(features)

    y_pred_networks = np.zeros((num_networks, len(X_new)))
    for i, model in enumerate(networks):
        y_pred_networks[i] = model.predict(X_new).flatten()
    with open("../saved_percent/xgb_model.pkl", "rb") as f:
        boosting_model = pickle.load(f)
    prediction = boosting_model.predict(y_pred_networks.T)
    return prediction[0]


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/api/regions')
@cross_origin()
def get_regions():
    regions = {
        'regions': [
            {
                'name': 'Голосіївський',
                'index': 0
            },
            {
                'name': 'Солом\'янський',
                'index': 8
            },
            {
                'name': 'Шевченківський',
                'index': 9
            },
            {
                'name': 'Святошинський',
                'index': 7
            },
            {
                'name': 'Подільський',
                'index': 6
            },
            {
                'name': 'Дарницький',
                'index': 1
            },
            {
                'name': 'Печерський',
                'index': 5
            },
            {
                'name': 'Оболонський',
                'index': 4
            },
            {
                'name': 'Деснянський',
                'index': 2
            },
            {
                'name': 'Дніпровський',
                'index': 3
            }
        ]
    }
    return jsonify(regions)


@app.route('/api/price', methods=['POST'])
@cross_origin()
def get_price():
    data = request.json
    prediction = get_prediction(int(data['Region']), int(data['Room count']), float(data['Total Square']),
                                int(data['Floor']), int(data['Subway']), int(data['Max Floor']),
                                float(data['Living Square']), float(data['Kitchen Square']))
    result = {'price': float(prediction)}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
