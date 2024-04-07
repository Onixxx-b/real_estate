from keras.models import load_model
import numpy as np
import pickle


def get_prediction(region, room_count, total_sq, floor, subway, max_fl, liv_sq, kitchen_sq):
    num_networks = 5
    networks = []
    for i in range(num_networks):
        model = load_model(f"saved/model_{i}.h5")
        networks.append(model)

    features = np.array([region, room_count, total_sq, floor, subway, max_fl, liv_sq, kitchen_sq])
    X_new = np.column_stack(features)

    y_pred_networks = np.zeros((num_networks, len(X_new)))
    for i, model in enumerate(networks):
        y_pred_networks[i] = model.predict(X_new).flatten()
    with open("saved/xgb_model.pkl", "rb") as f:
        boosting_model = pickle.load(f)
    prediction = boosting_model.predict(y_pred_networks.T)
    return prediction


print(get_prediction(4, 2, 72.5, 10, 1, 16, 20.5, 30)[0])
