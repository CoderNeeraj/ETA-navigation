from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('map.pkl', 'rb') as file:
    model = pickle.load(file)

# Haversine formula to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_lat = float(request.form['start_lat'])
        start_lon = float(request.form['start_lon'])
        end_lat = float(request.form['end_lat'])
        end_lon = float(request.form['end_lon'])
        num_turns = int(request.form['num_turns'])
        num_traffic_lights = int(request.form['num_traffic_lights'])
        time_of_day = int(request.form['time_of_day'])
        day_of_week = int(request.form['day_of_week'])
        avg_speed = float(request.form['avg_speed'])
        current_traffic = float(request.form['current_traffic'])

        distance = haversine(start_lat, start_lon, end_lat, end_lon)
        speed_factor = avg_speed / (current_traffic + 1)

        data = pd.DataFrame({
            'distance': [distance],
            'num_turns': [num_turns],
            'num_traffic_lights': [num_traffic_lights],
            'time_of_day': [time_of_day],
            'day_of_week': [day_of_week],
            'speed_factor': [speed_factor]
        })

        eta_pred = model.predict(data)[0]

        return render_template('index.html', prediction=f'Estimated Time of Arrival (ETA): {eta_pred:.2f} hours', distance=f'Distance: {distance:.2f} km')

    return render_template('index.html', prediction='', distance='')

if __name__ == '__main__':
    app.run(debug=True)
