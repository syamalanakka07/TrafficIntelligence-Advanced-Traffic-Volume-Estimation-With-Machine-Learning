from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scale.pkl", "rb"))

# Manual mappings used during training
holiday_mapping = {
    'None': 7,
    'Columbus Day': 1,
    'Veterans Day': 10,
    'Thanksgiving Day': 9,
    'Christmas Day': 6,
    'New Years Day': 2,
    'Washingtons Birthday': 11,
    'Memorial Day': 5,
    'Independence Day': 12,
    'State Fair': 3,
    'Labor Day': 9,
    'Martin Luther King Jr Day': 4
}

weather_mapping = {
    'Clear': 0,
    'Clouds': 1,
    'Rain': 3,
    'Drizzle': 4,
    'Mist': 5,
    'Haze': 4,
    'Fog': 7,
    'Thunderstorm': 10,
    'RainSnow': 8,
    'Squall': 9,
    'Smoke': 10
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        holiday = request.form['holiday']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])

        # Map string values to numbers
        holiday_encoded = holiday_mapping.get(holiday, 7)
        weather_encoded = weather_mapping.get(weather, 0)

        # Arrange features in correct order
        input_data = pd.DataFrame([[holiday_encoded, temp, rain, snow, weather_encoded, day, month, year, hours, minutes, seconds]],
                                  columns=['holiday', 'temp', 'rain', 'snow', 'weather', 'day', 'month', 'year', 'hours', 'minutes', 'seconds'])

        # Feature scaling
        scaled_data = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(scaled_data)

        return render_template('output.html', prediction=int(prediction))

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True, use_reloader=False)