from flask import Flask, render_template
from fbprophet import Prophet
from datetime import datetime, timedelta
from flask_caching import Cache
import pandas as pd

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'redis'})

def fetch_historical_data():
    # This is just an example dataset for demonstration
    data = {
        'ds': [datetime(2024, 3, 29, hour) for hour in range(24)],
        'y': [70000, 71000, 72000, 73000, 74000, 73000, 72000, 71000, 
              70000, 69000, 68000, 67000, 68000, 69000, 70000, 71000, 
              72000, 73000, 74000, 73000, 72000, 71000, 70000, 69000]
    }
    return data

def preprocess_data(data):
    return data

def train_model(data):
    model = Prophet()
    model.fit(data)
    return model

def make_prediction(model):
    future = model.make_future_dataframe(periods=30, freq='min')  # Predict for the next 30 minutes
    forecast = model.predict(future)
    start_time = forecast.tail(1)['ds'].values[0]
    end_time = forecast.tail(1)['ds'].values[-1]
    starting_price = forecast.tail(1)['yhat'].values[0]
    ending_price = forecast.tail(1)['yhat'].values[-1]
    return start_time, end_time, starting_price, ending_price

@app.route('/')
@cache.cached(timeout=300)
def index():
    historical_data = fetch_historical_data()
    preprocessed_data = preprocess_data(historical_data)
    model = train_model(preprocessed_data)
    start_time, end_time, starting_price, ending_price = make_prediction(model)
    return render_template('index.html', start_time=start_time, end_time=end_time,
                           starting_price=starting_price, ending_price=ending_price)

if __name__ == '__main__':
    app.run(debug=True)
