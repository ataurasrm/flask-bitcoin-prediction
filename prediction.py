from flask import Flask, render_template
from fbprophet import Prophet
from datetime import datetime, timedelta
from flask_caching import Cache
import pandas as pd

app = Flask(__name__)

# Configure Flask-Caching to use Redis as the caching backend
cache = Cache(app, config={'CACHE_TYPE': 'redis'})

# Function to fetch historical Bitcoin price data
def fetch_historical_data():
    # This is just an example dataset for demonstration
    data = {
        'ds': [datetime(2024, 3, 29, hour) for hour in range(24)],
        'y': [70000, 71000, 72000, 73000, 74000, 73000, 72000, 71000, 
              70000, 69000, 68000, 67000, 68000, 69000, 70000, 71000, 
              72000, 73000, 74000, 73000, 72000, 71000, 70000, 69000]
    }
    return data

# Function for any preprocessing steps (if needed)
def preprocess_data(data):
    return data

# Function to train the Prophet model
def train_model(data):
    model = Prophet()
    model.fit(data)
    return model

# Function to make predictions
def make_prediction(model):
    # Make predictions for the next hour
    future = model.make_future_dataframe(periods=1, freq='H')
    forecast = model.predict(future)
    # Return the predicted price for the next hour
    return forecast.tail(1)['yhat'].values[0]

# Route for the homepage
@app.route('/')
@cache.cached(timeout=300)  # Cache the result for 5 minutes
def index():
    # Step 1: Fetch historical data
    historical_data = fetch_historical_data()
    
    # Step 2: Preprocess data if needed
    preprocessed_data = preprocess_data(historical_data)
    
    # Step 3: Train the model
    model = train_model(preprocessed_data)
    
    # Step 4: Make predictions
    next_hour_prediction = make_prediction(model)
    
    # Render the template with the prediction
    return render_template('index.html', prediction=next_hour_prediction)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
