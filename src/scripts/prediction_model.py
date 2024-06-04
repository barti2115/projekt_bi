import sqlite3
from sklearn.model_selection import train_test_split
import xgboost as xgb
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

def create_lagged_features(data, lag):
    lagged_data = []
    for i in range(lag, len(data)):
        lagged_data.append(data[i-lag:i])
    return np.array(lagged_data)

def train_and_evaluate_models():
    # Connect to the SQLite database
    conn = sqlite3.connect('data/offers.db')
    cursor = conn.cursor()

    # Fetch USD data from the database
    cursor.execute("SELECT rate FROM USD_pre_processed_data")
    usd_data = cursor.fetchall()

    # Fetch EUR data from the database
    cursor.execute("SELECT rate FROM EUR_pre_processed_data")
    eur_data = cursor.fetchall()

    # Convert the USD and EUR data to numpy arrays
    usd_data = np.squeeze(np.array(usd_data).astype(np.float64), axis=1)
    eur_data = np.squeeze(np.array(eur_data).astype(np.float64), axis=1)
 
    # Create lagged features for USD data
    usd_X = create_lagged_features(usd_data, 7)
    usd_y = usd_data[7:]

    # Create lagged features for EUR data
    eur_X = create_lagged_features(eur_data, 7)
    eur_y = eur_data[7:]

    # Split the data into training and testing sets for USD model
    usd_X_train, usd_X_test, usd_y_train, usd_y_test = train_test_split(usd_X, usd_y, test_size=0.2, shuffle=False)

    # Split the data into training and testing sets for EUR model
    eur_X_train, eur_X_test, eur_y_train, eur_y_test = train_test_split(eur_X, eur_y, test_size=0.2, shuffle=False)

    # Perform hyperparameter tuning for USD model
    usd_params = {
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': [100, 200, 300, 400],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.1, 0.5, 1]
}
    usd_model = xgb.XGBRegressor()
    usd_random = RandomizedSearchCV(estimator=usd_model,n_iter=200, param_distributions=usd_params, scoring='r2', cv=3)
    usd_random.fit(usd_X_train, usd_y_train)
    usd_model = usd_random.best_estimator_

    # Perform hyperparameter tuning for EUR model
    eur_params = {
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': [100, 200, 300, 400],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.1, 0.5, 1]
}
    eur_model = xgb.XGBRegressor()
    eur_random = RandomizedSearchCV(estimator=eur_model, n_iter=200, param_distributions=eur_params, scoring='r2', cv=3)
    eur_random.fit(eur_X_train, eur_y_train)
    eur_model = eur_random.best_estimator_

    # Evaluate the models
    usd_score = usd_model.score(usd_X_test, usd_y_test)
    eur_score = eur_model.score(eur_X_test, eur_y_test)

    # Print the scores
    print("USD Model Score:", usd_score)
    print("EUR Model Score:", eur_score)

    # Close the database connection
    conn.close()
    return usd_model, eur_model

def predict_future_values(usd_model, eur_model):
    # Connect to the SQLite database
    conn = sqlite3.connect('data/offers.db')
    cursor = conn.cursor()

    # Fetch the most recent data from the database
    cursor.execute("SELECT rate FROM USD_pre_processed_data")
    usd_data = cursor.fetchall()

    cursor.execute("SELECT rate FROM EUR_pre_processed_data")
    eur_data = cursor.fetchall()


    # Convert the USD and EUR data to numpy arrays
    usd_data = np.squeeze(np.array(usd_data).astype(np.float64), axis=1)
    eur_data = np.squeeze(np.array(eur_data).astype(np.float64), axis=1)

    # Extract the features (X)
    usd_X = create_lagged_features(usd_data, 7)
    eur_X = create_lagged_features(eur_data, 7)
    # Get the current date
    current_date = datetime.date.today()

    # Predict the future values for the next week
    future_dates = [current_date + timedelta(days=i) for i in range(1, 8)]

    cursor.execute("CREATE TABLE IF NOT EXISTS future_values_usd (date DATE, usd_prediction REAL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS future_values_eur (date DATE, eur_prediction REAL)")
    
    for date in future_dates:
        # Predict the future values using the trained models
        usd_prediction = usd_model.predict([usd_X[-1][:]])[0]
        eur_prediction = eur_model.predict([eur_X[-1][:]])[0]
        # Store the predicted values in separate tables for each currency with future dates
        cursor.execute("INSERT INTO future_values_usd (date, usd_prediction) VALUES (?, ?)", [date, float(usd_prediction)])
        cursor.execute("INSERT INTO future_values_eur (date, eur_prediction) VALUES (?, ?)", [date, float(eur_prediction)])

        usd_data = np.append(usd_data, usd_prediction)
        eur_data = np.append(eur_data, eur_prediction)
        usd_X = create_lagged_features(usd_data, 7)
        eur_X = create_lagged_features(eur_data, 7)
    conn.commit()

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    # Train and evaluate the models
    usd_model, eur_model = train_and_evaluate_models()

    # Predict future values
    predict_future_values(usd_model, eur_model)
