# Install these packages if necessary
# pip install pandas numpy scikit-learn Flask pymongo

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
import pickle
from dotenv import load_dotenv
import os

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# Connect to MongoDB and fetch historical parking data
client = MongoClient(mongo_uri)
db = client['test']
collection = db['occupancy_new']

# Convert MongoDB data to a DataFrame
data = pd.DataFrame(list(collection.find()))

# Drop the MongoDB ID and preprocess if needed
data.drop('_id', axis=1, inplace=True)

# Example preprocessing
# Assume we have columns: day_of_week, hour, weather, events_nearby, and is_available (target)

# Separate features and target variable
X = data[['hour', 'day', 'weather', 'events']]
y = data['occupancy']
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

# Set up the parameter grid for GridSearchCV
#param_grid = {
 #   'n_estimators': [100, 200, 300],  # Number of trees
  #  'max_depth': [None, 10, 20, 30],  # Depth of trees
   # 'min_samples_split': [2, 5, 10],  # Min samples for splitting a node
    #'min_samples_leaf': [1, 2, 4],  # Min samples for leaf nodes
    #'bootstrap': [True, False],  # Whether bootstrap samples are used
#}

#grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
model.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
#print("Best parameters found: ", grid_search.best_params_)

# Get the best model from grid search
#best_model = grid_search.best_estimator_

# Make predictions on the test set
y = model.predict(X_test)

# Calculate evaluation metrics
#mse = mean_squared_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)
#mae = mean_absolute_error(y_test, y_pred)

# Print the evaluation metrics
#print(f"Mean Squared Error: {mse:.2f}")
#print(f"R^2 Score: {r2:.2f}")
#print(f"Mean Absolute Error: {mae:.2f}")

with open('parking_model.pkl', 'wb') as f:
    pickle.dump(model, f)
