import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample dataset for forest fire prediction
data = {
    'Temperature': [30, 35, 40, 25, 20, 15],
    'Oxygen': [21, 20, 19, 22, 23, 24],
    'Humidity': [50, 40, 30, 60, 70, 80],
    'Fire_Risk': [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

# Features and labels
X = df[['Temperature', 'Oxygen', 'Humidity']]
y = df['Fire_Risk']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test the accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved as model.pkl")
