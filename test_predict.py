from joblib import load
import numpy as np

# Load model and scaler
model = load('best_model.pkl')
scaler = load('scaler.pkl')

# Sample data
sample = {
    'math_score': 80,
    'reading_score': 75,
    'writing_score': 70,
    'score_percentage': 75
}

features = [
    sample['math_score'],
    sample['reading_score'],
    sample['writing_score'],
    sample['score_percentage']
]

# Scale
features_scaled = scaler.transform([features])

# Predict
prediction = model.predict(features_scaled)[0]
print(f'Predicted level: {prediction}')