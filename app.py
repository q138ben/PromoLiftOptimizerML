from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained regression model
model = joblib.load('models/lift_model.pkl')

def find_optimal_values(model, discount_range, freq_range):
    best_lift = -np.inf
    best_discount = None
    best_freq = None
    for d in discount_range:
        for f in freq_range:
            features = np.array([[d, f]])
            predicted_lift = model.predict(features)[0]
            if predicted_lift > best_lift:
                best_lift = predicted_lift
                best_discount = d
                best_freq = f
    return best_discount, best_freq, best_lift

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload with:
    {
      "discount_rate": float,         # Current discount rate (decimal)
      "promo_frequency": int          # Current promotion frequency
    }
    """
    data = request.get_json()
    # Convert input data into a DataFrame
    df = pd.DataFrame([data])
    features = ['discount_rate', 'promo_frequency']
    
    # Predict current lift using the model.
    predicted_lift = model.predict(df[features])[0]
    df['predicted_lift'] = predicted_lift
    
    # Define search ranges for optimization.
    discount_range = np.linspace(0.05, 0.5, 10)
    freq_range = np.arange(1, 11)
    
    # Find optimal discount and frequency.
    optimal_discount, optimal_frequency, optimal_lift = find_optimal_values(model, discount_range, freq_range)
    
    recs = []
    # Challenge 2: Stop promotion if performance is negative.
    if predicted_lift < 0:
        recs.append('Stop Promotion')
    # Challenge 1: Unbalanced discounting.
    if predicted_lift < optimal_lift:
        if data['discount_rate'] > optimal_discount:
            recs.append('Reduce Discount')
        elif data['discount_rate'] < optimal_discount:
            recs.append('Increase Discount')
    # Challenge 3: Promotion frequency adjustment.
    if data['promo_frequency'] > optimal_frequency:
        recs.append('Decrease Promotion Frequency')
    elif data['promo_frequency'] < optimal_frequency:
        recs.append('Increase Promotion Frequency')
    if not recs:
        recs.append('Maintain Strategy')
    
    return jsonify({
        'predicted_lift': predicted_lift,
        'optimal_discount': optimal_discount,
        'optimal_frequency': optimal_frequency,
        'optimal_lift': optimal_lift,
        'recommendation': ', '.join(recs)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
