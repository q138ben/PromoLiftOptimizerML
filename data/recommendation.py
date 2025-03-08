# src/recommendation_engine.py
import pandas as pd
import joblib
import numpy as np

class RecommendationEngine:
    def __init__(self, data_path, model_path, output_path):
        """
        :param data_path: Path to the aggregated data CSV (should contain current discount_rate and promo_frequency).
        :param model_path: Path to the trained regression model.
        :param output_path: Path to save the product recommendations.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.output_path = output_path

    def load_model(self):
        return joblib.load(self.model_path)
    
    def find_optimal_values(self, model):
        # Define search ranges: these can be refined based on business context.
        discount_range = np.linspace(0.05, 0.5, 10)  # Discount as a decimal (5% to 50%)
        freq_range = np.arange(1, 11)                 # Frequency between 1 and 10
        
        # For each product, we run the grid search.
        def optimize_row(row):
            # Here we assume the model can be applied directly. In practice, you might segment the search
            # based on product features. For now, we search over the fixed ranges.
            best_discount, best_frequency, best_lift = None, None, -np.inf
            for d in discount_range:
                for f in freq_range:
                    features = np.array([[d, f]])
                    predicted_lift = model.predict(features)[0]
                    if predicted_lift > best_lift:
                        best_lift = predicted_lift
                        best_discount = d
                        best_frequency = f
            return pd.Series({'optimal_discount': best_discount, 'optimal_frequency': best_frequency, 'optimal_lift': best_lift})
        
        return optimize_row

    def generate_recommendations(self):
        df = pd.read_csv(self.data_path)
        model = self.load_model()
        
        # For each product, compute optimal discount and frequency using grid search.
        optimizer = self.find_optimal_values(model)
        optimal_df = df.apply(lambda row: optimizer(row), axis=1)
        df = pd.concat([df, optimal_df], axis=1)
        
        # Predict current lift based on current discount_rate and promo_frequency.
        features = ['discount_rate', 'promo_frequency']
        df['predicted_lift'] = model.predict(df[features])
        
        # Generate recommendations:
        def recommendation(row):
            recs = []
            # Challenge 2: Stop promotion if current predicted lift is negative.
            if row['predicted_lift'] < 0:
                recs.append('Stop Promotion')
            # Challenge 1: Adjust discount if current discount differs significantly from optimal.
            if row['discount_rate'] > row['optimal_discount'] and row['predicted_lift'] < row['optimal_lift']:
                recs.append('Reduce Discount')
            elif row['discount_rate'] < row['optimal_discount'] and row['predicted_lift'] < row['optimal_lift']:
                recs.append('Increase Discount')
            # Challenge 3: Adjust frequency.
            if row['promo_frequency'] > row['optimal_frequency']:
                recs.append('Decrease Promotion Frequency')
            elif row['promo_frequency'] < row['optimal_frequency']:
                recs.append('Increase Promotion Frequency')
            if not recs:
                recs.append('Maintain Strategy')
            return ', '.join(recs)
        
        df['recommendation'] = df.apply(recommendation, axis=1)
        df.to_csv(self.output_path, index=False)
        print("Recommendations generated and saved to", self.output_path)
        return df

# Usage:
# engine = RecommendationEngine(data_path='data/merged_data.csv',
#                               model_path='models/lift_model.pkl',
#                               output_path='data/product_recommendations.csv')
# recommendations = engine.generate_recommendations()
