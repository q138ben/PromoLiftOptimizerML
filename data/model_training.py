import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

class ModelTrainer:
    def __init__(self, data_path, model_path, random_state=42):
        """
        data_path: Path to the aggregated data CSV file.
        model_path: Where to save the trained model.
        random_state: For reproducible random splitting.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.random_state = random_state

    def load_and_sort_data(self):
        """Load data, convert 'date' to datetime, and sort by date."""
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    def auto_split_data_by_date(self, df):
        """
        Split the data into training, validation, and test sets by day.
        - Training: Earliest 70% of unique days.
        - Last 30% of unique days are randomly split into validation and test,
          such that validation gets 2/3 of the last 30% (20% overall) and test gets 1/3 (10% overall).
        """
        # Extract unique days from the sorted data.
        df['day'] = df['date'].dt.date
        unique_days = np.array(sorted(df['day'].unique()))
        n_days = len(unique_days)
        
        # Determine the number of days for the training set.
        n_train = int(0.7 * n_days)
        train_days = unique_days[:n_train]
        
        # The remaining 30% of days.
        remaining_days = unique_days[n_train:]
        # Randomly shuffle the remaining days.
        np.random.seed(self.random_state)
        shuffled_days = np.random.permutation(remaining_days)
        
        # Calculate counts: out of total days, 70% train, 20% validation, 10% test.
        # For the remaining portion, validation should get 2/3 and test 1/3.
        n_remaining = len(shuffled_days)
        n_val = int(np.round((2/3) * n_remaining))
        val_days = shuffled_days[:n_val]
        test_days = shuffled_days[n_val:]
        
        # Split the dataframe based on day membership.
        train_df = df[df['day'].isin(train_days)].copy()
        val_df = df[df['day'].isin(val_days)].copy()
        test_df = df[df['day'].isin(test_days)].copy()
        
        # Clean up the temporary 'day' column.
        for d in [train_df, val_df, test_df]:
            d.drop(columns=['day'], inplace=True)
        
        print(f"Split data into: {len(train_df)} training records, {len(val_df)} validation records, {len(test_df)} test records.")
        return train_df, val_df, test_df

    def train_model(self):
        df = self.load_and_sort_data()
        train_df, val_df, test_df = self.auto_split_data_by_date(df)
        
        # Define features and target. Example: use 'discount_rate' and 'promo_frequency' to predict 'lift'.
        features = ['discount_rate', 'promo_frequency']
        target = 'lift'
        
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]
        X_test, y_test = test_df[features], test_df[target]
        
        # Train a regression model (Random Forest)
        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_predictions = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_predictions)
        print("Validation Mean Squared Error:", val_mse)
        
        # Evaluate on test set
        test_predictions = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        print("Test Mean Squared Error:", test_mse)
        
        # Save the trained model
        joblib.dump(model, self.model_path)
        print("Model saved at", self.model_path)
        return model

# Usage example:
# trainer = ModelTrainer(data_path='data/merged_data.csv', model_path='models/lift_model.pkl')
# trained_model = trainer.train_model()
