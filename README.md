# PromoLiftOptimizerML
PromoLiftOptimizerML uses historical promotion data (including sales, discounts, promotion frequency, etc.) to train a regression model that predicts the sales lift of a promotion.

Based on the predicted lift and the current promotion frequency, the system will explicitly addresses all three challenges:

*Unbalanced Discounting*: Retailers may discount products with low lift potential too often and not discount products with high lift potential enough.

*Worst-Performing Promotions*: Identify promotions that yield negative or negligible lift and recommend shutting them down.

*Promotion Frequency Issues*: Ensure products with strong uplift are promoted frequently while those with weak potential are over-promoted.


## Repository Structure
```bash

PromoLiftOptimizerML/
├── data/
│   ├── historical_promotions.csv    # Raw historical data
│   └── merged_data.csv              # Processed dataset with computed metrics
├── notebooks/
│   └── EDA.ipynb                    # Exploratory analysis
├── src/
│   ├── data_pipeline.py             # Ingestion and preprocessing
│   ├── model_training.py            # Train model to predict optimal discount and frequency (and lift)
│   ├── recommendation.py            # Generate recommendations based on model predictions vs. current values
│   └── config.py                    # Configurations
├── dashboard/
│   └── streamlit_app.py             # Interactive dashboard for recommendations
├── tests/
│   └── test_recommendation.py       # Unit tests for recommendation logic
└── README.md
└── app.py