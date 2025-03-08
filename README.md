# PromoLiftOptimizerML
PromoLiftOptimizerML uses historical promotion data (including sales, discounts, promotion frequency, etc.) to train a regression model that predicts the sales lift of a promotion.

Based on the predicted lift and the current promotion frequency, the system will explicitly addresses all three challenges:

*Unbalanced Discounting*: Retailers may discount products with low lift potential too often and not discount products with high lift potential enough.

*Worst-Performing Promotions*: Identify promotions that yield negative or negligible lift and recommend shutting them down.

*Promotion Frequency Issues*: Ensure products with strong uplift are promoted frequently while those with weak potential are over-promoted.


## Repository Structure

PromoLiftOptimizerML/
├── data/
│   ├── historical_promotions.csv    
│   └── merged_data.csv              
├── notebooks/
│   └── EDA.ipynb                    
├── src/
│   ├── data_pipeline.py            
│   ├── model_training.py            
│   ├── recommendation.py            
│   └── config.py                   
├── dashboard/
│   └── streamlit_app.py           
├── tests/
│   └── test_recommendation.py       
└── README.md
└── app.py