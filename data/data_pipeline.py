import pandas as pd
import glob
import os

class DataPipeline:
    def __init__(self, receipts_dir, output_path):
        self.receipts_dir = receipts_dir
        self.output_path = output_path

    def load_receipts(self):
        # Load all CSV receipt files from the directory
        files = glob.glob(os.path.join(self.receipts_dir, "*.csv"))
        dfs = [pd.read_csv(file) for file in files]
        receipts = pd.concat(dfs, ignore_index=True)
        return receipts

    def clean_receipts(self, receipts):
        receipts.drop_duplicates(inplace=True)
        receipts['date'] = pd.to_datetime(receipts['date'])
        receipts['product_id'] = receipts['product_id'].astype(str)
        return receipts

    def aggregate_receipts(self, receipts):
        # Compute baseline sales (non-promotion) and promo sales (promotion transactions)
        baseline = receipts[receipts['is_promotion'] == False].groupby('product_id')['price'].sum().reset_index().rename(columns={'price': 'baseline_sales'})
        promo = receipts[receipts['is_promotion'] == True].groupby('product_id')['price'].sum().reset_index().rename(columns={'price': 'promo_sales'})
        # Average discount during promotions
        discount = receipts[receipts['is_promotion'] == True].groupby('product_id')['discount'].mean().reset_index().rename(columns={'discount': 'avg_discount'})
        # Promotion frequency (number of promotions per product)
        frequency = receipts[receipts['is_promotion'] == True].groupby('product_id').size().reset_index(name='promo_frequency')

        # Merge aggregated metrics
        agg_df = baseline.merge(promo, on='product_id', how='outer')
        agg_df = agg_df.merge(discount, on='product_id', how='outer')
        agg_df = agg_df.merge(frequency, on='product_id', how='outer')
        agg_df.fillna(0, inplace=True)
        
        # Compute lift: (promo_sales - baseline_sales) / baseline_sales
        agg_df['lift'] = (agg_df['promo_sales'] - agg_df['baseline_sales']) / agg_df['baseline_sales'].replace(0, 1)
        
        # Derive "optimal" targets based on high-performing promotions (for example purposes, use simple rules)
        agg_df['optimal_discount'] = agg_df['lift'].apply(lambda x: 0.25 if x > 0.1 else 0.10)
        agg_df['optimal_frequency'] = agg_df['lift'].apply(lambda x: 4 if x > 0.1 else 2)
        
        return agg_df

    def run(self):
        receipts = self.load_receipts()
        receipts = self.clean_receipts(receipts)
        aggregated = self.aggregate_receipts(receipts)
        aggregated.to_csv(self.output_path, index=False)
        print("ETL pipeline complete, data saved to", self.output_path)
        return aggregated

# Usage example:
# pipeline = DataPipeline(receipts_dir='data/receipts', output_path='data/merged_data.csv')
# aggregated_data = pipeline.run()
