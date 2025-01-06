import pandas as pd
from datetime import datetime

# Load data (replace file paths with actual file paths)
transactions_df = pd.read_csv("/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/transactions.csv")
product_catalog_df = pd.read_csv("/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/product_catalog.csv")
product_category_map_df = pd.read_csv("/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/product_category_map.csv")

# Ensure proper datetime format
transactions_df['purchase_date'] = pd.to_datetime(transactions_df['purchase_date'])

# Calculate Product-Level Metrics for Each Customer
def calculate_product_metrics(transactions, product_catalog, category_map):
    # Merge transactions with product catalog
    merged_df = transactions.merge(product_catalog, on='product_id', how='left')
    
    # Merge with category map to get parent category
    merged_df = merged_df.merge(category_map, left_on='attribute_5', right_on='category_id', how='left')

    # Calculate Recency, Frequency, and Average Purchase Interval for each customer-product pair
    metrics = []
    for (customer_id, product_id), group in merged_df.groupby(['customer_id', 'product_id']):
        parent_category_id = group['parent_category_id'].iloc[0] if 'parent_category_id' in group else None
        recency_score = (merged_df['purchase_date'].max() - group['purchase_date'].max()).days
        frequency_score = len(group)
        avg_purchase_interval = group['purchase_date'].diff().dt.days.mean() if len(group) > 1 else None
        total_purchases = group['quantity'].sum()

        metrics.append({
            'customer_id': customer_id,
            'product_id': product_id,
            'recency_score': recency_score,
            'frequency_score': frequency_score,
            'avg_purchase_interval': avg_purchase_interval,
            'parent_category_id': parent_category_id,
            'total_purchases': total_purchases
        })

    metrics_df = pd.DataFrame(metrics)
    return metrics_df

product_metrics_df = calculate_product_metrics(transactions_df, product_catalog_df, product_category_map_df)

# Save the product-level metrics to a CSV file
product_metrics_df.to_csv("customer_product_metrics.csv", index=False)

print("Product-level metrics computed and saved to customer_product_metrics.csv.")
