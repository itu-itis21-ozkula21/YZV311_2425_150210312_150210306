import pandas as pd

# Load the transaction data
transactions = pd.read_csv('/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/transactions.csv')

# Display the first few rows to understand the structure of the data (optional)
print(transactions.head())

# Group by customer_id and product_id to calculate aggregated statistics
aggregated_stats = transactions.groupby(['customer_id', 'product_id']).agg(
    purchase_count=('quantity', 'count'),          # Frequency of purchases
    mean_quantity=('quantity', 'mean'),           # Mean of quantity purchased
    std_quantity=('quantity', 'std')              # Standard deviation of quantity purchased
).reset_index()

# Optional: Fill any NaN values for standard deviation (if there's only one transaction)
aggregated_stats['std_quantity'].fillna(0, inplace=True)

# Save the aggregated statistics to a CSV file for further use
aggregated_stats.to_csv('aggregated_customer_product_stats.csv', index=False)

# Display the aggregated statistics (optional)
print(aggregated_stats)
