import pandas as pd

# Load the transactions data
transactions = pd.read_csv('/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/transactions.csv')

# Load the updated customer data
customer_data_updated = pd.read_csv('/Users/ahsenbeyzaozkul/Desktop/codesofmine/customer_product_data_updated.csv')

# Merge the two datasets on customer_id and product_id
merged_data = transactions.merge(customer_data_updated, 
                                  on=['customer_id', 'product_id'], 
                                  how='left')

# Sort the merged data by customer_id and product_id
merged_data = merged_data.sort_values(by=['customer_id', 'product_id'])

# Save the new combined DataFrame to a CSV file
merged_data.to_csv('combined_data.csv', index=False)

print("The data has been combined, sorted by customer_id and product_id, and saved to 'combined_data.csv'.")
