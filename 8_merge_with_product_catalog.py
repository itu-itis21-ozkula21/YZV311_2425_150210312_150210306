import pandas as pd

# Define file paths
combined_customer_product_data_path = '/Users/ahsenbeyzaozkul/Desktop/codesofmine/combined_customer_product_data_with_weeks.csv'
product_catalog_path = '/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/product_catalog.csv'
output_path = 'merged_final_data.csv'

# Load the datasets
customer_product_df = pd.read_csv(combined_customer_product_data_path)
product_catalog_df = pd.read_csv(product_catalog_path)

# Drop the 'categories' column from product_catalog.csv
if 'categories' in product_catalog_df.columns:
    product_catalog_df = product_catalog_df.drop(columns=['categories'])

# Merge the datasets on 'product_id'
merged_df = pd.merge(customer_product_df, product_catalog_df, on='product_id', how='inner')

# Save the merged dataframe to a new CSV
merged_df.to_csv(output_path, index=False)

print(f"Merged data saved to {output_path}")
