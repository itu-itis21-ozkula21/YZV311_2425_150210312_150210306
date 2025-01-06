import pandas as pd

# Load the updated product catalog
updated_catalog = pd.read_csv('/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/updated_product_catalog.csv')

# Load the customer product data
customer_data = pd.read_csv('/Users/ahsenbeyzaozkul/Desktop/codesofmine/combined_customer_product_data.csv')

# Merge customer product data with updated catalog on product_id
merged_data = customer_data.merge(updated_catalog[['product_id', 'parent_category']], 
                                   on='product_id', 
                                   how='left')

# Update the parent_category_id in customer_data with the new parent_category
customer_data['parent_category_id'] = merged_data['parent_category']

# Save the updated customer data to a new CSV file
customer_data.to_csv('customer_product_data_updated.csv', index=False)

print("The parent_category_id column has been updated and saved to 'customer_product_data_updated.csv'.")