import pandas as pd
import numpy as np

# Load data
product_catalog = pd.read_csv("/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/product_catalog.csv")
transactions = pd.read_csv("/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/transactions.csv")
product_category_map = pd.read_csv("/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/product_category_map.csv")

# Get unique product IDs from transactions
transaction_product_ids = transactions['product_id'].unique()

# Strip and clean 'categories' column in product_catalog
product_catalog['categories'] = product_catalog['categories'].apply(lambda x: str(x).strip())

# Filter product_catalog based on transaction product IDs or non-empty 'categories'
filtered_product_catalog = product_catalog[
    (product_catalog['product_id'].isin(transaction_product_ids)) | (product_catalog['categories'] != "")
]

# Drop unnecessary columns
columns_to_drop = ["manufacturer_id", "attribute_1", "attribute_2", "attribute_3", "attribute_4", "attribute_5"]
filtered_product_catalog = filtered_product_catalog.drop(columns=columns_to_drop, errors='ignore')

# Handle 'categories' column: convert NaN to strings and split into lists
filtered_product_catalog['categories'] = filtered_product_catalog['categories'].astype(str)
filtered_product_catalog['categories'] = filtered_product_catalog['categories'].str.strip('[]').str.split(',')

# Expand 'categories' into separate columns
categories_df = filtered_product_catalog['categories'].apply(pd.Series)
df = pd.concat([filtered_product_catalog.drop(columns=['categories']), categories_df], axis=1)

# Clean cells by removing quotes and converting to integers where applicable
def clean_cell(cell):
    if isinstance(cell, str):
        cell = cell.replace("'", "").strip()
        return int(cell) if cell.isdigit() else np.nan
    return cell

df = df.applymap(clean_cell)

# Create a mapping from category_id to parent_category_id
mapping = product_category_map.set_index('category_id')['parent_category_id'].to_dict()

# Replace category IDs with their parent category IDs
def replace_categories(x):
    if pd.notna(x) and int(x) in mapping:
        return mapping[int(x)]
    return x

df = df.applymap(replace_categories)

# Find the most frequent category in each row
def most_frequent(row):
    counts = pd.Series(row).value_counts()
    counts = counts[counts.index != 0]  # Exclude zeros
    return counts.idxmax() if not counts.empty else 0

# Add 'target_category' column
df['target_category'] = df.iloc[:, 1:].apply(most_frequent, axis=1)

# Drop numerical columns and 'predicted_category' if they exist
columns_to_drop = list(range(35)) + ['predicted_category']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Merge with filtered product catalog
merged_df = pd.merge(df, filtered_product_catalog, on='product_id', how='inner')

# Drop 'categories' column and rename 'target_category' to 'parent_category'
merged_df = merged_df.drop("categories", axis=1, errors='ignore')
merged_df.rename(columns={'target_category': 'parent_category'}, inplace=True)

# Save to CSV
file_path = '/Users/ahsenbeyzaozkul/Desktop/codesofmine/updated_product_catalog.csv'
merged_df.to_csv(file_path, index=False)

# Display the result
print(merged_df)
