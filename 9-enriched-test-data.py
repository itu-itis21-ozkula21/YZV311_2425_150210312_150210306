import pandas as pd

# Load the training and test datasets
merged_data = pd.read_csv("/Users/ahsenbeyzaozkul/Desktop/codesofmine/merged_final_data.csv")
test_data = pd.read_csv("/Users/ahsenbeyzaozkul/Desktop/codesofmine/csv_project/test.csv")

# Select columns to assign to the test data
columns_to_assign = [
    "product_id", "quantity", "purchase_count", "mean_quantity", "std_quantity", 
    "recency_score", "frequency_score", "avg_purchase_interval", "parent_category_id", 
    "total_purchases", "manufacturer_id", "attribute_1", "attribute_2", 
    "attribute_3", "attribute_4", "attribute_5"
]

# Deduplicate merged_data based on product_id
deduplicated_data = merged_data.drop_duplicates(subset=["product_id"])

# Merge test data with deduplicated_data to assign the columns
enriched_test_data = test_data.merge(
    deduplicated_data[columns_to_assign],
    on="product_id",
    how="left"
)

# Reorder columns to match the training data structure
enriched_test_data = enriched_test_data[[
    "id", "customer_id", "product_id", "quantity", "purchase_count", "mean_quantity", "std_quantity", 
    "recency_score", "frequency_score", "avg_purchase_interval", "parent_category_id", 
    "total_purchases", "manufacturer_id", "attribute_1", "attribute_2", 
    "attribute_3", "attribute_4", "attribute_5", "prediction"
]]

# Save the enriched test data to a new CSV file
enriched_test_data.to_csv("enriched_test.csv", index=False)

print("Test data has been enriched and saved as 'enriched_test.csv'.")
