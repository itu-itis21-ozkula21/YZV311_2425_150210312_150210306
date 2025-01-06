import pandas as pd

# Load the two CSV files
df1 = pd.read_csv('/Users/ahsenbeyzaozkul/Desktop/codesofmine/aggregated_customer_product_stats.csv')
df2 = pd.read_csv('/Users/ahsenbeyzaozkul/Desktop/codesofmine/customer_product_metrics.csv')

# Clean the column names by stripping any leading/trailing spaces
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Merge the two dataframes on 'customer_id' and 'product_id', keeping all rows from both dataframes
combined_df = pd.merge(df1, df2, on=['customer_id', 'product_id'], how='outer')

# Save the combined dataframe to a new CSV file
combined_df.to_csv('combined_customer_product_data.csv', index=False)

# Optional: Check the combined dataframe
print(combined_df)
