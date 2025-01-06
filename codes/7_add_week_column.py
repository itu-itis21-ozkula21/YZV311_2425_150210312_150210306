import pandas as pd
from datetime import datetime, timedelta

# Load the combined customer product data
combined_data = pd.read_csv('/Users/ahsenbeyzaozkul/Desktop/codesofmine/combined_data.csv')

# Define the start date and calculate the week number
start_date = datetime(2020, 6, 1)

# Function to calculate the week number based on the cyclic pattern of 1-4
def assign_week(purchase_date):
    transaction_date = datetime.strptime(purchase_date, '%Y-%m-%d')
    days_diff = (transaction_date - start_date).days
    week_number = (days_diff // 7) % 4 + 1  # Cycles through 1-4
    return week_number

# Apply the function to assign weeks
combined_data['week'] = combined_data['purchase_date'].apply(assign_week)

# Save the updated DataFrame to a new CSV file
combined_data.to_csv('combined_customer_product_data_with_weeks.csv', index=False)

print("The week column has been added and saved to 'combined_customer_product_data_with_weeks.csv'.")
