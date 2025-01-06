import pandas as pd
from sklearn.utils import resample

# Load and preprocess data
def load_and_preprocess_data(training_data_path, test_data_path):
    # Read the data with specific parameters to handle potential issues
    train_df = pd.read_csv(training_data_path, na_values=['', ' ', 'nan', 'NaN'])
    test_df = pd.read_csv(test_data_path, na_values=['', ' ', 'nan', 'NaN'])
    
    # Clean column names by removing extra spaces and standardizing
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    
    print("Training columns:", train_df.columns.tolist())
    print("Test columns:", test_df.columns.tolist())
    
    # Feature columns based on your data structure
    feature_cols = [
        'quantity', 'purchase_count', 'mean_quantity', 'std_quantity', 
        'recency_score', 'frequency_score', 'avg_purchase_interval',
        'parent_category_id', 'total_purchases', 'manufacturer_id', 
        'attribute_1', 'attribute_2', 'attribute_3', 'attribute_4', 'attribute_5'
    ]
    target_col = 'week'

    # Categorical features
    categorical_features = [
        'manufacturer_id', 'parent_category_id', 'attribute_1', 'attribute_2', 
        'attribute_3', 'attribute_4', 'attribute_5'
    ]
    
    # Basic preprocessing on categorical features
    for col in categorical_features:
        train_df[col] = train_df[col].fillna('unknown')
        test_df[col] = test_df[col].fillna('unknown')
        # Convert to string after handling NA values
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
    
    # Handle numeric features
    numeric_features = list(set(feature_cols) - set(categorical_features))
    for col in numeric_features:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        # Fill NA with 0 or median
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(train_df[col].median())
    
    X = train_df[feature_cols].copy()
    y = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    
    return X, y, X_test, test_df, train_df, feature_cols, categorical_features, target_col

# Function to balance classes and downsample data
def balance_and_sample_data(X, y, X_test, test_df, train_df, feature_cols, target_col, output_train_path, output_test_path, sample_size=1000):
    # Check the class distribution before balancing
    class_counts = y.value_counts()
    print(f"Class distribution before balancing:\n{class_counts}")
    
    # Downsample the training data
    print("\nDownsampling the training data to a smaller size...")
    downsampled_data = pd.concat([X, y], axis=1)
    downsampled_data = downsampled_data.sample(n=sample_size, random_state=42)

    X_downsampled = downsampled_data[feature_cols]
    y_downsampled = downsampled_data[target_col]
    
    # If needed, balance the classes in the downsampled training data
    class_counts_downsampled = y_downsampled.value_counts()
    print(f"Class distribution in the downsampled data:\n{class_counts_downsampled}")
    
    if len(class_counts_downsampled) > 1 and class_counts_downsampled.max() > class_counts_downsampled.min() * 2:
        print("\nClasses are imbalanced. Balancing the classes...")
        
        # Balance the classes using resampling
        min_class_size = class_counts_downsampled.min()
        
        balanced_X = pd.DataFrame()
        balanced_y = pd.Series(dtype=int)
        
        for class_label in class_counts_downsampled.index:
            class_data_X = X_downsampled[y_downsampled == class_label]
            class_data_y = y_downsampled[y_downsampled == class_label]
            
            class_data_X_resampled, class_data_y_resampled = resample(
                class_data_X, class_data_y, replace=True,  # Sample with replacement
                n_samples=min_class_size,  # Match the smallest class size
                random_state=42
            )
            
            balanced_X = pd.concat([balanced_X, class_data_X_resampled])
            balanced_y = pd.concat([balanced_y, class_data_y_resampled])
        
        print(f"Class distribution after balancing:\n{balanced_y.value_counts()}")
        X_downsampled = balanced_X
        y_downsampled = balanced_y
    
    # Combine downsampled and balanced data into a single DataFrame for training
    balanced_train_df = X_downsampled.copy()
    balanced_train_df[target_col] = y_downsampled
    
    # Merge additional columns (customer_id, product_id) into the balanced training data
    extra_columns = ['customer_id', 'product_id', 'purchase_date']  # Include all necessary columns from train data
    balanced_train_df = pd.merge(balanced_train_df, train_df[extra_columns], 
                                 how='left', left_index=True, right_index=True)
    
    # For the test set, add 'prediction' and 'id' columns (important for test set)
    balanced_test_df = X_test.copy()
    balanced_test_df = pd.merge(balanced_test_df, test_df[['id', 'customer_id', 'product_id', 'prediction']], 
                                 how='left', left_index=True, right_index=True)
    
    # Save balanced and sampled data to CSV files with the correct column order
    # Ensure the columns are in the exact order required for both datasets
    
    # Correct column order for balanced training data
    train_column_order = ['customer_id', 'product_id', 'purchase_date', 'quantity', 'purchase_count', 'mean_quantity', 'std_quantity', 'recency_score', 'frequency_score', 'avg_purchase_interval', 'parent_category_id', 'total_purchases', 'week', 'manufacturer_id', 'attribute_1', 'attribute_2', 'attribute_3', 'attribute_4', 'attribute_5']
    balanced_train_df = balanced_train_df[train_column_order]
    
    # Correct column order for balanced test data
    test_column_order = ['id', 'customer_id', 'product_id', 'quantity', 'purchase_count', 'mean_quantity', 'std_quantity', 'recency_score', 'frequency_score', 'avg_purchase_interval', 'parent_category_id', 'total_purchases', 'manufacturer_id', 'attribute_1', 'attribute_2', 'attribute_3', 'attribute_4', 'attribute_5', 'prediction']
    balanced_test_df = balanced_test_df[test_column_order]
    
    # Save to CSV files
    balanced_train_df.to_csv(output_train_path, index=False)
    balanced_test_df.to_csv(output_test_path, index=False)
    
    print(f"Balanced and downsampled training data saved to {output_train_path}")
    print(f"Sampled test data saved to {output_test_path}")

# Main execution
if __name__ == "__main__":
    # Paths to your data files
    training_data_path = '/Users/ahsenbeyzaozkul/Desktop/sampling/merged_final_data.csv'
    test_data_path = '/Users/ahsenbeyzaozkul/Desktop/sampling/enriched_test.csv'
    
    # Paths to save the balanced data
    output_train_path = '/Users/ahsenbeyzaozkul/Desktop/sampling/balanced_train.csv'  # Change to your desired location
    output_test_path = '/Users/ahsenbeyzaozkul/Desktop/sampling/balanced_test.csv'    # Change to your desired location
    
    print("Loading and preprocessing data...")
    X, y, X_test, test_df, train_df, feature_cols, categorical_features, target_col = load_and_preprocess_data(training_data_path, test_data_path)
    
    # Balance and sample the data
    balance_and_sample_data(X, y, X_test, test_df, train_df, feature_cols, target_col, output_train_path, output_test_path)
    print("Done! Balanced and sampled data has been saved.")
