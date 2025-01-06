import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
        'quantity',
        'purchase_count',
        'mean_quantity',
        'std_quantity',
        'recency_score',
        'frequency_score',
        'avg_purchase_interval',
        'parent_category_id',
        'total_purchases',
        'manufacturer_id',
        'attribute_1',
        'attribute_2',
        'attribute_3',
        'attribute_4',
        'attribute_5'
    ]
    
    # Verify all features exist in both datasets
    missing_train = [col for col in feature_cols if col not in train_df.columns]
    missing_test = [col for col in feature_cols if col not in test_df.columns]
    
    if missing_train or missing_test:
        print("\nDetailed column comparison:")
        print("Training columns (exact):", [f"'{col}'" for col in train_df.columns])
        print("Test columns (exact):", [f"'{col}'" for col in test_df.columns])
        print("Missing columns in training:", missing_train)
        print("Missing columns in test:", missing_test)
        raise ValueError("Missing required columns in datasets")
    
    target_col = 'week'
    
    # Categorical features
    categorical_features = [
        'manufacturer_id',
        'parent_category_id',
        'attribute_1',
        'attribute_2',
        'attribute_3',
        'attribute_4',
        'attribute_5'
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
    
    return X, y, X_test, test_df, categorical_features

def train_and_predict(X, y, X_test, test_df, cat_features):
    # ---------------------
    # 1) Split the data
    # ---------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ---------------------
    # 2) Initialize and train the CatBoost model
    # ---------------------
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        cat_features=cat_features,
        early_stopping_rounds=50,
        verbose=100
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    # ---------------------
    # 3) Evaluate on validation set
    # ---------------------
    val_preds = model.predict(X_val)
    print("\nValidation Set Performance:")
    print(classification_report(y_val, val_preds))
    
    # ---------------------
    # 4) Predict on the test set
    # ---------------------
    test_preds = model.predict(X_test).astype(int)
    
    # If CatBoost outputs 2D array, flatten it
    if len(test_preds.shape) > 1:
        test_preds = test_preds.flatten()
    
    # ---------------------
    # 5) Create submission DataFrame
    # ---------------------
    submission = pd.DataFrame({
        'id': test_df['id'],
        'customer_id': test_df['customer_id'],
        'product_id': test_df['product_id'],
        'prediction': test_preds
    })
    
    # ---------------------
    # 6) Check range of predictions (assuming 0-4 range)
    #    Adjust if your data can have other range values
    # ---------------------
    assert submission['prediction'].min() >= 0 and submission['prediction'].max() <= 4, \
        "Predictions outside expected range [0-4]. Adjust assertion if needed."
    
    return submission, model

# Main execution
if __name__ == "__main__":
    # Paths to your data files
    training_data_path = '/Users/ahsenbeyzaozkul/Desktop/codesofmine/sampling/balanced_train.csv'
    test_data_path     = '/Users/ahsenbeyzaozkul/Desktop/codesofmine/sampling/balanced_test.csv'
    
    print("Loading and preprocessing data...")
    X, y, X_test, test_df, cat_features = load_and_preprocess_data(training_data_path, test_data_path)
    
    print("Training model and generating predictions...")
    submission, model = train_and_predict(X, y, X_test, test_df, cat_features)
    
    print("Saving predictions to submission.csv...")
    submission.to_csv('submission.csv', index=False)
    print("Done! Submission file has been created.")
    
    # Print some basic statistics about the predictions
    print("\nPrediction Statistics:")
    print(submission['prediction'].value_counts().sort_index())