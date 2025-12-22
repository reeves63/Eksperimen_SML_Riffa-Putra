import os
import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')


def load_dataset(input_path):
    print(f"[INFO] Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] Dataset loaded successfully with shape: {df.shape}")
    return df


def remove_duplicates(df):
    initial_shape = df.shape[0]
    duplicate_rows = df[df.duplicated()]
    print(f"[INFO] Number of duplicate rows found: {duplicate_rows.shape[0]}")
    
    df = df.drop_duplicates()
    print(f"[INFO] Duplicates removed. Rows: {initial_shape} -> {df.shape[0]}")
    return df


def check_missing_values(df):
    missing_values = df.isnull().sum()
    print(f"[INFO] Missing values per column:")
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("[INFO] No missing values detected.")


def remove_unnecessary_values(df):
    initial_shape = df.shape[0]
    df = df[df['gender'] != 'Other']
    print(f"[INFO] Removed 'Other' gender. Rows: {initial_shape} -> {df.shape[0]}")
    return df


def remap_smoking_status(status):
    if status in ['never', 'No Info']:
        return 'non-smoker'
    elif status == 'current':
        return 'current'
    elif status in ['ever', 'former', 'not current']:
        return 'past_smoker'
    return status


def process_smoking_history(df):
    print("[INFO] Remapping smoking history categories...")
    df['smoking_history'] = df['smoking_history'].apply(remap_smoking_status)
    print(f"[INFO] Smoking history value counts:")
    print(df['smoking_history'].value_counts())
    return df


def apply_one_hot_encoding(df, column_name):
    print(f"[INFO] Applying one-hot encoding to column: {column_name}")
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)
    return df


def encode_categorical_features(df):
    print("[INFO] Starting categorical feature encoding...")
    df = apply_one_hot_encoding(df, 'gender')
    df = apply_one_hot_encoding(df, 'smoking_history')
    print(f"[INFO] Final dataset shape after encoding: {df.shape}")
    return df


def save_processed_dataset(df, output_path):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")
    
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Processed dataset saved to: {output_path}")
    print(f"[SUCCESS] Final dataset shape: {df.shape}")


def run_preprocessing_pipeline(input_path, output_path):
    # Step 1: Load dataset
    df = load_dataset(input_path)
    
    # Step 2: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 3: Check missing values
    check_missing_values(df)
    
    # Step 4: Remove unnecessary values
    df = remove_unnecessary_values(df)
    
    # Step 5: Process smoking history
    df = process_smoking_history(df)
    
    # Step 6: Encode categorical features
    df = encode_categorical_features(df)
    
    # Step 7: Save processed dataset
    save_processed_dataset(df, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Automated Preprocessing Pipeline for Diabetes Prediction Dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='diabetes_prediction_dataset_raw/diabetes_prediction_dataset.csv',
        help='Path to the raw dataset CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='preprocessing/diabetes_dataset_preprocessing/diabetes_dataset_processed.csv',
        help='Path to save the processed dataset CSV file'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        print("[ERROR] Please ensure the raw dataset is in the correct location.")
        return 1
    
    try:
        run_preprocessing_pipeline(args.input, args.output)
        return 0
    except Exception as e:
        print(f"[ERROR] Preprocessing failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())