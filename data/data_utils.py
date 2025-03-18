"""
Data utilities for the Baccarat Prediction System.
Handles reading, writing, and updating data files.
"""
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from config import REALTIME_FILE, LOG_FILE, BACARAT_DATA_FILE

def update_realtime_data(prev_rounds, actual_result, realtime_file=REALTIME_FILE):
    """
    Update the realtime data file with new results - CLEAN VERSION.
    """
    # Normalize input
    if isinstance(prev_rounds, np.ndarray):
        prev_rounds = prev_rounds.tolist()
    
    if isinstance(actual_result, np.ndarray) or isinstance(actual_result, list):
        actual_result = actual_result[0] if len(actual_result) > 0 else 0
    
    # Prepare data
    new_entry = prev_rounds + [actual_result]
    column_names = [f'Prev_{i+1}' for i in range(5)] + ['Target']
    
    # Try updating the main file
    strategies = ["direct_write", "new_filename", "append_only"]
    
    for strategy in strategies:
        try:
            if strategy == "direct_write":
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(realtime_file), exist_ok=True)
                
                # Read existing data or create new DataFrame
                if os.path.exists(realtime_file):
                    df = pd.read_csv(realtime_file)
                else:
                    df = pd.DataFrame(columns=column_names)
                
                # Add new data
                new_df = pd.DataFrame([new_entry], columns=column_names)
                df = pd.concat([df, new_df], ignore_index=True)
                
                # Try to write with pandas
                df.to_csv(realtime_file, index=False)
                return True
            
            elif strategy == "new_filename":
                # Try writing to a new uniquely named file instead
                timestamp = int(time.time())
                new_file = f"{realtime_file}.{timestamp}.csv"
                
                # Read existing data if possible
                if os.path.exists(realtime_file):
                    try:
                        df = pd.read_csv(realtime_file)
                    except Exception:
                        df = pd.DataFrame(columns=column_names)
                else:
                    df = pd.DataFrame(columns=column_names)
                
                # Add new data
                new_df = pd.DataFrame([new_entry], columns=column_names)
                df = pd.concat([df, new_df], ignore_index=True)
                
                # Write to new file
                df.to_csv(new_file, index=False)
                return True
            
            elif strategy == "append_only":
                # Try append-only mode which might avoid some locking issues
                if not os.path.exists(realtime_file):
                    # Need to create the file with headers first
                    with open(realtime_file, 'w') as f:
                        f.write(','.join(column_names) + '\n')
                    
                # Append the single row directly
                with open(realtime_file, 'a') as f:
                    f.write(','.join(map(str, new_entry)) + '\n')
                
                return True
                
        except Exception as e:
            # If this strategy failed, try the next one
            continue
    
    # If all strategies failed, save to emergency file
    try:
        emergency_file = f"emergency_data_{int(time.time())}.csv"
        with open(emergency_file, 'w') as f:
            f.write(','.join(column_names) + '\n')
            f.write(','.join(map(str, new_entry)) + '\n')
    except Exception:
        pass
    
    return False

def log_prediction(prev_rounds, predicted, actual_result, confidence=None, distribution=None):
    """
    Log prediction results for analysis and tracking with enhanced details.
    
    Args:
        prev_rounds: List of 5 previous outcomes
        predicted: The model's prediction
        actual_result: The actual outcome that occurred
        confidence: Prediction confidence percentage
        distribution: Distribution of prediction probabilities
    """
    log_columns = [f'Prev_{i+1}' for i in range(5)] + [
        'Predicted', 'Actual', 'Correct', 'Confidence', 
        'Banker_Prob', 'Player_Prob', 'Tie_Prob', 'Timestamp'
    ]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if prediction was correct
    correct = 1 if predicted == actual_result else 0
    
    # Extract distribution probabilities with defaults
    banker_prob = distribution.get(0, 33.3) if distribution else 33.3
    player_prob = distribution.get(1, 33.3) if distribution else 33.3
    tie_prob = distribution.get(2, 33.3) if distribution else 33.3
    confidence = confidence or 33.3
    
    # Create log entry
    log_entry = prev_rounds + [
        predicted, actual_result, correct, confidence,
        banker_prob, player_prob, tie_prob, timestamp
    ]
    
    # Initialize or load log file
    if not os.path.exists(LOG_FILE):
        log_df = pd.DataFrame(columns=log_columns)
    else:
        try:
            log_df = pd.read_csv(LOG_FILE)
            
            # Check if columns match, if not adjust them
            if set(log_columns) != set(log_df.columns):
                # Create a new DataFrame with the right columns
                new_log_df = pd.DataFrame(columns=log_columns)
                
                # Copy over existing data where columns match
                for col in set(log_columns).intersection(set(log_df.columns)):
                    new_log_df[col] = log_df[col]
                
                # Initialize new columns with default values
                for col in set(log_columns) - set(log_df.columns):
                    if col == 'Correct':
                        # If we have both Predicted and Actual, we can compute Correct
                        if 'Predicted' in log_df.columns and 'Actual' in log_df.columns:
                            new_log_df['Correct'] = (log_df['Predicted'] == log_df['Actual']).astype(int)
                        else:
                            new_log_df['Correct'] = 0
                    elif col == 'Timestamp':
                        new_log_df['Timestamp'] = timestamp
                    elif col == 'Confidence':
                        new_log_df['Confidence'] = 33.3
                    elif col in ['Banker_Prob', 'Player_Prob', 'Tie_Prob']:
                        new_log_df[col] = 33.3
                    else:
                        new_log_df[col] = None
                
                log_df = new_log_df
        except Exception as e:
            print(f"Error reading log file: {e}")
            log_df = pd.DataFrame(columns=log_columns)
    
    # Add new log entry
    log_df = pd.concat([log_df, pd.DataFrame([log_entry], columns=log_columns)], ignore_index=True)
    
    # Save updated log
    log_df.to_csv(LOG_FILE, index=False)
    
    # Print confirmation
    outcome_names = {0: 'Banker', 1: 'Player', 2: 'Tie'}
    print(f"Log updated: Predicted {outcome_names[predicted]}, Actual {outcome_names[actual_result]}")

def prepare_combined_dataset(bacarat_file=BACARAT_DATA_FILE, realtime_file=REALTIME_FILE, min_records=300):
    """
    Prepare a combined dataset from both bacarat_data.csv and realtime_bacarat.csv
    to improve training data quality and quantity, ensuring minimum dataset size.
    
    Args:
        bacarat_file: Path to main dataset file
        realtime_file: Path to realtime data collected during predictions
        min_records: Minimum number of records required for robust model training
        
    Returns:
        tuple: (X, y) features and target variables
    """
    # Load datasets
    df_bacarat = pd.DataFrame()
    df_realtime = pd.DataFrame()
    
    if os.path.exists(realtime_file):
        try:
            df_realtime = pd.read_csv(realtime_file)
            print(f"Loaded {len(df_realtime)} rows from {realtime_file}")
        except Exception as e:
            print(f"Error loading {realtime_file}: {e}")
    
    if os.path.exists(bacarat_file):
        try:
            df_bacarat = pd.read_csv(bacarat_file)
            print(f"Loaded {len(df_bacarat)} rows from {bacarat_file}")
        except Exception as e:
            print(f"Error loading {bacarat_file}: {e}")
    
    # Check if we have sufficient data
    total_records = len(df_bacarat) + len(df_realtime)
    if total_records < min_records:
        print(f"Warning: Insufficient data ({total_records} < {min_records})")
    
    # Combine the datasets
    if len(df_bacarat) == 0 and len(df_realtime) == 0:
        raise ValueError("No data available in either file. Cannot train model.")
    
    if len(df_bacarat) == 0:
        df_combined = df_realtime.copy()
    elif len(df_realtime) == 0:
        df_combined = df_bacarat.copy()
    else:
        # Ensure same columns exist in both dataframes
        required_columns = [f'Prev_{i+1}' for i in range(5)] + ['Target']
        
        for df in [df_bacarat, df_realtime]:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                print(f"Missing columns in dataset: {missing_cols}")
                for col in missing_cols:
                    df[col] = 0  # Add missing columns with default values
        
        # Concat and remove duplicates
        df_combined = pd.concat([df_bacarat, df_realtime]).drop_duplicates().reset_index(drop=True)
    
    # Shuffle to avoid sequence-based bias, but keep a small portion of recent data in sequence
    if len(df_combined) > 20:
        recent_data = df_combined.iloc[-20:].copy()
        older_data = df_combined.iloc[:-20].sample(frac=1, random_state=42)
        df_combined = pd.concat([older_data, recent_data]).reset_index(drop=True)
    else:
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Combined dataset size: {len(df_combined)} rows")
    
    # Check for data issues
    if 'Target' not in df_combined.columns:
        raise ValueError("Missing 'Target' column in dataset")
    
    # Display data distribution 
    target_counts = df_combined['Target'].value_counts()
    print("Target distribution:")
    for target, count in target_counts.items():
        target_name = {0: 'Banker', 1: 'Player', 2: 'Tie'}.get(target, target)
        print(f"  {target_name}: {count} ({count/len(df_combined)*100:.1f}%)")
    
    # Split into features and target
    feature_cols = [col for col in df_combined.columns if col != 'Target']
    X = df_combined[feature_cols]
    y = df_combined['Target']
    
    return X, y

def check_data_balance():
    """
    Check the balance of outcomes in the training data with enhanced visualization.
    """
    files = [BACARAT_DATA_FILE, REALTIME_FILE]
    for file in files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                if 'Target' in df.columns:
                    # Use value_counts() correctly
                    counts = df['Target'].value_counts().to_dict()
                    total = len(df)
                    print(f"\nData distribution in {file} ({total} rows):")
                    
                    for outcome, count in counts.items():
                        # Ensure outcome is treated as a key, not a callable
                        outcome_key = int(outcome) if isinstance(outcome, (int, float)) else outcome
                        outcome_name = {0: 'Banker', 1: 'Player', 2: 'Tie'}.get(outcome_key, str(outcome))
                        percentage = (count / total) * 100
                        
                        # Create visual representation
                        bar_length = int(percentage / 2)  # Scale to reasonable length
                        bar = '█' * bar_length
                        print(f"  {outcome_name}: {count} ({percentage:.1f}%) {bar}")
                    
                    # Check for data balance issues
                    min_count = min(counts.values())
                    min_pct = (min_count / total) * 100
                    if min_pct < 10:
                        print("  ⚠️ Warning: Data is significantly imbalanced")
            except Exception as e:
                print(f"Error checking {file}: {e}")

def preprocess_data(input_file, output_file):
    """
    Preprocess raw baccarat data into a format suitable for model training.
    
    Args:
        input_file: Path to raw data file with 'Result' column
        output_file: Path to save processed data
    """
    print(f"Processing data from {input_file}...")
    
    # Read data
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check if 'Result' column exists
    if 'Result' not in df.columns:
        print("Error: File must have a 'Result' column")
        return
    
    # Filter out invalid values
    valid_values = ['B', 'P', 'S']
    invalid_count = df[~df['Result'].isin(valid_values)].shape[0]
    if invalid_count > 0:
        print(f"Removing {invalid_count} invalid entries (not B, P, or S)")
        df = df[df['Result'].isin(valid_values)]
    
    # Map values to numbers
    mapping = {'B': 0, 'P': 1, 'S': 2}
    df['Result'] = df['Result'].map(mapping)
    
    # Check for missing values after mapping
    if df['Result'].isna().any():
        print("Warning: Some values could not be mapped. Removing these entries.")
        df = df.dropna(subset=['Result'])
    
    # Create features using sliding window of 5 previous rounds
    sequence_length = 5
    data = []
    
    print(f"Creating sequences with {sequence_length} previous rounds...")
    for i in range(len(df) - sequence_length):
        features = df.iloc[i:i+sequence_length]['Result'].values
        target = df.iloc[i+sequence_length]['Result']
        data.append(list(features) + [target])
    
    # Convert to new DataFrame
    columns = [f'Prev_{i+1}' for i in range(sequence_length)] + ['Target']
    df_transformed = pd.DataFrame(data, columns=columns)
    
    # Save processed data
    df_transformed.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Generated {len(df_transformed)} training samples.")
    print(f"Data saved to {output_file}")

def merge_datasets(files, output_file):
    """
    Merge multiple datasets into one consolidated file.
    
    Args:
        files: List of input CSV files to merge
        output_file: Path to save the merged data
    """
    print(f"Merging {len(files)} datasets...")
    
    dfs = []
    for file in files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"  Read {len(df)} rows from {file}")
                dfs.append(df)
            except Exception as e:
                print(f"  Error reading {file}: {e}")
    
    if not dfs:
        print("No valid datasets found to merge.")
        return
    
    # Check column consistency
    first_columns = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], 1):
        if set(df.columns) != first_columns:
            print(f"Warning: Columns in file {i+1} don't match the first file.")
            print(f"  First file: {first_columns}")
            print(f"  File {i+1}: {set(df.columns)}")
    
    # Merge datasets
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates
    original_len = len(merged_df)
    merged_df.drop_duplicates(inplace=True)
    
    # Save merged dataset
    merged_df.to_csv(output_file, index=False)
    
    print(f"Merged data saved to {output_file}")
    print(f"Total rows: {len(merged_df)} (removed {original_len - len(merged_df)} duplicates)")

def convert_raw_results(results_string, output_file):
    """
    Convert a string of raw results (like "BPBPSBBPPS") to a structured CSV.
    
    Args:
        results_string: String of B, P, S characters representing outcomes
        output_file: Path to save the structured data
    """
    # Clean and validate input
    results_string = results_string.upper().strip()
    valid_chars = set(['B', 'P', 'S'])
    
    if not all(c in valid_chars for c in results_string):
        invalid_chars = [c for c in results_string if c not in valid_chars]
        print(f"Error: Invalid characters found: {', '.join(invalid_chars)}")
        print("Only B (Banker), P (Player), and S (Tie) are allowed.")
        return
    
    # Create DataFrame with single Result column
    df = pd.DataFrame({'Result': list(results_string)})
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Converted {len(results_string)} results to {output_file}")

def ensure_all_directories_exist():
    """
    Ensure all necessary directories exist.
    """
    directories = [
        "data/csv",
        "data/csv/log",
        "data/json",
        "data/images",
        "models/pkl",
        "models/registry"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")