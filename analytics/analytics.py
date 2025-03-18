"""
Enhanced Analytics and Visualization Tool for Baccarat Prediction System with Stacking Ensemble.

This module provides comprehensive analysis of prediction performance, pattern effectiveness,
model contributions, and stacking behavior for the Baccarat prediction system.

Key features:
- Enhanced visualization of prediction accuracy and trends
- Model contribution analysis for stacking ensemble
- Pattern-specific performance metrics
- Meta-feature importance analysis
- Advanced visualization tools with model breakdown
- Comprehensive reporting capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from datetime import datetime, timedelta
import seaborn as sns
from collections import defaultdict, Counter
import matplotlib.dates as mdates
from colorama import Fore, Style  # Add this import for colored console output
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

from config import LOG_FILE, REALTIME_FILE, MODEL_REGISTRY_PATH

# Set better visualization defaults
plt.style.use('ggplot')
sns.set_style("whitegrid")
sns.set_context("talk")

# Define standard model names for consistent reference
BASE_MODELS = ['baccarat_rf', 'markov_1', 'markov_2', 'xgboost_base']
ALL_MODELS = BASE_MODELS + ['stacking_ensemble']

# Color mapping for consistent model colors
MODEL_COLORS = {
    'baccarat_rf': '#4285F4',       # Google Blue
    'markov_1': '#EA4335',          # Google Red
    'markov_2': '#FBBC05',          # Google Yellow
    'xgboost_base': '#34A853',      # Google Green
    'stacking_ensemble': '#9C27B0', # Purple
}

# Pattern type mappings
PATTERN_TYPES = ['no_pattern', 'streak', 'alternating', 'tie']
PATTERN_COLORS = {
    'no_pattern': '#7F7F7F',    # Gray
    'streak': '#E31A1C',        # Red
    'alternating': '#1F78B4',   # Blue
    'tie': '#33A02C',           # Green
}

def analyze_prediction_history(days=None, pattern_type=None):
    """
    Analyze prediction accuracy over time with enhanced filtering and model breakdown.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        pattern_type: Filter by specific pattern type
        
    Returns:
        dict: Analysis results with accuracy metrics
    """
    if not os.path.exists(LOG_FILE):
        print("No prediction log found. Make some predictions first.")
        return None
    
    # Load log data
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Make sure required columns exist
        if 'Predicted' not in log_df.columns or 'Actual' not in log_df.columns:
            print("Log file doesn't have complete prediction data")
            if 'Predicted' not in log_df.columns:
                print("Missing 'Predicted' column")
            if 'Actual' not in log_df.columns:
                print("Missing 'Actual' column")
            return None
        
        # Add Correct column if missing
        if 'Correct' not in log_df.columns:
            log_df['Correct'] = (log_df['Predicted'] == log_df['Actual']).astype(int)
        
        # Filter by date if specified
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {days} days ({len(log_df)} predictions)")
            
        # Filter by pattern type if specified
        if pattern_type and 'pattern_type' in log_df.columns:
            log_df = log_df[log_df['pattern_type'] == pattern_type]
            print(f"Filtered to pattern type: {pattern_type} ({len(log_df)} predictions)")
        
        # Calculate rolling accuracy with multiple windows if enough data
        if len(log_df) >= 20:
            log_df['RollingAccuracy5'] = log_df['Correct'].rolling(window=5).mean() * 100
            log_df['RollingAccuracy10'] = log_df['Correct'].rolling(window=10).mean() * 100
            log_df['RollingAccuracy20'] = log_df['Correct'].rolling(window=20).mean() * 100
        elif len(log_df) >= 10:
            log_df['RollingAccuracy5'] = log_df['Correct'].rolling(window=5).mean() * 100
            log_df['RollingAccuracy10'] = log_df['Correct'].rolling(window=10).mean() * 100
        elif len(log_df) >= 5:
            log_df['RollingAccuracy5'] = log_df['Correct'].rolling(window=5).mean() * 100
        
        # Define outcome names
        outcomes = {0: 'Banker', 1: 'Player', 2: 'Tie'}
        results = {}
        
        # Calculate overall accuracy
        total = len(log_df)
        correct = log_df['Correct'].sum()
        accuracy = (correct / total) * 100 if total > 0 else 0
        results['overall'] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }
        
        # Calculate accuracy by actual outcome
        for outcome, name in outcomes.items():
            outcome_df = log_df[log_df['Actual'] == outcome]
            outcome_total = len(outcome_df)
            outcome_correct = outcome_df['Correct'].sum()
            outcome_accuracy = (outcome_correct / outcome_total) * 100 if outcome_total > 0 else 0
            
            results[name] = {
                'total': outcome_total,
                'correct': outcome_correct,
                'accuracy': outcome_accuracy
            }
        
        # Calculate accuracy by predicted outcome
        pred_results = {}
        for outcome, name in outcomes.items():
            pred_df = log_df[log_df['Predicted'] == outcome]
            pred_total = len(pred_df)
            pred_correct = pred_df['Correct'].sum()
            pred_accuracy = (pred_correct / pred_total) * 100 if pred_total > 0 else 0
            
            pred_results[name] = {
                'total': pred_total,
                'correct': pred_correct,
                'accuracy': pred_accuracy
            }
        
        # Calculate accuracy by confidence level if available
        conf_results = {}
        if 'Confidence' in log_df.columns:
            # Create confidence bins
            log_df['ConfidenceBin'] = pd.cut(log_df['Confidence'], 
                                            bins=[0, 40, 50, 60, 70, 80, 100],
                                            labels=['0-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-100%'])
            
            for conf_bin in log_df['ConfidenceBin'].unique():
                if pd.isna(conf_bin):
                    continue
                    
                bin_df = log_df[log_df['ConfidenceBin'] == conf_bin]
                bin_total = len(bin_df)
                bin_correct = bin_df['Correct'].sum()
                bin_accuracy = (bin_correct / bin_total) * 100 if bin_total > 0 else 0
                
                conf_results[str(conf_bin)] = {
                    'total': bin_total,
                    'correct': bin_correct,
                    'accuracy': bin_accuracy
                }
        
        # Analyze model contributions if available
        model_contributions = {}
        if 'model_contributions' in log_df.columns:
            # Extract model contributions from JSON strings
            contributions_list = []
            for _, row in log_df.iterrows():
                if isinstance(row['model_contributions'], str):
                    try:
                        contrib = json.loads(row['model_contributions'])
                        contributions_list.append((row['Correct'], contrib))
                    except:
                        pass
                elif isinstance(row['model_contributions'], dict):
                    contributions_list.append((row['Correct'], row['model_contributions']))
            
            # Calculate aggregated model contributions
            if contributions_list:
                # Overall contribution (all predictions)
                all_contribs = defaultdict(float)
                for _, contrib in contributions_list:
                    for model, weight in contrib.items():
                        all_contribs[model] += weight
                
                # Normalize
                total_weight = sum(all_contribs.values())
                if total_weight > 0:
                    all_contribs = {k: v/total_weight for k, v in all_contribs.items()}
                
                # Contribution to correct predictions
                correct_contribs = defaultdict(float)
                correct_total = 0
                for correct, contrib in contributions_list:
                    if correct:
                        correct_total += 1
                        for model, weight in contrib.items():
                            correct_contribs[model] += weight
                
                # Normalize
                if correct_total > 0:
                    correct_contribs = {k: v/correct_total for k, v in correct_contribs.items()}
                
                model_contributions = {
                    'all_predictions': dict(all_contribs),
                    'correct_predictions': dict(correct_contribs)
                }
        
        # Calculate accuracy by pattern type if available
        pattern_results = {}
        if 'pattern_type' in log_df.columns:
            for pattern in log_df['pattern_type'].unique():
                if pd.isna(pattern):
                    continue
                    
                pattern_df = log_df[log_df['pattern_type'] == pattern]
                pattern_total = len(pattern_df)
                pattern_correct = pattern_df['Correct'].sum()
                pattern_accuracy = (pattern_correct / pattern_total) * 100 if pattern_total > 0 else 0
                
                pattern_results[pattern] = {
                    'total': pattern_total,
                    'correct': pattern_correct,
                    'accuracy': pattern_accuracy
                }
        
        # Store results and data
        results['by_prediction'] = pred_results
        results['by_confidence'] = conf_results
        results['by_pattern'] = pattern_results
        results['model_contributions'] = model_contributions
        results['data'] = log_df
        
        return results
    
    except Exception as e:
        print(f"Error analyzing prediction history: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_accuracy_over_time(breakdown_by_model=False):
    """
    Plot accuracy trend over time with enhanced visualization and optional model breakdown.
    
    Args:
        breakdown_by_model: If True, show contributions from each model
    """
    ensure_image_directory()
    results = analyze_prediction_history()
    if not results:
        return
    
    log_df = results['data']
    if len(log_df) < 5:
        print("Not enough data for meaningful visualization (minimum 5 predictions needed)")
        return
    
    fig = plt.figure(figsize=(14, 8))
    
    # If we're showing model breakdown and have model contribution data
    if breakdown_by_model and 'model_contributions' in log_df.columns and 'Timestamp' in log_df.columns:
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(log_df['Timestamp']):
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
        
        # Calculate rolling model-specific accuracy
        models_accuracy = calculate_model_specific_accuracy(log_df)
        
        # Plot model-specific accuracy
        for model, values in models_accuracy.items():
            # Skip models with insufficient data
            if len(values) < 5:
                continue
                
            timestamps = values['timestamps']
            accuracy = values['accuracy']
            
            # Get consistent color for this model
            color = MODEL_COLORS.get(model, None)
            
            plt.plot(timestamps, accuracy, label=f"{model}", 
                   alpha=0.7, linewidth=2, color=color)
    
    # Plot rolling accuracy with multiple windows if available
    if 'RollingAccuracy20' in log_df.columns:
        plt.plot(log_df.index, log_df['RollingAccuracy20'], 'b-', linewidth=3, 
                label='20-Game Rolling Accuracy', alpha=0.8)
    if 'RollingAccuracy10' in log_df.columns:
        plt.plot(log_df.index, log_df['RollingAccuracy10'], 'g-', linewidth=2,
                label='10-Game Rolling Accuracy', alpha=0.8)
    if 'RollingAccuracy5' in log_df.columns:
        plt.plot(log_df.index, log_df['RollingAccuracy5'], 'r-', linewidth=1.5,
                label='5-Game Rolling Accuracy', alpha=0.7)
    
    # Plot overall accuracy line
    overall_acc = results['overall']['accuracy']
    plt.axhline(y=overall_acc, color='purple', linestyle='-', linewidth=2, 
              label=f'Overall: {overall_acc:.1f}%')
    
    # Plot outcome-specific accuracy
    for outcome, color in [('Banker', 'green'), ('Player', 'blue'), ('Tie', 'red')]:
        if outcome in results:
            acc = results[outcome]['accuracy']
            plt.axhline(y=acc, color=color, linestyle='--', alpha=0.6, 
                       label=f'{outcome}: {acc:.1f}%')
    
    # Add expected random accuracy (33.33%)
    plt.axhline(y=33.33, color='gray', linestyle=':', alpha=0.8, 
               label='Random: 33.3%')
    
    # Add trend line if enough data
    if len(log_df) >= 10:
        z = np.polyfit(log_df.index, log_df['Correct'] * 100, 1)
        p = np.poly1d(z)
        plt.plot(log_df.index, p(log_df.index), "k--", alpha=0.5, label='Trend')
    
    # Enhance visuals
    plt.title('Prediction Accuracy Over Time', fontsize=16)
    plt.xlabel('Prediction Number', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    
    # Add key stats as text
    stats_text = (
        f"Overall: {overall_acc:.1f}%\n"
        f"Total predictions: {len(log_df)}\n"
        f"Correct predictions: {log_df['Correct'].sum()}"
    )
    plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Save and show
    os.makedirs("data/images", exist_ok=True)
    plt.tight_layout()
    plt.savefig('data/images/accuracy_trend.png', dpi=300)
    plt.show()

def calculate_model_specific_accuracy(log_df, window=10):
    """
    Calculate rolling accuracy for each model based on their contribution to predictions.
    
    Args:
        log_df: DataFrame with prediction logs
        window: Window size for rolling accuracy
        
    Returns:
        dict: Model-specific accuracy data
    """
    # Initialize model accuracy tracking
    models_data = {model: {'correct': [], 'total': [], 'timestamps': [], 'accuracy': []} for model in ALL_MODELS}
    
    # Process each prediction
    for i in range(len(log_df)):
        row = log_df.iloc[i]
        
        # Skip if no model contributions
        if 'model_contributions' not in row or pd.isna(row['model_contributions']):
            continue
            
        # Parse model contributions
        if isinstance(row['model_contributions'], str):
            try:
                model_contribs = json.loads(row['model_contributions'])
            except:
                continue
        elif isinstance(row['model_contributions'], dict):
            model_contribs = row['model_contributions']
        else:
            continue
            
        # Update each model's stats
        timestamp = row['Timestamp']
        correct = row['Correct']
        
        for model, contrib in model_contribs.items():
            if model in models_data:
                models_data[model]['timestamps'].append(timestamp)
                models_data[model]['correct'].append(correct)
                models_data[model]['total'].append(1)
                
                # Calculate rolling accuracy
                if len(models_data[model]['correct']) >= window:
                    recent_correct = sum(models_data[model]['correct'][-window:])
                    recent_total = sum(models_data[model]['total'][-window:])
                    rolling_acc = (recent_correct / recent_total) * 100 if recent_total > 0 else 0
                else:
                    # If not enough data for window, use all available
                    total_correct = sum(models_data[model]['correct'])
                    total = sum(models_data[model]['total'])
                    rolling_acc = (total_correct / total) * 100 if total > 0 else 0
                    
                models_data[model]['accuracy'].append(rolling_acc)
    
    return models_data

def plot_outcome_distribution(compare_with_prediction=True):
    """
    Plot enhanced distribution of baccarat outcomes with option to compare with predictions.
    
    Args:
        compare_with_prediction: If True, compare actual outcomes with predictions
    """
    ensure_image_directory()
    
    # Check actual outcome distribution
    if not os.path.exists(REALTIME_FILE):
        print("No real-time data found.")
        return
    
    try:
        df = pd.read_csv(REALTIME_FILE)
        if 'Target' not in df.columns:
            print("Data doesn't have Target column")
            return
        
        target_counts = df['Target'].value_counts().sort_index()
        labels = ['Banker', 'Player', 'Tie']
        colors = ['green', 'blue', 'red']
        
        # Figure setup depends on whether we're comparing with predictions
        if compare_with_prediction and os.path.exists(LOG_FILE):
            # Load prediction data
            pred_df = pd.read_csv(LOG_FILE)
            if 'Predicted' not in pred_df.columns or 'Actual' not in pred_df.columns:
                compare_with_prediction = False
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            else:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
                pred_counts = pred_df['Predicted'].value_counts().sort_index()
                actual_counts = pred_df['Actual'].value_counts().sort_index()
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar chart of overall data distribution
        bars = ax1.bar([labels[i] if i in target_counts.index else labels[i] for i in range(3)], 
                     [target_counts.get(i, 0) for i in range(3)], 
                     color=[colors[i] for i in range(3)])
        
        ax1.set_title('Distribution of Baccarat Outcomes (All Data)', fontsize=16)
        ax1.set_xlabel('Outcome', fontsize=14)
        ax1.set_ylabel('Frequency', fontsize=14)
        
        # Add percentage labels
        total = target_counts.sum()
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = height / total * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontsize=12)
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(
            [target_counts.get(i, 0) for i in range(3)],
            labels=[labels[i] for i in range(3)],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=(0.05, 0.05, 0.1)  # Explode all slices slightly, Tie more
        )
        
        # Customize pie chart text
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        ax2.set_title('Percentage Distribution', fontsize=16)
        
        # If comparing with predictions, add comparison chart
        if compare_with_prediction and 'ax3' in locals():
            # Ensure all outcomes are represented
            for i in range(3):
                if i not in pred_counts.index:
                    pred_counts[i] = 0
                if i not in actual_counts.index:
                    actual_counts[i] = 0
            
            pred_counts = pred_counts.sort_index()
            actual_counts = actual_counts.sort_index()
            
            # Create grouped bar chart
            width = 0.35
            x = np.arange(len(labels))
            
            # Create grouped bars
            ax3.bar(x - width/2, [pred_counts.get(i, 0) for i in range(3)], width, 
                   label='Predicted', color=[colors[i] for i in range(3)], alpha=0.7)
            ax3.bar(x + width/2, [actual_counts.get(i, 0) for i in range(3)], width,
                   label='Actual', color=[colors[i] for i in range(3)], alpha=0.3, hatch='//')
            
            ax3.set_title('Predicted vs Actual Outcomes', fontsize=16)
            ax3.set_xlabel('Outcome', fontsize=14)
            ax3.set_ylabel('Frequency', fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels)
            ax3.legend()
            
            # Calculate prediction bias
            pred_total = pred_counts.sum()
            actual_total = actual_counts.sum()
            
            bias_text = "Prediction Bias:\n"
            for i, label in enumerate(labels):
                pred_pct = pred_counts.get(i, 0) / pred_total * 100 if pred_total > 0 else 0
                actual_pct = actual_counts.get(i, 0) / actual_total * 100 if actual_total > 0 else 0
                bias = pred_pct - actual_pct
                bias_text += f"{label}: {'+'if bias>0 else ''}{bias:.1f}%\n"
                
            ax3.text(0.95, 0.05, bias_text, transform=ax3.transAxes, 
                    fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add expected distribution reference (assuming standard baccarat odds)
        expected_text = (
            "Expected distribution:\n"
            "Banker: 45.9%\n"
            "Player: 44.6%\n"
            "Tie: 9.5%"
        )
        fig.text(0.98, 0.15, expected_text, verticalalignment='bottom', 
                horizontalalignment='right', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Add dataset stats
        stats_text = f"Dataset: {len(df)} hands"
        fig.text(0.02, 0.02, stats_text, verticalalignment='bottom', 
                horizontalalignment='left', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('data/images/outcome_distribution.png', dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()

def analyze_confidence_vs_accuracy(by_model=False):
    """
    Analyze relationship between prediction confidence and actual accuracy,
    with optional breakdown by model.
    
    Args:
        by_model: If True, break down by model (requires model_contributions in log)
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        if 'Confidence' not in log_df.columns or 'Correct' not in log_df.columns:
            print("Log file missing confidence or correctness data")
            return
        
        # Create confidence bins
        log_df['ConfidenceBin'] = pd.cut(log_df['Confidence'], 
                                      bins=[0, 40, 50, 60, 70, 80, 100],
                                      labels=['0-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-100%'])
        
        # Calculate accuracy by confidence bin
        results = log_df.groupby('ConfidenceBin')['Correct'].agg(['mean', 'count']).reset_index()
        results['accuracy'] = results['mean'] * 100
        
        # Plot the results
        plt.figure(figsize=(14, 8))
        
        # If by_model is True and we have model_contributions, break down by model
        if by_model and 'model_contributions' in log_df.columns:
            # Get model-specific confidence vs accuracy
            model_calibration = calculate_model_specific_calibration(log_df)
            
            # Plot each model's calibration curve
            for model, data in model_calibration.items():
                if len(data['bins']) >= 3:  # Need minimum data points
                    plt.plot(data['mean_confidence'], data['accuracy'], 
                           'o-', label=f"{model}", alpha=0.7, 
                           color=MODEL_COLORS.get(model))
            
            # Plot overall calibration for comparison
            plt.plot(results['ConfidenceBin'].astype(str), results['accuracy'], 
                   'k--', linewidth=2, label='Overall', alpha=0.9)
                
            # Perfect calibration reference line (diagonal)
            conf_values = [0, 40, 50, 60, 70, 80, 100]
            plt.plot(conf_values, conf_values, 'r-', alpha=0.3, label='Perfect Calibration')
            
            plt.xlabel('Confidence', fontsize=14)
            plt.ylabel('Actual Accuracy (%)', fontsize=14)
            plt.title('Model-Specific Confidence Calibration', fontsize=16)
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
        else:
            # Bar plot for overall calibration
            bar_positions = range(len(results))
            bars = plt.bar(bar_positions, results['accuracy'], 
                         color=plt.cm.viridis(results['accuracy']/100))
            
            # Add count labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + 2,
                       f"{int(results.iloc[i]['count'])} pred.", 
                       ha='center', fontsize=10)
            
            # Add perfect calibration reference line
            x_positions = range(len(results))
            expected_values = [30, 45, 55, 65, 75, 90]  # Middle of each bin
            plt.plot(x_positions[:len(expected_values)], expected_values[:len(results)], 
                   'r--', label='Perfect Calibration')
            
            plt.xticks(bar_positions, results['ConfidenceBin'])
            plt.xlabel('Confidence Range', fontsize=14)
            plt.ylabel('Actual Accuracy (%)', fontsize=14)
            plt.title('Prediction Confidence vs. Actual Accuracy', fontsize=16)
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                   "Bars above the calibration line indicate underconfidence; below the line indicate overconfidence", 
                   ha="center", fontsize=11, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig('data/images/confidence_calibration.png', dpi=300)
        plt.show()
        
        # Add calibration metrics calculation
        calculate_calibration_metrics(log_df)
        
    except Exception as e:
        print(f"Error analyzing confidence vs accuracy: {e}")
        import traceback
        traceback.print_exc()

def calculate_model_specific_calibration(log_df):
    """
    Calculate confidence vs accuracy for each model based on their contribution.
    
    Args:
        log_df: DataFrame with prediction logs
        
    Returns:
        dict: Model-specific calibration data
    """
    model_data = {model: {'confidence': [], 'correct': []} for model in ALL_MODELS}
    
    # Process each prediction
    for i in range(len(log_df)):
        row = log_df.iloc[i]
        
        # Skip if no model contributions
        if 'model_contributions' not in row or pd.isna(row['model_contributions']):
            continue
            
        # Parse model contributions
        if isinstance(row['model_contributions'], str):
            try:
                model_contribs = json.loads(row['model_contributions'])
            except:
                continue
        elif isinstance(row['model_contributions'], dict):
            model_contribs = row['model_contributions']
        else:
            continue
            
        # Get prediction details
        confidence = row['Confidence']
        correct = row['Correct']
        
        # Assign to models based on their contribution
        for model, weight in model_contribs.items():
            if model in model_data and weight > 0:
                model_data[model]['confidence'].append(confidence)
                model_data[model]['correct'].append(correct)
    
    # Calculate accuracy by confidence bins for each model
    model_calibration = {}
    
    for model, data in model_data.items():
        if len(data['confidence']) < 10:  # Skip models with too little data
            continue
            
        # Create DataFrame for this model
        model_df = pd.DataFrame({
            'confidence': data['confidence'],
            'correct': data['correct']
        })
        
# Create confidence bins
        model_df['bin'] = pd.cut(model_df['confidence'], 
                               bins=[0, 40, 50, 60, 70, 80, 100],
                               labels=['0-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-100%'])
        
        # Calculate accuracy by bin
        bin_results = model_df.groupby('bin')['correct'].agg(['mean', 'count']).reset_index()
        
        # Get mean confidence for each bin
        bin_conf_means = model_df.groupby('bin')['confidence'].mean().reset_index()
        
        # Skip if less than 3 bins with data
        if len(bin_results) < 3:
            continue
            
        model_calibration[model] = {
            'bins': bin_results['bin'].astype(str).tolist(),
            'accuracy': (bin_results['mean'] * 100).tolist(),
            'counts': bin_results['count'].tolist(),
            'mean_confidence': bin_conf_means['confidence'].tolist()
        }
    
    return model_calibration

def calculate_calibration_metrics(log_df):
    """
    Calculate calibration metrics including Brier score and ECE.
    
    Args:
        log_df: DataFrame with prediction logs
        
    Returns:
        dict: Calibration metrics
    """
    if 'Confidence' not in log_df.columns or 'Correct' not in log_df.columns:
        return None
    
    # Calculate Brier Score (mean squared error between confidence and outcome)
    brier_score = np.mean((log_df['Confidence'] / 100 - log_df['Correct']) ** 2)
    
    # Calculate Expected Calibration Error (ECE)
    log_df['ConfidenceBin'] = pd.cut(log_df['Confidence'], 
                                   bins=10, 
                                   labels=False)
    
    ece = 0
    n_samples = len(log_df)
    
    for bin_idx in range(10):
        bin_mask = log_df['ConfidenceBin'] == bin_idx
        if sum(bin_mask) > 0:
            bin_confidence = log_df.loc[bin_mask, 'Confidence'].mean() / 100
            bin_accuracy = log_df.loc[bin_mask, 'Correct'].mean()
            bin_samples = sum(bin_mask)
            
            ece += (bin_samples / n_samples) * abs(bin_confidence - bin_accuracy)
    
    print("\nCalibration Metrics:")
    print(f"Brier Score: {brier_score:.4f} (lower is better, perfect = 0)")
    print(f"Expected Calibration Error: {ece:.4f} (lower is better, perfect = 0)")
    
    return {
        'brier_score': brier_score,
        'ece': ece
    }

def analyze_patterns(detailed=False):
    """
    Analyze patterns in the data to find potential biases or trends.
    
    Args:
        detailed: If True, perform more detailed pattern analysis
    """
    ensure_image_directory()
    if not os.path.exists(REALTIME_FILE):
        print("No real-time data found.")
        return
    
    try:
        df = pd.read_csv(REALTIME_FILE)
        
        # Check for streaks (consecutive same outcomes)
        print("Looking for streaks in the data...")
        for outcome, name in [(0, 'Banker'), (1, 'Player'), (2, 'Tie')]:
            max_streak = 0
            current_streak = 0
            
            for i in range(len(df)):
                if df.iloc[i]['Target'] == outcome:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            
            print(f"Longest {name} streak: {max_streak}")
        
        # Check if certain patterns of previous outcomes predict next outcome
        print("\nAnalyzing common patterns...")
        
        # Make sure required columns exist
        prev_cols = [f'Prev_{i+1}' for i in range(5)]
        if not all(col in df.columns for col in prev_cols):
            print("Data is missing previous outcome columns")
            return
        
        pattern_counts = {}
        for i in range(len(df)):
            pattern_key = tuple(df.iloc[i][prev_cols].values)
            outcome = df.iloc[i]['Target']
            
            if pattern_key in pattern_counts:
                pattern_counts[pattern_key][outcome] = pattern_counts[pattern_key].get(outcome, 0) + 1
            else:
                pattern_counts[pattern_key] = {outcome: 1}
        
        # Find patterns with strong biases
        strong_patterns = []
        for pattern, outcomes in pattern_counts.items():
            if sum(outcomes.values()) >= 3:  # Only consider patterns seen at least 3 times
                max_outcome = max(outcomes, key=outcomes.get)
                max_count = outcomes[max_outcome]
                total = sum(outcomes.values())
                if max_count / total >= 0.70:  # 70% or more towards one outcome
                    pattern_names = [['B', 'P', 'T'][p] for p in pattern]
                    outcome_name = ['Banker', 'Player', 'Tie'][max_outcome]
                    strong_patterns.append({
                        'pattern': ''.join(pattern_names),
                        'outcome': outcome_name,
                        'probability': max_count / total * 100,
                        'occurrences': total
                    })
        
        # Show strongest patterns
        if strong_patterns:
            print("\nPotentially significant patterns:")
            for p in sorted(strong_patterns, key=lambda x: (x['probability'], x['occurrences']), reverse=True)[:10]:
                print(f"  Pattern {p['pattern']} → {p['outcome']} ({p['probability']:.1f}%, seen {p['occurrences']} times)")
            
            # Visualize top patterns
            if len(strong_patterns) >= 3:
                top_patterns = sorted(strong_patterns, key=lambda x: (x['probability'], x['occurrences']), reverse=True)[:8]
                
                plt.figure(figsize=(12, 6))
                pattern_labels = [p['pattern'] for p in top_patterns]
                pattern_probabilities = [p['probability'] for p in top_patterns]
                pattern_occurrences = [p['occurrences'] for p in top_patterns]
                
                # Create colormap based on probability
                colors = plt.cm.RdYlGn(np.array(pattern_probabilities) / 100)
                
                # Create bar chart
                bars = plt.bar(range(len(top_patterns)), pattern_probabilities, color=colors)
                
                # Annotate with occurrence counts
                for i, bar in enumerate(bars):
                    plt.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + 2,
                           f"{pattern_occurrences[i]} times", 
                           ha='center', fontsize=9)
                    
                    # Also show the outcome
                    plt.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() / 2,
                           f"→ {top_patterns[i]['outcome']}", 
                           ha='center', fontsize=9, color='black',
                           fontweight='bold')
                
                plt.xticks(range(len(top_patterns)), pattern_labels)
                plt.xlabel('Pattern (sequence of past 5 outcomes)', fontsize=12)
                plt.ylabel('Prediction Accuracy (%)', fontsize=12)
                plt.title('Top Predictive Patterns', fontsize=14)
                plt.ylim(0, 105)  # Give space for annotations
                plt.grid(axis='y', alpha=0.3)
                
                # Add explanation (B=Banker, P=Player, T=Tie)
                plt.figtext(0.5, 0.01, 
                          "B=Banker, P=Player, T=Tie. Patterns read from left to right (most recent outcome first)", 
                          ha="center", fontsize=10, 
                          bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
                
                plt.tight_layout(rect=[0, 0.05, 1, 1])
                plt.savefig('data/images/predictive_patterns.png', dpi=300)
                plt.show()
        else:
            print("No strong patterns found in the current data.")
            
        # Analyze pattern effectiveness using prediction log if available
        if os.path.exists(LOG_FILE) and detailed:
            analyze_pattern_effectiveness()
    
    except Exception as e:
        print(f"Error analyzing patterns: {e}")
        import traceback
        traceback.print_exc()

def analyze_pattern_effectiveness(breakdown_by_model=True):
    """
    Analyze effectiveness of different pattern types with model breakdown.
    
    Args:
        breakdown_by_model: If True, show which models perform best on each pattern
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check if pattern_type column exists
        if 'pattern_type' not in log_df.columns:
            print("Log file doesn't contain pattern type information")
            return
        
        # Calculate accuracy by pattern type
        pattern_results = {}
        overall_accuracy = log_df['Correct'].mean() * 100
        
        for pattern in log_df['pattern_type'].unique():
            if pd.isna(pattern):
                continue
                
            pattern_df = log_df[log_df['pattern_type'] == pattern]
            
            # Skip if too few examples
            if len(pattern_df) < 5:
                continue
                
            pattern_accuracy = pattern_df['Correct'].mean() * 100
            
            # Calculate model-specific accuracy if requested
            model_breakdown = {}
            if breakdown_by_model and 'model_contributions' in log_df.columns:
                # Extract model contributions from JSON strings
                model_contribs = []
                for _, row in pattern_df.iterrows():
                    if isinstance(row['model_contributions'], str):
                        try:
                            contrib = json.loads(row['model_contributions'])
                            model_contribs.append((row['Correct'], contrib))
                        except:
                            pass
                    elif isinstance(row['model_contributions'], dict):
                        model_contribs.append((row['Correct'], row['model_contributions']))
                
                # Calculate model-specific accuracy
                if model_contribs:
                    for model in ALL_MODELS:
                        # Only include predictions where this model contributed
                        model_preds = [(correct, contribs[model]) for correct, contribs in model_contribs 
                                   if model in contribs and contribs[model] > 0]
                        
                        if model_preds:
                            # Weight by contribution
                            correct_weighted = sum(correct * weight for correct, weight in model_preds)
                            total_weight = sum(weight for _, weight in model_preds)
                            
                            if total_weight > 0:
                                model_acc = (correct_weighted / total_weight) * 100
                                model_breakdown[model] = {
                                    'accuracy': model_acc,
                                    'predictions': len(model_preds)
                                }
            
            pattern_results[pattern] = {
                'accuracy': pattern_accuracy,
                'sample_count': len(pattern_df),
                'model_breakdown': model_breakdown,
                'difference': pattern_accuracy - overall_accuracy
            }
        
        # Create visual comparison
        if pattern_results:
            plt.figure(figsize=(14, 8))
            
            # Sort patterns by accuracy
            sorted_patterns = sorted(pattern_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            
            # Set up bar positions
            pattern_labels = [p[0] for p in sorted_patterns]
            accuracy_values = [p[1]['accuracy'] for p in sorted_patterns]
            sample_counts = [p[1]['sample_count'] for p in sorted_patterns]
            
            # Create bar chart
            bars = plt.bar(range(len(pattern_labels)), accuracy_values,
                         color=[PATTERN_COLORS.get(p, '#333333') for p in pattern_labels])
            
            # Add reference line for overall accuracy
            plt.axhline(y=overall_accuracy, color='red', linestyle='--', 
                       label=f'Overall: {overall_accuracy:.1f}%')
            
            # Add sample count annotations
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, 
                       min(bar.get_height() + 5, 95),
                       f"n={sample_counts[i]}", 
                       ha='center', va='bottom', fontsize=10)
            
            plt.xticks(range(len(pattern_labels)), pattern_labels)
            plt.xlabel('Pattern Type', fontsize=14)
            plt.ylabel('Accuracy (%)', fontsize=14)
            plt.title('Prediction Accuracy by Pattern Type', fontsize=16)
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            
            # If model breakdown is available and requested, add subplot with model comparison
            if breakdown_by_model and any('model_breakdown' in result and result['model_breakdown'] 
                                       for _, result in pattern_results.items()):
                # Create subplot for model breakdown
                plt.figure(figsize=(16, 10))
                
                # Get unique patterns and models
                patterns = list(pattern_results.keys())
                models = []
                for _, result in pattern_results.items():
                    if 'model_breakdown' in result:
                        models.extend(result['model_breakdown'].keys())
                models = sorted(set(models))
                
                # Prepare data
                data_dict = {model: [] for model in models}
                
                # Fill in data
                for pattern in patterns:
                    result = pattern_results[pattern]
                    breakdown = result.get('model_breakdown', {})
                    
                    for model in models:
                        if model in breakdown:
                            data_dict[model].append(breakdown[model]['accuracy'])
                        else:
                            data_dict[model].append(0)
                
                # Create grouped bar chart
                bar_width = 0.8 / len(models)
                for i, model in enumerate(models):
                    if data_dict[model] and any(x > 0 for x in data_dict[model]):
                        position = np.arange(len(patterns)) + (i - len(models)/2 + 0.5) * bar_width
                        plt.bar(position, data_dict[model], bar_width, 
                               label=model, color=MODEL_COLORS.get(model))
                
                plt.axhline(y=overall_accuracy, color='red', linestyle='--', 
                           label=f'Overall: {overall_accuracy:.1f}%')
                
                plt.xticks(range(len(patterns)), patterns)
                plt.xlabel('Pattern Type', fontsize=14)
                plt.ylabel('Accuracy (%)', fontsize=14)
                plt.title('Model-Specific Accuracy by Pattern Type', fontsize=16)
                plt.ylim(0, 100)
                plt.grid(axis='y', alpha=0.3)
                plt.legend(title='Model', title_fontsize=12)
                
                plt.tight_layout()
                plt.savefig('data/images/model_pattern_breakdown.png', dpi=300)
            
            plt.tight_layout()
            plt.savefig('data/images/pattern_effectiveness.png', dpi=300)
            plt.show()
            
            # Print detailed results
            print("\nPattern Effectiveness Analysis:")
            print(f"Overall accuracy: {overall_accuracy:.2f}%")
            
            for pattern, result in sorted_patterns:
                diff = result['difference']
                diff_str = f"{'+' if diff > 0 else ''}{diff:.2f}%"
                print(f"  {pattern}: {result['accuracy']:.2f}% ({diff_str} vs overall, n={result['sample_count']})")
                
                # Print model breakdown if available
                if breakdown_by_model and 'model_breakdown' in result and result['model_breakdown']:
                    model_results = sorted(result['model_breakdown'].items(), 
                                          key=lambda x: x[1]['accuracy'], reverse=True)
                    
                    print("    Model breakdown:")
                    for model, model_result in model_results:
                        model_diff = model_result['accuracy'] - overall_accuracy
                        model_diff_str = f"{'+' if model_diff > 0 else ''}{model_diff:.2f}%"
                        print(f"      {model}: {model_result['accuracy']:.2f}% ({model_diff_str}, n={model_result['predictions']})")
            
            return pattern_results
        else:
            print("No pattern effectiveness data available.")
            return None
    
    except Exception as e:
        print(f"Error analyzing pattern effectiveness: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_streak_behavior():
    """
    Analyze how outcomes behave after streaks of the same result.
    This can help identify if there's a tendency to continue or break streaks.
    """
    ensure_image_directory()
    if not os.path.exists(REALTIME_FILE):
        print("No real-time data found.")
        return
    
    try:
        df = pd.read_csv(REALTIME_FILE)
        
        # Check if required columns exist
        if 'Target' not in df.columns:
            print("Data doesn't have Target column")
            return
        
        # Convert data to sequence for streak analysis
        sequence = df['Target'].tolist()
        
        # Analyze what happens after streaks of different lengths
        streak_results = {0: {}, 1: {}, 2: {}}  # Results indexed by outcome type
        
        for outcome_type in [0, 1, 2]:  # Banker, Player, Tie
            outcome_name = {0: 'Banker', 1: 'Player', 2: 'Tie'}[outcome_type]
            print(f"\nAnalyzing streaks of {outcome_name}:")
            
            for streak_length in range(1, 4):  # Analyze streaks of length 1-3
                # Find all instances where we have a streak of this length
                current_streak = 0
                after_streak = []
                
                for i in range(len(sequence)):
                    if sequence[i] == outcome_type:
                        current_streak += 1
                    else:
                        # If we just ended a streak of the target length, record what came next
                        if current_streak == streak_length and i < len(sequence) - 1:
                            after_streak.append(sequence[i+1])
                        current_streak = 0
                
                # Calculate statistics if we found any streaks
                if after_streak:
                    total = len(after_streak)
                    counts = {k: after_streak.count(k) for k in set(after_streak)}
                    
                    print(f"  After {streak_length} {outcome_name}(s) in a row ({total} occurrences):")
                    for next_outcome, count in counts.items():
                        next_name = {0: 'Banker', 1: 'Player', 2: 'Tie'}[next_outcome]
                        percentage = (count / total) * 100
                        print(f"    {next_name}: {count}/{total} ({percentage:.1f}%)")
                    
                    streak_results[outcome_type][streak_length] = counts
                else:
                    print(f"  No streaks of {streak_length} {outcome_name}(s) found")
        
        # Visualize streak behavior if we have data
        has_data = any(len(streak_results[outcome]) > 0 for outcome in [0, 1, 2])
        if has_data:
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            outcome_names = {0: 'Banker', 1: 'Player', 2: 'Tie'}
            colors = {0: 'green', 1: 'blue', 2: 'red'}
            
            for outcome, ax in zip([0, 1, 2], axes):
                outcome_name = outcome_names[outcome]
                ax.set_title(f"After {outcome_name} Streaks", fontsize=14)
                ax.set_xlabel("Next Outcome", fontsize=12)
                ax.set_ylabel("Probability (%)", fontsize=12)
                
                # For each streak length
                bar_width = 0.25
                index = np.arange(3)  # 3 possible next outcomes
                
                for i, streak_length in enumerate([1, 2, 3]):
                    if streak_length in streak_results[outcome]:
                        counts = streak_results[outcome][streak_length]
                        total = sum(counts.values())
                        
                        # Calculate percentages for each possible next outcome
                        percentages = []
                        for next_outcome in [0, 1, 2]:
                            percentage = (counts.get(next_outcome, 0) / total) * 100 if total > 0 else 0
                            percentages.append(percentage)
                        
                        # Plot bar for this streak length
                        position = index + (i - 1) * bar_width
                        bars = ax.bar(position, percentages, bar_width, 
                                     label=f"{streak_length} in a row",
                                     alpha=0.7)
                        
                        # Add count labels if values are significant
                        for j, bar in enumerate(bars):
                            if bar.get_height() > 10:  # Only show label if bar is tall enough
                                count = counts.get(j, 0)
                                ax.text(bar.get_x() + bar.get_width()/2, 
                                      bar.get_height() + 2,
                                      f"{count}", 
                                      ha='center', fontsize=8)
                
                ax.set_xticks(index)
                ax.set_xticklabels([outcome_names[0], outcome_names[1], outcome_names[2]])
                ax.legend()
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('data/images/streak_behavior.png', dpi=300)
            plt.show()
        
        return streak_results
    
    except Exception as e:
        print(f"Error analyzing streak behavior: {e}")
        import traceback
        traceback.print_exc()

def plot_temporal_pattern():
    """
    Analyze how outcomes distribute over time to identify potential cycles.
    """
    ensure_image_directory()
    if not os.path.exists(REALTIME_FILE):
        print("No real-time data found.")
        return
    
    try:
        df = pd.read_csv(REALTIME_FILE)
        
        if 'Target' not in df.columns:
            print("Data doesn't have Target column")
            return
            
        # Create a time-series representation of outcomes
        sequence = df['Target'].tolist()
        
        if len(sequence) < 20:
            print("Not enough data for temporal analysis (minimum 20 points needed)")
            return
        
        # Create a rolling window of outcomes for banker and player
        window_size = 10
        banker_ratio = []
        player_ratio = []
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            banker_ratio.append(window.count(0) / window_size * 100)
            player_ratio.append(window.count(1) / window_size * 100)
        
        # Plot the temporal pattern
        plt.figure(figsize=(14, 7))
        
        # Plot banker and player ratios
        plt.plot(range(len(banker_ratio)), banker_ratio, 'g-', label='Banker %', linewidth=2)
        plt.plot(range(len(player_ratio)), player_ratio, 'b-', label='Player %', linewidth=2)
        
        # Add 50% reference line
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% line')
        
        # Add an alternating pattern reference
        if len(banker_ratio) > 20:
            alternating = [60 if i % 2 == 0 else 40 for i in range(len(banker_ratio))]
            plt.plot(range(len(alternating)), alternating, 'r:', alpha=0.5, label='Alternating pattern')
        
        # Enhance the plot
        plt.title('Temporal Pattern of Outcomes', fontsize=16)
        plt.xlabel('Game Sequence (Rolling Window)', fontsize=14)
        plt.ylabel('Percentage in Window (%)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                   f"Analysis based on {window_size}-game rolling window", 
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('data/images/temporal_pattern.png', dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error analyzing temporal pattern: {e}")
        import traceback
        traceback.print_exc()

def analyze_markov_transitions():
    """
    Analyze first-order Markov transition probabilities in the data.
    """
    ensure_image_directory()
    if not os.path.exists(REALTIME_FILE):
        print("No real-time data found.")
        return
    
    try:
        df = pd.read_csv(REALTIME_FILE)
        
        if 'Target' not in df.columns:
            print("Data doesn't have Target column")
            return
            
        # Create a sequence of outcomes
        sequence = df['Target'].tolist()
        
        if len(sequence) < 10:
            print("Not enough data for Markov analysis (minimum 10 points needed)")
            return
        
        # Calculate transition counts
        transitions = np.zeros((3, 3))
        state_counts = [0, 0, 0]
        
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i+1]
            
            if current_state in [0, 1, 2] and next_state in [0, 1, 2]:
                transitions[current_state][next_state] += 1
                state_counts[current_state] += 1
        
        # Calculate transition probabilities
        transition_probs = np.zeros((3, 3))
        for i in range(3):
            if state_counts[i] > 0:
                transition_probs[i] = transitions[i] / state_counts[i]
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_probs, annot=True, fmt='.2f', cmap='YlGnBu',
                   xticklabels=['Banker', 'Player', 'Tie'],
                   yticklabels=['Banker', 'Player', 'Tie'])
        
        plt.title('Markov Transition Probabilities', fontsize=16)
        plt.xlabel('Next Outcome', fontsize=14)
        plt.ylabel('Current Outcome', fontsize=14)
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                   "Values represent probability of transitioning from row state to column state", 
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('data/images/markov_transitions.png', dpi=300)
        plt.show()
        
        # Return the transition matrix for further analysis
        return transition_probs
        
    except Exception as e:
        print(f"Error analyzing Markov transitions: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_prediction_distribution():
    """
    Analyze distribution of prediction outcomes vs actual outcomes.
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        if 'Predicted' not in log_df.columns or 'Actual' not in log_df.columns:
            print("Log file doesn't have complete prediction data")
            return
        
        # Count predictions and actual outcomes
        pred_counts = log_df['Predicted'].value_counts().sort_index()
        actual_counts = log_df['Actual'].value_counts().sort_index()
        
        # Ensure all outcomes are represented
        for i in range(3):
            if i not in pred_counts.index:
                pred_counts[i] = 0
            if i not in actual_counts.index:
                actual_counts[i] = 0
        
        pred_counts = pred_counts.sort_index()
        actual_counts = actual_counts.sort_index()
        
        # Create bar chart comparing predictions vs actuals
        labels = ['Banker', 'Player', 'Tie']
        colors = ['green', 'blue', 'red']
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        pred_bars = ax.bar(x - width/2, pred_counts, width, label='Predicted', 
                 color=colors, alpha=0.7, edgecolor=colors, linewidth=1.5)
        actual_bars = ax.bar(x + width/2, actual_counts, width, label='Actual', 
                           color=colors, alpha=0.8)
        
        # Add counts and percentages
        total_preds = sum(pred_counts)
        total_actuals = sum(actual_counts)
        
        for i, bars in enumerate(zip(pred_bars, actual_bars)):
            pred_bar, actual_bar = bars
            
            pred_pct = (pred_counts[i] / total_preds) * 100 if total_preds > 0 else 0
            actual_pct = (actual_counts[i] / total_actuals) * 100 if total_actuals > 0 else 0
            
        
        # Add chart details
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title('Predicted vs Actual Outcome Distribution', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add accuracy info
        accuracy = (log_df['Predicted'] == log_df['Actual']).mean() * 100
        
        # Calculate class-specific accuracy
        class_accuracy = []
        for outcome in range(3):
            class_df = log_df[log_df['Actual'] == outcome]
            if len(class_df) > 0:
                acc = (class_df['Predicted'] == class_df['Actual']).mean() * 100
                class_accuracy.append(f"{labels[outcome]}: {acc:.1f}%")
        
        # Add text box with accuracy info
        ax.text(0.02, 0.97, 
              f"Overall accuracy: {accuracy:.2f}%\n" + "\n".join(class_accuracy),
              transform=ax.transAxes, fontsize=10,
              verticalalignment='top', horizontalalignment='left',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('data/images/prediction_distribution.png', dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error plotting prediction distribution: {e}")
        import traceback
        traceback.print_exc()

def analyze_model_contributions():
    """
    Analyze how different models contribute to predictions.
    
    Returns:
        dict: Contribution metrics by model
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return
        
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        if 'model_contributions' not in log_df.columns:
            print("Log file doesn't contain model contribution data")
            return
            
        # Extract model contributions from each prediction
        all_contribs = defaultdict(list)  # Contributions for all predictions
        correct_contribs = defaultdict(list)  # Contributions for correct predictions
        incorrect_contribs = defaultdict(list)  # Contributions for incorrect predictions
        
        for i, row in log_df.iterrows():
            try:
                # Parse contributions data
                if isinstance(row['model_contributions'], str):
                    contribs = json.loads(row['model_contributions'])
                elif isinstance(row['model_contributions'], dict):
                    contribs = row['model_contributions']
                else:
                    continue
                    
                correct = row['Correct'] if 'Correct' in row else (row['Predicted'] == row['Actual'])
                
                # Record contributions by correctness
                for model, contrib in contribs.items():
                    all_contribs[model].append(contrib)
                    
                    if correct:
                        correct_contribs[model].append(contrib)
                    else:
                        incorrect_contribs[model].append(contrib)
                        
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue
        
        # Calculate average contributions
        avg_all_contribs = {model: np.mean(contribs) for model, contribs in all_contribs.items()}
        avg_correct_contribs = {model: np.mean(contribs) if contribs else 0 
                              for model, contribs in correct_contribs.items()}
        avg_incorrect_contribs = {model: np.mean(contribs) if contribs else 0 
                                for model, contribs in incorrect_contribs.items()}
        
        # Calculate model effectiveness (ratio of correct to incorrect contributions)
        model_effectiveness = {}
        for model in all_contribs:
            if model in avg_incorrect_contribs and avg_incorrect_contribs[model] > 0:
                effectiveness = avg_correct_contribs.get(model, 0) / avg_incorrect_contribs[model]
            else:
                effectiveness = avg_correct_contribs.get(model, 0) * 2  # High value if no incorrect
                
            model_effectiveness[model] = effectiveness
        
        # Calculate prediction count and accuracy by model
        model_predictions = {model: len(contribs) for model, contribs in all_contribs.items()}
        model_accuracy = {}
        for model in all_contribs:
            correct = len(correct_contribs.get(model, []))
            total = model_predictions[model]
            model_accuracy[model] = (correct / total) * 100 if total > 0 else 0
        
        # Compile results
        contribution_results = {
            'avg_all_contributions': avg_all_contribs,
            'avg_correct_contributions': avg_correct_contribs,
            'avg_incorrect_contributions': avg_incorrect_contribs,
            'model_effectiveness': model_effectiveness,
            'model_predictions': model_predictions,
            'model_accuracy': model_accuracy
        }
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Overall contribution distribution
        plt.subplot(2, 2, 1)
        models = list(avg_all_contribs.keys())
        contributions = list(avg_all_contribs.values())
        
        # Sort by contribution
        sorted_indices = np.argsort(contributions)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_contributions = [contributions[i] for i in sorted_indices]
        
        bars = plt.bar(range(len(sorted_models)), sorted_contributions, 
                     color=[MODEL_COLORS.get(model, '#333333') for model in sorted_models])
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{sorted_contributions[i]:.3f}", ha='center', va='bottom', fontsize=9)
            
        plt.title('Overall Model Contributions', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Average Contribution', fontsize=12)
        plt.xticks(range(len(sorted_models)), sorted_models, rotation=45, ha='right')
        plt.tight_layout()
        
        # Plot 2: Correct vs Incorrect contributions
        plt.subplot(2, 2, 2)
        
        # Get models with both correct and incorrect data
        common_models = [m for m in models if m in avg_correct_contribs and m in avg_incorrect_contribs]
        
        # Sort by effectiveness
        common_models.sort(key=lambda m: model_effectiveness.get(m, 0), reverse=True)
        
        x = np.arange(len(common_models))
        width = 0.35
        
        # Plot correct contributions
        correct_vals = [avg_correct_contribs.get(m, 0) for m in common_models]
        incorrect_vals = [avg_incorrect_contribs.get(m, 0) for m in common_models]
        
        plt.bar(x - width/2, correct_vals, width, label='Correct Predictions', color='green', alpha=0.7)
        plt.bar(x + width/2, incorrect_vals, width, label='Incorrect Predictions', color='red', alpha=0.7)
        
        plt.title('Contribution to Correct vs Incorrect Predictions', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Average Contribution', fontsize=12)
        plt.xticks(x, common_models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Plot 3: Model effectiveness
        plt.subplot(2, 2, 3)
        
        # Sort models by effectiveness
        sorted_models_by_eff = sorted(model_effectiveness.items(), key=lambda x: x[1], reverse=True)
        eff_models = [m[0] for m in sorted_models_by_eff]
        eff_values = [m[1] for m in sorted_models_by_eff]
        
        bars = plt.bar(range(len(eff_models)), eff_values, 
                     color=[MODEL_COLORS.get(model, '#333333') for model in eff_models])
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f"{eff_values[i]:.2f}", ha='center', va='bottom', fontsize=9)
            
        plt.title('Model Effectiveness (Correct/Incorrect Contribution Ratio)', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Effectiveness Ratio', fontsize=12)
        plt.xticks(range(len(eff_models)), eff_models, rotation=45, ha='right')
        plt.tight_layout()
        
        # Plot 4: Model accuracy
        plt.subplot(2, 2, 4)
        
        # Sort models by accuracy
        sorted_models_by_acc = sorted(model_accuracy.items(), key=lambda x: x[1], reverse=True)
        acc_models = [m[0] for m in sorted_models_by_acc]
        acc_values = [m[1] for m in sorted_models_by_acc]
        prediction_counts = [model_predictions[m] for m in acc_models]
        
        # Normalize marker size by prediction count
        min_size = 50
        max_size = 500
        if max(prediction_counts) > min(prediction_counts):
            norm_sizes = [min_size + (max_size - min_size) * (count - min(prediction_counts)) / 
                        (max(prediction_counts) - min(prediction_counts)) 
                        for count in prediction_counts]
        else:
            norm_sizes = [min_size + (max_size - min_size) / 2 for _ in prediction_counts]
        
        # Scatter plot with size representing prediction count
        plt.scatter(range(len(acc_models)), acc_values, s=norm_sizes, 
                  c=[MODEL_COLORS.get(model, '#333333') for model in acc_models], alpha=0.7)
        
        # Add model names and values
        for i, (model, acc) in enumerate(zip(acc_models, acc_values)):
            plt.text(i, acc + 2, f"{model}\n{acc:.1f}%\n(n={prediction_counts[i]})", 
                   ha='center', va='bottom', fontsize=9)
            
        plt.title('Model Accuracy (Size = Prediction Count)', fontsize=14)
        plt.xlabel('Model Rank', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xticks([])  # Hide x-ticks since we have text labels
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/images/model_contributions.png', dpi=300)
        plt.show()
        
        # Print detailed analysis
        print("\nModel Contribution Analysis:")
        print(f"{'Model':<20} {'Contribution':<15} {'Accuracy':<10} {'Effectiveness':<15} {'Predictions':<10}")
        print("-" * 70)
        
        # Sort by overall contribution
        for model, contrib in sorted(avg_all_contribs.items(), key=lambda x: x[1], reverse=True):
            acc = model_accuracy.get(model, 0)
            eff = model_effectiveness.get(model, 0)
            preds = model_predictions.get(model, 0)
            
            print(f"{model:<20} {contrib:.4f}{' '*8} {acc:.1f}%{' '*4} {eff:.2f}{' '*9} {preds}")
            
        return contribution_results
        
    except Exception as e:
        print(f"Error analyzing model contributions: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_meta_feature_importance():
    """
    Analyze which meta-features are most important in the stacking ensemble.
    Requires access to the stacking model in registry.
    """
    ensure_image_directory()
    
    try:
        # Load model registry to get stacking model
        registry_file = os.path.join(MODEL_REGISTRY_PATH, "registry.json")
        if not os.path.exists(registry_file):
            print("Model registry not found.")
            return
            
        # Try to load registry data
        with open(registry_file, 'r') as f:
            registry_data = json.load(f)
            
        # Check if stacking ensemble exists
        if "stacking_ensemble" not in registry_data.get("model_ids", []):
            print("Stacking ensemble not found in registry.")
            return
            
        # Try to load stacking model
        stacking_file = os.path.join(MODEL_REGISTRY_PATH, "stacking_ensemble.pkl")
        if not os.path.exists(stacking_file):
            print("Stacking model file not found.")
            return
            
        with open(stacking_file, 'rb') as f:
            stacking_model = pickle.load(f)
            
        # Check if model has LogisticRegression meta_model with coefficients
        if not hasattr(stacking_model, 'meta_model') or not hasattr(stacking_model.meta_model, 'coef_'):
            print("Stacking model doesn't have accessible coefficients.")
            return
            
        # Check if model was trained
        if not getattr(stacking_model, 'is_trained', False):
            print("Stacking model hasn't been trained yet.")
            return
            
        # Extract feature importance from coefficients
        coef = stacking_model.meta_model.coef_
        
        # Calculate average absolute importance across classes
        importance = np.abs(coef).mean(axis=0)
        
        # Generate feature names
        base_models = [m for m in registry_data.get("model_ids", []) if m != "stacking_ensemble"]
        pattern_types = ['no_pattern', 'streak', 'alternating', 'tie']
        
        # Construct meta-feature names
        feature_names = []
        for model in base_models:
            for outcome in ['Banker', 'Player', 'Tie']:
                feature_names.append(f"{model}_{outcome}")
                
        # Add pattern feature names
        for pattern in pattern_types:
            feature_names.append(f"pattern_{pattern}")
            
        # Check if feature count matches
        if len(feature_names) != len(importance):
            print(f"Warning: Feature name count ({len(feature_names)}) doesn't match coefficient count ({len(importance)})")
            
            # Try to adjust the feature names
            if len(importance) > len(feature_names):
                # Add placeholder names for extra features
                for i in range(len(importance) - len(feature_names)):
                    feature_names.append(f"unknown_feature_{i+1}")
            else:
                # Truncate feature names
                feature_names = feature_names[:len(importance)]
                
        # Create importance dictionary
        importance_dict = {feature: float(imp) for feature, imp in zip(feature_names, importance)}
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Visualize feature importance
        plt.figure(figsize=(14, 8))
        
        # Plot top 20 features or all if less
        top_n = min(20, len(sorted_importance))
        top_features = sorted_importance[:top_n]
        
        # Feature names and values
        names = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        
        # Determine feature types and colors
        colors = []
        for name in names:
            if 'pattern_' in name:
                colors.append('#8B0000')  # Dark red for pattern features
            else:
                model_name = name.split('_')[0]
                colors.append(MODEL_COLORS.get(model_name, '#333333'))
                
        # Create horizontal bar chart
        bars = plt.barh(range(len(names)), values, color=colors, alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{values[i]:.4f}", va='center', fontsize=9)
            
        plt.yticks(range(len(names)), names)
        plt.title('Meta-Feature Importance in Stacking Ensemble', fontsize=16)
        plt.xlabel('Importance (Mean Absolute Coefficient)', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        
        # Add legend for feature types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#8B0000', lw=4, label='Pattern Features'),
        ]
        
        # Add one entry per model color
        for model in set([name.split('_')[0] for name in names if 'pattern_' not in name]):
            if model in MODEL_COLORS:
                legend_elements.append(
                    Line2D([0], [0], color=MODEL_COLORS[model], lw=4, label=f'{model} Features')
                )
                
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('data/images/meta_feature_importance.png', dpi=300)
        plt.show()
        
        # Print feature importance breakdown
        print("\nMeta-Feature Importance in Stacking Ensemble:")
        print(f"{'Feature':<30} {'Importance':<10}")
        print("-" * 40)
        
        for feature, importance in sorted_importance[:20]:  # Show top 20
            print(f"{feature:<30} {importance:.4f}")
        
        # Group by feature type
        print("\nImportance by Feature Type:")
        
        # Group by model
        model_importance = defaultdict(float)
        pattern_importance = 0
        
        for feature, importance in sorted_importance:
            if 'pattern_' in feature:
                pattern_importance += importance
            else:
                model_name = feature.split('_')[0]
                model_importance[model_name] += importance
                
        # Calculate total for percentage
        total_importance = sum(v for v in model_importance.values()) + pattern_importance
        
        # Print model importance
        for model, importance in sorted(model_importance.items(), key=lambda x: x[1], reverse=True):
            percentage = (importance / total_importance) * 100
            print(f"{model:<20} {importance:.4f} ({percentage:.1f}%)")
            
        # Print pattern importance
        percentage = (pattern_importance / total_importance) * 100
        print(f"{'Pattern Features':<20} {pattern_importance:.4f} ({percentage:.1f}%)")
        
        return {
            'feature_importance': sorted_importance,
            'model_importance': dict(model_importance),
            'pattern_importance': pattern_importance
        }
        
    except Exception as e:
        print(f"Error analyzing meta-feature importance: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_stacking_weights_over_time():
    """
    Visualize how the stacking ensemble weights models over time.
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return
        
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check if required columns exist
        if 'model_contributions' not in log_df.columns or 'Timestamp' not in log_df.columns:
            print("Log file doesn't have model contribution or timestamp data")
            return
            
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(log_df['Timestamp']):
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            
        # Extract model contributions and organize by timestamp
        timestamps = []
        model_weights = defaultdict(list)
        
        for _, row in log_df.iterrows():
            try:
                # Parse model contributions
                if isinstance(row['model_contributions'], str):
                    contributions = json.loads(row['model_contributions'])
                elif isinstance(row['model_contributions'], dict):
                    contributions = row['model_contributions']
                else:
                    continue
                    
                # Add to data
                timestamps.append(row['Timestamp'])
                
                # Record each model's weight
                for model in ALL_MODELS:
                    model_weights[model].append(contributions.get(model, 0))
            except:
                continue
                
        # Create DataFrame for time series
        weight_df = pd.DataFrame({
            'timestamp': timestamps,
            **{model: weights for model, weights in model_weights.items() if any(w > 0 for w in weights)}
        })
        
        if len(weight_df) < 5:
            print("Not enough data for meaningful visualization")
            return
            
        # Sort by timestamp
        weight_df = weight_df.sort_values('timestamp')
        
        # Create plot
        plt.figure(figsize=(14, 8))
        
        # Plot each model's weight over time
        for model, color in MODEL_COLORS.items():
            if model in weight_df.columns:
                plt.plot(weight_df['timestamp'], weight_df[model], 
                       label=model, color=color, linewidth=2, alpha=0.7)
                
        # Add rolling average for smoother visualization
        if len(weight_df) >= 10:
            window = min(10, len(weight_df) // 3)
            for model, color in MODEL_COLORS.items():
                if model in weight_df.columns:
                    rolling_avg = weight_df[model].rolling(window=window).mean()
                    plt.plot(weight_df['timestamp'], rolling_avg, 
                           color=color, linewidth=3, alpha=1.0, 
                           linestyle='--', label=f"{model} (smooth)")
        
        plt.title('Model Weights in Stacking Ensemble Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Weight', fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig('data/images/stacking_weights_over_time.png', dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing stacking weights: {e}")
        import traceback
        traceback.print_exc()

def ensure_image_directory():
    """
    Ensure the images directory exists.
    """
    os.makedirs("data/images", exist_ok=True)

def generate_full_report():
    """
    Generate a comprehensive enhanced analysis report with visualizations.
    """
    print("=== Enhanced Baccarat Prediction System Analysis Report ===")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Analyze prediction accuracy
    results = analyze_prediction_history()
    
    if results:
        print(f"\nOverall accuracy: {results['overall']['accuracy']:.2f}% ({results['overall']['correct']}/{results['overall']['total']})")
        
        print("\nAccuracy by actual outcome:")
        for outcome in ['Banker', 'Player', 'Tie']:
            if outcome in results:
                r = results[outcome]
                print(f"  {outcome}: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
        
        print("\nAccuracy by predicted outcome:")
        for outcome, r in results['by_prediction'].items():
            print(f"  When predicting {outcome}: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
            
        # Show confidence-based accuracy if available
        if results['by_confidence'] and len(results['by_confidence']) > 0:
            print("\nAccuracy by confidence level:")
            for conf_bin, r in sorted(results['by_confidence'].items()):
                print(f"  Confidence {conf_bin}: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
                
        # Show pattern-based accuracy if available
        if 'by_pattern' in results and results['by_pattern'] and len(results['by_pattern']) > 0:
            print("\nAccuracy by pattern type:")
            for pattern, r in results['by_pattern'].items():
                print(f"  Pattern {pattern}: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
                
        # Show model contributions if available
        if 'model_contributions' in results and results['model_contributions']:
            print("\nModel Contributions:")
            
            # Overall contributions
            all_contribs = results['model_contributions'].get('all_predictions', {})
            correct_contribs = results['model_contributions'].get('correct_predictions', {})
            
            # Sort by contribution
            for model, contrib in sorted(all_contribs.items(), key=lambda x: x[1], reverse=True):
                correct_contrib = correct_contribs.get(model, 0)
                print(f"  {model}: {contrib:.4f} overall, {correct_contrib:.4f} for correct predictions")
    
    # Analyze patterns
    print("\n=== Pattern Analysis ===")
    analyze_patterns()
    
    # Analyze pattern effectiveness with model breakdown
    print("\n=== Pattern Effectiveness Analysis ===")
    analyze_pattern_effectiveness(breakdown_by_model=True)
    
    # Analyze model contributions in detail
    print("\n=== Model Contribution Analysis ===")
    analyze_model_contributions()
    
    # Analyze meta-feature importance
    print("\n=== Meta-Feature Importance Analysis ===")
    analyze_meta_feature_importance()
    
    # Analyze streak behavior
    print("\n=== Streak Analysis ===")
    analyze_streak_behavior()
    
    # Analyze Markov transitions
    print("\n=== Markov Transition Analysis ===")
    transition_probs = analyze_markov_transitions()
    
    if transition_probs is not None:
        # Identify significant transitions
        for i in range(3):
            current = {0: 'Banker', 1: 'Player', 2: 'Tie'}[i]
            max_prob = max(transition_probs[i])
            max_idx = np.argmax(transition_probs[i])
            next_outcome = {0: 'Banker', 1: 'Player', 2: 'Tie'}[max_idx]
            
            if max_prob > 0.4:  # Only show if probability is significant
                print(f"  After {current}, most likely next outcome is {next_outcome} ({max_prob:.2f} probability)")
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    try:
        print("\n1. Accuracy Over Time:")
        plot_accuracy_over_time(breakdown_by_model=True)
        
        print("\n2. Outcome Distribution:")
        plot_outcome_distribution(compare_with_prediction=True)
        
        print("\n3. Prediction vs Actual Distribution:")
        plot_prediction_distribution()
        
        print("\n4. Confidence Calibration:")
        analyze_confidence_vs_accuracy(by_model=True)
        
        print("\n5. Temporal Patterns:")
        plot_temporal_pattern()
        
        print("\n6. Stacking Weights Over Time:")
        visualize_stacking_weights_over_time()
        
        print("\nVisualizations saved in the data/images directory")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== End of Report ===")

def run_analytics_menu():
    """
    Interactive menu for running enhanced analytics functions
    """
    while True:
        print("\n=== Enhanced Baccarat Prediction Analytics ===")
        print("Select an analysis to perform:")
        print("1. Basic prediction accuracy analysis")
        print("2. Visualize accuracy trend over time")
        print("3. Analyze outcome distribution")
        print("4. Analyze prediction patterns")
        print("5. Analyze streak behavior")
        print("6. Analyze Markov transitions")
        print("7. Analyze prediction confidence calibration")
        print("8. Plot prediction vs actual distribution")
        print("9. Analyze model contributions")
        print("10. Analyze meta-feature importance")
        print("11. Visualize stacking weights over time")
        print("12. Generate comprehensive report with all analyses")
        print("0. Exit to main menu")
        
        try:
            choice = input("\nEnter your choice (0-12): ")
            
            if choice == '1':
                results = analyze_prediction_history()
                if results:
                    print(f"\nOverall accuracy: {results['overall']['accuracy']:.2f}% ({results['overall']['correct']}/{results['overall']['total']})")
                    
                    print("\nAccuracy by actual outcome:")
                    for outcome in ['Banker', 'Player', 'Tie']:
                        if outcome in results:
                            r = results[outcome]
                            print(f"  {outcome}: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
                    
                    print("\nAccuracy by predicted outcome:")
                    for outcome, r in results['by_prediction'].items():
                        print(f"  When predicting {outcome}: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
            
            elif choice == '2':
                plot_accuracy_over_time(breakdown_by_model=True)
            
            elif choice == '3':
                plot_outcome_distribution(compare_with_prediction=True)
                
            elif choice == '4':
                analyze_patterns(detailed=True)
                
            elif choice == '5':
                analyze_streak_behavior()
                
            elif choice == '6':
                analyze_markov_transitions()
                
            elif choice == '7':
                analyze_confidence_vs_accuracy(by_model=True)
                
            elif choice == '8':
                plot_prediction_distribution()
                
            elif choice == '9':
                analyze_model_contributions()
                
            elif choice == '10':
                analyze_meta_feature_importance()
                
            elif choice == '11':
                visualize_stacking_weights_over_time()
                
            elif choice == '12':
                generate_full_report()
                
            elif choice == '0':
                print(f"{Fore.YELLOW}Returning to main menu...")
                return
            
            else:
                print(f"{Fore.RED}Invalid choice. Please enter a number between 0 and 12.")
                
            # Pause before showing menu again
            input("\nPress Enter to continue...")
                
        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}\nOperation cancelled.")
            return
        except Exception as e:
            print(f"{Fore.RED}Error: {e}")
            import traceback
            traceback.print_exc()

def analyze_model_performance(timeframe='all', model_types=None):
    """
    Analyze the performance of individual models and the ensemble over time.
    
    Args:
        timeframe: 'all', 'recent', or number of days
        model_types: List of specific model types to analyze, or None for all
        
    Returns:
        dict: Comprehensive model performance metrics
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check if required columns exist
        if 'model_contributions' not in log_df.columns:
            print("Log file doesn't contain model contribution data")
            return None
        
        # Apply time filtering
        if 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            
            if timeframe == 'recent':
                # Use the most recent 50 predictions
                log_df = log_df.sort_values('Timestamp').tail(50)
            elif isinstance(timeframe, int):
                # Filter by specified number of days
                cutoff_date = datetime.now() - timedelta(days=timeframe)
                log_df = log_df[log_df['Timestamp'] >= cutoff_date]
        
        # Extract model contributions
        model_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidence_sum': 0})
        model_predictions = []
        
        for _, row in log_df.iterrows():
            # Parse contributions
            if not pd.isna(row['model_contributions']):
                try:
                    if isinstance(row['model_contributions'], str):
                        contributions = json.loads(row['model_contributions'])
                    else:
                        contributions = row['model_contributions']
                        
                    # Track individual model predictions
                    actual = row['Actual']
                    predicted = row['Predicted']
                    correct = 1 if predicted == actual else 0
                    confidence = row.get('Confidence', 50.0)
                    
                    # Calculate weighted contributions to this prediction
                    for model, weight in contributions.items():
                        # Filter by model type if specified
                        if model_types and not any(mt in model for mt in model_types):
                            continue
                            
                        model_results[model]['total'] += weight
                        if correct:
                            model_results[model]['correct'] += weight
                        model_results[model]['confidence_sum'] += confidence * weight
                        
                        # Add to model predictions list for detailed analysis
                        model_predictions.append({
                            'model': model,
                            'weight': weight,
                            'predicted': predicted,
                            'actual': actual,
                            'correct': correct,
                            'confidence': confidence,
                            'timestamp': row.get('Timestamp')
                        })
                except Exception as e:
                    print(f"Error parsing model contributions: {e}")
        
        # Calculate model-specific metrics
        model_metrics = {}
        for model, stats in model_results.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                avg_confidence = stats['confidence_sum'] / stats['total']
                
                # Calculate calibration (difference between confidence and accuracy)
                calibration_error = abs(avg_confidence - accuracy)
                
                model_metrics[model] = {
                    'accuracy': accuracy,
                    'weighted_samples': stats['total'],
                    'average_confidence': avg_confidence,
                    'calibration_error': calibration_error
                }
        
        # Create performance visualizations
        if model_metrics:
            # Sort models by accuracy
            sorted_models = sorted(
                [(model, metrics['accuracy']) for model, metrics in model_metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Plot comparative model performance
            plt.figure(figsize=(12, 8))
            
            # Set up bar positions and data
            models = [m[0] for m in sorted_models]
            accuracies = [m[1] for m in sorted_models]
            
            # Create bar chart with consistent model colors
            bars = plt.bar(
                range(len(models)), 
                accuracies, 
                color=[MODEL_COLORS.get(m, '#777777') for m in models]
            )
            
            # Add sample count annotations
            for i, bar in enumerate(bars):
                sample_count = model_metrics[models[i]]['weighted_samples']
                plt.text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 1, 
                    f"n={sample_count:.1f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=9
                )
            
            # Enhance chart appearance
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title('Model Performance Comparison', fontsize=14)
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)
            
            # Add reference line for overall accuracy
            overall_acc = log_df['Correct'].mean() * 100
            plt.axhline(y=overall_acc, color='red', linestyle='--', 
                       label=f'Overall: {overall_acc:.1f}%')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('data/images/model_performance_comparison.png', dpi=300)
            plt.show()
            
            # Plot calibration chart (confidence vs accuracy)
            plt.figure(figsize=(10, 8))
            
            # Extract calibration data
            model_names = []
            confidences = []
            accuracies = []
            
            for model, metrics in model_metrics.items():
                model_names.append(model)
                confidences.append(metrics['average_confidence'])
                accuracies.append(metrics['accuracy'])
            
            # Plot each model as a point
            scatter = plt.scatter(
                confidences, 
                accuracies, 
                c=[MODEL_COLORS.get(m, '#777777') for m in model_names],
                s=100,
                alpha=0.7
            )
            
            # Add perfect calibration reference line
            max_val = max(max(confidences), max(accuracies)) + 5
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Calibration')
            
            # Add model labels
            for i, model in enumerate(model_names):
                plt.annotate(
                    model, 
                    (confidences[i], accuracies[i]),
                    xytext=(7, 4),
                    textcoords='offset points',
                    fontsize=9
                )
            
            plt.xlabel('Average Confidence (%)', fontsize=12)
            plt.ylabel('Actual Accuracy (%)', fontsize=12)
            plt.title('Model Calibration: Confidence vs. Accuracy', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add explanatory text
            plt.figtext(0.5, 0.01, 
                       "Points above the line: Underconfident models\nPoints below the line: Overconfident models", 
                       ha="center", fontsize=10, 
                       bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.savefig('data/images/model_calibration.png', dpi=300)
            plt.show()
        
        # Return detailed analysis
        return {
            'model_metrics': model_metrics,
            'model_predictions': model_predictions,
            'timeframe': timeframe,
            'total_samples': len(log_df)
        }
        
    except Exception as e:
        print(f"Error analyzing model performance: {e}")
        import traceback
        traceback.print_exc()
        return None