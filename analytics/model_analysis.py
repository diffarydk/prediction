"""
Specialized module for stacking ensemble and model contribution analysis.
This module provides advanced analytics for model performance, inter-model relationships,
and comparative analysis of stacking ensemble components.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pickle

from config import LOG_FILE, MODEL_REGISTRY_PATH, REALTIME_FILE
from analytics.analytics import ensure_image_directory

# Define standard model names and colors for consistency with analytics module
BASE_MODELS = ['baccarat_rf', 'markov_1', 'markov_2', 'xgboost_base']
ALL_MODELS = BASE_MODELS + ['stacking_ensemble', 'xgb_variant', 'markov_3']

# Color mapping for consistent model colors
MODEL_COLORS = {
    'baccarat_rf': '#4285F4',       # Google Blue
    'markov_1': '#EA4335',          # Google Red
    'markov_2': '#FBBC05',          # Google Yellow
    'xgboost_base': '#34A853',      # Google Green
    'stacking_ensemble': '#9C27B0', # Purple
    'xgb_variant': '#FF9800',       # Orange
    'markov_3': '#00BCD4',          # Cyan
}

def model_contribution_analysis(days=None, pattern_type=None):
    """
    Analyze how different models contribute to predictions.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        pattern_type: Filter by pattern type (None for all patterns)
        
    Returns:
        dict: Model contribution analysis
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check for required columns
        required_cols = ['model_contributions', 'Predicted', 'Actual']
        if not all(col in log_df.columns for col in required_cols):
            print(f"Log file missing required columns: {', '.join([col for col in required_cols if col not in log_df.columns])}")
            return None
        
        # Add Correct column if missing
        if 'Correct' not in log_df.columns:
            log_df['Correct'] = (log_df['Predicted'] == log_df['Actual']).astype(int)
        
        # Filter by time if requested
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {days} days ({len(log_df)} predictions)")
        
        # Filter by pattern type if requested
        if pattern_type and 'pattern_type' in log_df.columns:
            log_df = log_df[log_df['pattern_type'] == pattern_type]
            print(f"Analyzing data for '{pattern_type}' pattern type ({len(log_df)} predictions)")
        
        # Extract contribution data
        model_data = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidence_sum': 0, 'confidence_count': 0})
        
        for _, row in log_df.iterrows():
            # Parse the contributions
            if isinstance(row['model_contributions'], str):
                try:
                    contribs = json.loads(row['model_contributions'])
                except:
                    continue
            elif isinstance(row['model_contributions'], dict):
                contribs = row['model_contributions']
            else:
                continue
            
            correct = row['Correct']
            confidence = row.get('Confidence', 50.0)
            
            # Update stats for each model based on contribution weight
            for model, weight in contribs.items():
                model_data[model]['total'] += weight
                
                if correct:
                    model_data[model]['correct'] += weight
                    
                model_data[model]['confidence_sum'] += confidence * weight
                model_data[model]['confidence_count'] += weight
        
        # Calculate metrics for each model
        model_metrics = {}
        for model, data in model_data.items():
            if data['total'] > 0:
                accuracy = (data['correct'] / data['total']) * 100
                avg_confidence = data['confidence_sum'] / data['confidence_count'] if data['confidence_count'] > 0 else 0
                
                model_metrics[model] = {
                    'contribution': data['total'],
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'calibration_error': abs(avg_confidence - accuracy)
                }
        
        # Visualize comparative performance
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Accuracy vs Contribution
        plt.subplot(2, 1, 1)
        
        models = list(model_metrics.keys())
        x = [model_metrics[m]['contribution'] for m in models]
        y = [model_metrics[m]['accuracy'] for m in models]
        sizes = [max(50, min(500, model_metrics[m]['contribution'] * 100)) for m in models]
        
        scatter = plt.scatter(
            x, y, 
            s=sizes,
            c=[MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(models)],
            alpha=0.7
        )
        
        # Add model labels
        for i, model in enumerate(models):
            plt.annotate(
                model, 
                (x[i], y[i]),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=10
            )
        
        # Add overall accuracy reference line
        overall_acc = log_df['Correct'].mean() * 100
        plt.axhline(y=overall_acc, color='red', linestyle='--', 
                   label=f'Overall: {overall_acc:.1f}%')
        
        plt.xlabel('Contribution (Weight Sum)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Model Accuracy vs Contribution Weight', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Calibration Error
        plt.subplot(2, 1, 2)
        
        # Sort models by calibration error (lower is better)
        sorted_models = sorted(
            [(model, model_metrics[model]['calibration_error']) for model in models],
            key=lambda x: x[1]
        )
        
        calibration_models = [m[0] for m in sorted_models]
        calibration_errors = [m[1] for m in sorted_models]
        
        # Create bar chart
        bars = plt.bar(
            range(len(calibration_models)),
            calibration_errors,
            color=[MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(calibration_models)],
            alpha=0.7
        )
        
        # Add confidence and accuracy text
        for i, model in enumerate(calibration_models):
            conf = model_metrics[model]['avg_confidence']
            acc = model_metrics[model]['accuracy']
            plt.text(
                i, calibration_errors[i] + 0.5,
                f"Conf: {conf:.1f}%\nAcc: {acc:.1f}%",
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Calibration Error (|confidence-accuracy|)', fontsize=12)
        plt.title('Model Calibration Error (Lower is Better)', fontsize=14)
        plt.xticks(range(len(calibration_models)), calibration_models, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/images/model_contribution_analysis.png', dpi=300)
        plt.show()
        
        return {
            'model_metrics': model_metrics,
            'overall_accuracy': overall_acc,
            'data_points': len(log_df),
            'filtered_by': {
                'days': days,
                'pattern_type': pattern_type
            }
        }
        
    except Exception as e:
        print(f"Error comparing model contributions: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_stacking_weights(days=None):
    """
    Analyze how stacking ensemble weights have evolved over time.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        
    Returns:
        dict: Analysis of stacking weights by model
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found. Make some predictions first.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check if model_contributions exists
        if 'model_contributions' not in log_df.columns:
            print("Log file doesn't contain model contribution data")
            return None
        
        # Convert timestamp to datetime if present
        if 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            
            # Filter by days if requested
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                log_df = log_df[log_df['Timestamp'] >= cutoff_date]
                print(f"Analyzing data from the last {days} days ({len(log_df)} predictions)")
        
        # Parse model contributions from each prediction
        model_weights_over_time = []
        timestamps = []
        
        for _, row in log_df.iterrows():
            timestamp = row.get('Timestamp', None)
            
            # Parse contributions
            if isinstance(row['model_contributions'], str):
                try:
                    contribs = json.loads(row['model_contributions'])
                except Exception as e:
                    print(f"Error parsing JSON contribution: {e}")
                    continue
            elif isinstance(row['model_contributions'], dict):
                contribs = row['model_contributions']
            else:
                continue
                
            # Add to tracking
            model_weights_over_time.append(contribs)
            timestamps.append(timestamp)
        
        if not model_weights_over_time:
            print("No valid model contribution data found.")
            return None
            
        # Convert to DataFrame for time series analysis
        weights_df = pd.DataFrame(model_weights_over_time)
        
        # Ensure all models have columns in the dataframe
        for model in ALL_MODELS:
            if model not in weights_df.columns:
                weights_df[model] = 0.0
        
        # Add timestamp column if available
        if timestamps[0] is not None:
            weights_df['Timestamp'] = timestamps
            weights_df = weights_df.sort_values('Timestamp')
        
        # Calculate statistics on weights
        weight_stats = {}
        for model in weights_df.columns:
            if model == 'Timestamp':
                continue
                
            weight_stats[model] = {
                'mean': weights_df[model].mean(),
                'median': weights_df[model].median(),
                'std': weights_df[model].std(),
                'min': weights_df[model].min(),
                'max': weights_df[model].max(),
                'trend': 'increasing' if weights_df[model].iloc[-5:].mean() > weights_df[model].iloc[:5].mean() else 'decreasing'
            }
        
        # Visualize weight evolution over time
        plt.figure(figsize=(14, 8))
        
        # Determine if we have timestamps for proper time-series visualization
        if 'Timestamp' in weights_df.columns:
            x_values = weights_df['Timestamp']
            x_label = 'Date'
        else:
            x_values = range(len(weights_df))
            x_label = 'Prediction Number'
        
        # Plot each model's weight over time
        for model in weights_df.columns:
            if model == 'Timestamp':
                continue
                
            plt.plot(
                x_values, 
                weights_df[model], 
                label=model, 
                color=MODEL_COLORS.get(model, None),
                alpha=0.7,
                linewidth=2
            )
            
        # Add smoothed versions if enough data
        if len(weights_df) > 15:
            window = min(10, len(weights_df) // 3)
            
            for model in weights_df.columns:
                if model == 'Timestamp':
                    continue
                    
                # Calculate rolling average
                smooth_weights = weights_df[model].rolling(window=window).mean()
                
                plt.plot(
                    x_values,
                    smooth_weights,
                    linestyle='--',
                    color=MODEL_COLORS.get(model, None),
                    linewidth=3,
                    alpha=1.0
                )
        
        plt.title('Model Weights in Stacking Ensemble Over Time', fontsize=16)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel('Weight', fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates if applicable
        if 'Timestamp' in weights_df.columns:
            plt.gcf().autofmt_xdate()
            
        plt.tight_layout()
        plt.savefig('data/images/stacking_weights_evolution.png', dpi=300)
        plt.show()
        
        # Create pie chart of average weights
        plt.figure(figsize=(10, 10))
        
        # Calculate average weights for pie chart
        avg_weights = {model: weight_stats[model]['mean'] for model in weight_stats}
        
        # Filter models with non-zero weights
        avg_weights = {k: v for k, v in avg_weights.items() if v > 0.01}
        
        # Create pie chart
        plt.pie(
            avg_weights.values(),
            labels=avg_weights.keys(),
            autopct='%1.1f%%',
            colors=[MODEL_COLORS.get(model, f'C{i}') for i, model in enumerate(avg_weights.keys())],
            explode=[0.05 if model == 'stacking_ensemble' else 0 for model in avg_weights.keys()],
            shadow=True,
            startangle=90
        )
        
        plt.title('Average Model Contribution to Stacking Ensemble', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.tight_layout()
        plt.savefig('data/images/stacking_weights_average.png', dpi=300)
        plt.show()
        
        return {
            'weight_stats': weight_stats,
            'time_series': weights_df.to_dict() if len(weights_df) < 100 else "Too large for return",
            'data_points': len(weights_df)
        }
        
    except Exception as e:
        print(f"Error analyzing stacking weights: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_model_contributions(days=None, pattern_type=None):
    """
    Compare contributions of different models to prediction accuracy.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        pattern_type: Filter by pattern type (None for all patterns)
        
    Returns:
        dict: Model contribution analysis
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check for required columns
        required_cols = ['model_contributions', 'Predicted', 'Actual']
        if not all(col in log_df.columns for col in required_cols):
            print(f"Log file missing required columns: {', '.join([col for col in required_cols if col not in log_df.columns])}")
            return None
        
        # Add Correct column if missing
        if 'Correct' not in log_df.columns:
            log_df['Correct'] = (log_df['Predicted'] == log_df['Actual']).astype(int)
        
        # Filter by time if requested
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {days} days ({len(log_df)} predictions)")
        
        # Filter by pattern type if requested
        if pattern_type and 'pattern_type' in log_df.columns:
            log_df = log_df[log_df['pattern_type'] == pattern_type]
            print(f"Analyzing data for '{pattern_type}' pattern type ({len(log_df)} predictions)")
        
        # Extract contribution data
        model_data = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidence_sum': 0, 'confidence_count': 0})
        
        for _, row in log_df.iterrows():
            # Parse the contributions
            if isinstance(row['model_contributions'], str):
                try:
                    contribs = json.loads(row['model_contributions'])
                except:
                    continue
            elif isinstance(row['model_contributions'], dict):
                contribs = row['model_contributions']
            else:
                continue
            
            correct = row['Correct']
            confidence = row.get('Confidence', 50.0)
            
            # Update stats for each model based on contribution weight
            for model, weight in contribs.items():
                model_data[model]['total'] += weight
                
                if correct:
                    model_data[model]['correct'] += weight
                    
                model_data[model]['confidence_sum'] += confidence * weight
                model_data[model]['confidence_count'] += weight
        
        # Calculate metrics for each model
        model_metrics = {}
        for model, data in model_data.items():
            if data['total'] > 0:
                accuracy = (data['correct'] / data['total']) * 100
                avg_confidence = data['confidence_sum'] / data['confidence_count'] if data['confidence_count'] > 0 else 0
                
                model_metrics[model] = {
                    'contribution': data['total'],
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'calibration_error': abs(avg_confidence - accuracy)
                }
        
        # Visualize comparative performance
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Accuracy vs Contribution
        plt.subplot(2, 1, 1)
        
        models = list(model_metrics.keys())
        x = [model_metrics[m]['contribution'] for m in models]
        y = [model_metrics[m]['accuracy'] for m in models]
        sizes = [max(50, min(500, model_metrics[m]['contribution'] * 100)) for m in models]
        
        scatter = plt.scatter(
            x, y, 
            s=sizes,
            c=[MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(models)],
            alpha=0.7
        )
        
        # Add model labels
        for i, model in enumerate(models):
            plt.annotate(
                model, 
                (x[i], y[i]),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=10
            )
        
        # Add overall accuracy reference line
        overall_acc = log_df['Correct'].mean() * 100
        plt.axhline(y=overall_acc, color='red', linestyle='--', 
                   label=f'Overall: {overall_acc:.1f}%')
        
        plt.xlabel('Contribution (Weight Sum)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Model Accuracy vs Contribution Weight', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Calibration Error
        plt.subplot(2, 1, 2)
        
        # Sort models by calibration error (lower is better)
        sorted_models = sorted(
            [(model, model_metrics[model]['calibration_error']) for model in models],
            key=lambda x: x[1]
        )
        
        calibration_models = [m[0] for m in sorted_models]
        calibration_errors = [m[1] for m in sorted_models]
        
        # Create bar chart
        bars = plt.bar(
            range(len(calibration_models)),
            calibration_errors,
            color=[MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(calibration_models)],
            alpha=0.7
        )
        
        # Add confidence and accuracy text
        for i, model in enumerate(calibration_models):
            conf = model_metrics[model]['avg_confidence']
            acc = model_metrics[model]['accuracy']
            plt.text(
                i, calibration_errors[i] + 0.5,
                f"Conf: {conf:.1f}%\nAcc: {acc:.1f}%",
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Calibration Error (|confidence-accuracy|)', fontsize=12)
        plt.title('Model Calibration Error (Lower is Better)', fontsize=14)
        plt.xticks(range(len(calibration_models)), calibration_models, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/images/model_contribution_analysis.png', dpi=300)
        plt.show()
        
        return {
            'model_metrics': model_metrics,
            'overall_accuracy': overall_acc,
            'data_points': len(log_df),
            'filtered_by': {
                'days': days,
                'pattern_type': pattern_type
            }
        }
        
    except Exception as e:
        print(f"Error comparing model contributions: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_model_confidence(days=None):
    """
    Analyze relationship between model confidence and accuracy.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        
    Returns:
        dict: Confidence analysis results
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check if required columns exist
        required_cols = ['Confidence', 'Correct', 'model_contributions']
        if not all(col in log_df.columns for col in required_cols):
            print(f"Log file missing required columns: {', '.join([col for col in required_cols if col not in log_df.columns])}")
            return None
        
        # Filter by time if requested
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {days} days ({len(log_df)} predictions)")
        
        # Create confidence bins
        log_df['ConfidenceBin'] = pd.cut(
            log_df['Confidence'],
            bins=[0, 40, 50, 60, 70, 80, 90, 100],
            labels=['0-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        )
        
        # Calculate baseline accuracy by confidence bin
        overall_results = log_df.groupby('ConfidenceBin')['Correct'].agg(['mean', 'count']).reset_index()
        overall_results['accuracy'] = overall_results['mean'] * 100
        
        # Extract model-specific confidence-accuracy data
        model_conf_data = {}
        
        for _, row in log_df.iterrows():
            confidence_bin = row['ConfidenceBin']
            if pd.isna(confidence_bin):
                continue
                
            correct = row['Correct']
            confidence = row['Confidence']
            
            # Parse model contributions
            if isinstance(row['model_contributions'], str):
                try:
                    contribs = json.loads(row['model_contributions'])
                except:
                    continue
            elif isinstance(row['model_contributions'], dict):
                contribs = row['model_contributions']
            else:
                continue
            
            # Update contribution-weighted confidence data for each model
            for model, weight in contribs.items():
                if model not in model_conf_data:
                    model_conf_data[model] = defaultdict(lambda: {'correct': 0, 'total': 0})
                
                model_conf_data[model][confidence_bin]['total'] += weight
                if correct:
                    model_conf_data[model][confidence_bin]['correct'] += weight
        
        # Calculate accuracy by confidence bin for each model
        model_results = {}
        
        for model, bins in model_conf_data.items():
            model_results[model] = []
            
            for bin_name in overall_results['ConfidenceBin']:
                bin_data = bins.get(bin_name, {'correct': 0, 'total': 0})
                
                if bin_data['total'] > 0:
                    accuracy = (bin_data['correct'] / bin_data['total']) * 100
                    model_results[model].append({
                        'confidence_bin': bin_name,
                        'accuracy': accuracy,
                        'total': bin_data['total']
                    })
        
        # Visualize confidence calibration
        plt.figure(figsize=(14, 8))
        
        # Plot confidence calibration curves for each model with sufficient data
        for model, results in model_results.items():
            # Skip models with insufficient data
            if len(results) < 3:
                continue
                
            # Extract data points
            bins = [r['confidence_bin'] for r in results]
            accuracies = [r['accuracy'] for r in results]
            totals = [r['total'] for r in results]
            
            # Only include models with sufficient data points
            if sum(totals) < 10:
                continue
                
            # Get middle of each bin for plotting
            bin_centers = [30, 45, 55, 65, 75, 85, 95]
            bin_centers = bin_centers[:len(bins)]
            
            # Plot model calibration curve
            plt.plot(
                bin_centers, 
                accuracies, 
                'o-', 
                label=f"{model}", 
                color=MODEL_COLORS.get(model, None),
                alpha=0.7,
                linewidth=2
            )
        
        # Plot perfect calibration reference line
        plt.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Perfect Calibration')
        
        # Add overall calibration for comparison
        bin_centers = [30, 45, 55, 65, 75, 85, 95]
        bin_centers = bin_centers[:len(overall_results)]
        overall_accuracies = overall_results['accuracy'].tolist()
        
        plt.plot(
            bin_centers,
            overall_accuracies,
            'ko-',
            linewidth=2,
            label='Overall'
        )
        
        plt.xlabel('Confidence (%)', fontsize=14)
        plt.ylabel('Actual Accuracy (%)', fontsize=14)
        plt.title('Model-Specific Confidence Calibration', fontsize=16)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Add explanatory text
        plt.figtext(
            0.5, 0.01,
            "Points above the line indicate underconfidence (model is more accurate than confident)\n"
            "Points below the line indicate overconfidence (model is less accurate than confident)",
            ha="center",
            fontsize=11,
            bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('data/images/model_confidence_calibration.png', dpi=300)
        plt.show()
        
        # Calculate calibration metrics for each model
        calibration_metrics = {}
        
        for model, results in model_results.items():
            if len(results) < 3:
                continue
                
            # Extract data points
            bins = [r['confidence_bin'] for r in results]
            accuracies = [r['accuracy'] for r in results]
            totals = [r['total'] for r in results]
            
            if sum(totals) < 10:
                continue
                
            # Get middle of each bin
            bin_centers = [30, 45, 55, 65, 75, 85, 95]
            bin_centers = bin_centers[:len(bins)]
            
            # Calculate calibration error - mean absolute difference between confidence and accuracy
            calibration_error = np.mean([abs(conf - acc) for conf, acc in zip(bin_centers, accuracies)])
            
            # Calculate direction bias - positive means underconfident, negative means overconfident
            direction_bias = np.mean([acc - conf for conf, acc in zip(bin_centers, accuracies)])
            
            calibration_metrics[model] = {
                'calibration_error': calibration_error,
                'direction_bias': direction_bias,
                'bias_type': 'underconfident' if direction_bias > 5 else 'overconfident' if direction_bias < -5 else 'well-calibrated',
                'data_points': sum(totals)
            }
        
        return {
            'overall_calibration': overall_results.to_dict(),
            'model_calibration': model_results,
            'calibration_metrics': calibration_metrics,
            'data_points': len(log_df)
        }
        
    except Exception as e:
        print(f"Error analyzing model confidence: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_model_drift(lookback_days=90, window_size=10):
    """
    Detect and analyze model drift over time.
    
    Args:
        lookback_days: Maximum days to look back
        window_size: Size of windows for change detection
        
    Returns:
        dict: Model drift analysis results
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check for required columns
        if 'Timestamp' not in log_df.columns or 'Correct' not in log_df.columns:
            print("Log file missing required timestamp or correctness data")
            return None
        
        # Convert timestamp to datetime
        log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
        
        # Filter to lookback period
        if lookback_days:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {lookback_days} days ({len(log_df)} predictions)")
        
        # Sort by timestamp
        log_df = log_df.sort_values('Timestamp')
        
        # Create time windows
        if window_size:
            # Use fixed-size windows
            window_count = max(2, len(log_df) // window_size)
            windows = [log_df.iloc[i*window_size:(i+1)*window_size] for i in range(window_count)]
        else:
            # Use fixed number of windows
            window_count = 5
            window_size = len(log_df) // window_count
            windows = [log_df.iloc[i*window_size:(i+1)*window_size] for i in range(window_count)]
        
        # Calculate accuracy for each window
        window_metrics = []
        
        for i, window_df in enumerate(windows):
            if len(window_df) == 0:
                continue
                
            start_time = window_df['Timestamp'].min()
            end_time = window_df['Timestamp'].max()
            
            window_accuracy = window_df['Correct'].mean() * 100
            
            # Extract model contributions if available
            model_accuracies = {}
            
            if 'model_contributions' in window_df.columns:
                model_data = defaultdict(lambda: {'correct': 0, 'total': 0})
                
                for _, row in window_df.iterrows():
                    correct = row['Correct']
                    
                    # Parse model contributions
                    if isinstance(row['model_contributions'], str):
                        try:
                            contribs = json.loads(row['model_contributions'])
                        except:
                            continue
                    elif isinstance(row['model_contributions'], dict):
                        contribs = row['model_contributions']
                    else:
                        continue
                    
                    # Update stats for each model
                    for model, weight in contribs.items():
                        model_data[model]['total'] += weight
                        if correct:
                            model_data[model]['correct'] += weight
                
                # Calculate accuracy for each model
                for model, data in model_data.items():
                    if data['total'] > 0:
                        model_accuracies[model] = (data['correct'] / data['total']) * 100
            
            window_metrics.append({
                'window': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'sample_count': len(window_df),
                'accuracy': window_accuracy,
                'model_accuracies': model_accuracies
            })
        
        # Analyze model drift
        drift_metrics = {}
        
        # Only analyze drift if we have at least 2 windows
        if len(window_metrics) >= 2:
            # Calculate overall drift
            first_window = window_metrics[0]
            last_window = window_metrics[-1]
            
            overall_drift = last_window['accuracy'] - first_window['accuracy']
            
            # Calculate drift for each model
            model_drift = {}
            for model in ALL_MODELS:
                if (model in first_window.get('model_accuracies', {}) and 
                    model in last_window.get('model_accuracies', {})):
                    
                    model_drift[model] = last_window['model_accuracies'][model] - first_window['model_accuracies'][model]
            
            drift_metrics = {
                'overall_drift': overall_drift,
                'drift_direction': 'improving' if overall_drift > 5 else 'declining' if overall_drift < -5 else 'stable',
                'model_drift': model_drift
            }
        
        # Visualize model drift
        plt.figure(figsize=(14, 8))
        
        # Plot overall accuracy trend
        window_numbers = [w['window'] for w in window_metrics]
        accuracies = [w['accuracy'] for w in window_metrics]
        
        plt.plot(window_numbers, accuracies, 'ko-', linewidth=2, label='Overall')
        
        # Plot model-specific trends
        for model in ALL_MODELS:
            model_accs = []
            
            for w in window_metrics:
                if model in w.get('model_accuracies', {}):
                    model_accs.append(w['model_accuracies'][model])
                else:
                    model_accs.append(None)  # Missing data point
            
            # Only plot if we have at least 2 data points
            if sum(1 for acc in model_accs if acc is not None) >= 2:
                # Create x-values only for non-None data points
                valid_indices = [i for i, acc in enumerate(model_accs) if acc is not None]
                x_values = [window_numbers[i] for i in valid_indices]
                y_values = [model_accs[i] for i in valid_indices]
                
                plt.plot(
                    x_values, 
                    y_values, 
                    'o-', 
                    label=model, 
                    color=MODEL_COLORS.get(model, None),
                    alpha=0.7,
                    linewidth=2
                )
        
        plt.xlabel('Time Window', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Model Performance Drift Over Time', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        
        plt.savefig('data/images/model_drift_analysis.png', dpi=300)
        plt.show()
        
        return {
            'window_metrics': window_metrics,
            'drift_metrics': drift_metrics,
            'window_count': len(window_metrics),
            'timespan_days': (window_metrics[-1]['end_time'] - window_metrics[0]['start_time']).days if window_metrics else 0
        }
        
    except Exception as e:
        print(f"Error detecting model drift: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_ensemble_performance(days=None):
    """
    Evaluate the performance of the stacking ensemble compared to base models.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        
    Returns:
        dict: Performance comparison metrics
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check if model_contributions exists
        if 'model_contributions' not in log_df.columns:
            print("Log file doesn't contain model contribution data")
            return None
        
        # Apply time filtering
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {days} days ({len(log_df)} predictions)")
        
        # Extract model performance data
        model_data = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for _, row in log_df.iterrows():
            # Parse model contributions
            if isinstance(row['model_contributions'], str):
                try:
                    contribs = json.loads(row['model_contributions'])
                except:
                    continue
            elif isinstance(row['model_contributions'], dict):
                contribs = row['model_contributions']
            else:
                continue
                
            # Get prediction outcome
            correct = row['Correct'] if 'Correct' in row else (row['Predicted'] == row['Actual'])
            
            # Update metrics for each model
            for model, weight in contribs.items():
                model_data[model]['total'] += weight
                if correct:
                    model_data[model]['correct'] += weight
        
        # Calculate performance metrics
        model_metrics = {}
        for model, data in model_data.items():
            if data['total'] > 0:
                accuracy = (data['correct'] / data['total']) * 100
                model_metrics[model] = {
                    'accuracy': accuracy,
                    'contributions': data['total']
                }
        
        # Identify baseline models vs stacking for comparison
        stacking_metrics = model_metrics.get('stacking_ensemble', {'accuracy': 0, 'contributions': 0})
        base_models = {m: metrics for m, metrics in model_metrics.items() 
                     if m != 'stacking_ensemble'}
        
        # Calculate average base model performance
        if base_models:
            avg_base_accuracy = sum(m['accuracy'] for m in base_models.values()) / len(base_models)
            
            # Find best base model
            best_base_model = max(base_models.items(), key=lambda x: x[1]['accuracy'])
            
            # Calculate improvement metrics
            improvement_over_avg = stacking_metrics['accuracy'] - avg_base_accuracy
            improvement_over_best = stacking_metrics['accuracy'] - best_base_model[1]['accuracy']
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            
            # Plot 1: Accuracy comparison
            models = list(model_metrics.keys())
            accuracies = [model_metrics[m]['accuracy'] for m in models]
            contributions = [model_metrics[m]['contributions'] for m in models]
            
            # Sort by accuracy
            sorted_indices = np.argsort(accuracies)[::-1]
            sorted_models = [models[i] for i in sorted_indices]
            sorted_accs = [accuracies[i] for i in sorted_indices]
            sorted_contribs = [contributions[i] for i in sorted_indices]
            
            # Create enhanced bar chart
            plt.subplot(2, 1, 1)
            bars = plt.bar(
                range(len(sorted_models)),
                sorted_accs,
                color=[MODEL_COLORS.get(m, '#777777') for m in sorted_models],
                alpha=0.7
            )
            
            # Add contribution markers
            for i, bar in enumerate(bars):
                # Scale marker size by contribution proportion
                size = min(40, max(10, sorted_contribs[i] / sum(sorted_contribs) * 100))
                plt.plot(i, bar.get_height() + 2, 'ko', markersize=size, alpha=0.6)
                
                # Add exact value
                plt.text(
                    i,
                    bar.get_height() + 8,
                    f"{sorted_accs[i]:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
            
            plt.xticks(range(len(sorted_models)), sorted_models, rotation=45, ha='right')
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title('Model Accuracy Comparison (marker size = contribution weight)', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            
            # Plot 2: Stacking improvement
            plt.subplot(2, 1, 2)
            plt.bar(['vs. Average', 'vs. Best Base'], 
                   [improvement_over_avg, improvement_over_best],
                   color=['forestgreen' if improvement_over_avg > 0 else 'firebrick',
                         'forestgreen' if improvement_over_best > 0 else 'firebrick'])
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.ylabel('Accuracy Improvement (%)', fontsize=12)
            plt.title('Stacking Ensemble Improvement', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            
            # Annotate with values
            for i, val in enumerate([improvement_over_avg, improvement_over_best]):
                plt.text(i, val + np.sign(val) * 0.5, f"{val:+.2f}%", 
                       ha='center', va='bottom' if val > 0 else 'top')
            
            plt.tight_layout()
            plt.savefig('data/images/stacking_performance.png', dpi=300)
            plt.show()
            
            return {
                'model_metrics': model_metrics,
                'stacking_metrics': stacking_metrics,
                'avg_base_accuracy': avg_base_accuracy,
                'best_base_model': {'model': best_base_model[0], 'accuracy': best_base_model[1]['accuracy']},
                'improvement_over_avg': improvement_over_avg,
                'improvement_over_best': improvement_over_best
            }
        else:
            print("No base model data available for comparison")
            return {'model_metrics': model_metrics}
            
    except Exception as e:
        print(f"Error evaluating ensemble performance: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_ensemble_weights():
    """
    Analyze the internal weights of the stacking ensemble.
    
    Returns:
        dict: Analysis of stacking weights and importance
    """
    ensure_image_directory()
    
    try:
        # Attempt to load the stacking ensemble from registry
        registry_path = MODEL_REGISTRY_PATH
        stacking_file = os.path.join(registry_path, "stacking_ensemble.pkl")
        
        if not os.path.exists(stacking_file):
            print("Stacking ensemble model file not found.")
            return None
            
        # Load model
        with open(stacking_file, 'rb') as f:
            import pickle
            stacking_model = pickle.load(f)
            
        # Check for coefficients in meta-model
        if not hasattr(stacking_model, 'meta_model') or not hasattr(stacking_model.meta_model, 'coef_'):
            print("Stacking model doesn't have accessible coefficients.")
            return None
            
        # Extract coefficients
        coef = stacking_model.meta_model.coef_
        
        # Extract feature importance
        if hasattr(stacking_model, 'meta_X') and hasattr(stacking_model, 'meta_y'):
            # Analyze feature importance
            importance = np.abs(coef).mean(axis=0)
            
            # Try to map features to model names
            feature_names = []
            registry_file = os.path.join(registry_path, "registry.json")
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                    
                base_models = [m for m in registry_data.get("model_ids", []) if m != "stacking_ensemble"]
                
                # Construct feature names based on model structure
                for model in base_models:
                    for outcome in ['Banker', 'Player', 'Tie']:
                        feature_names.append(f"{model}_{outcome}")
                
                # Add pattern feature names
                for pattern in ['no_pattern', 'streak', 'alternating', 'tie']:
                    feature_names.append(f"pattern_{pattern}")
            else:
                # Generic feature names
                feature_names = [f"feature_{i+1}" for i in range(len(importance))]
            
            # Adjust list length if needed
            if len(feature_names) != len(importance):
                if len(feature_names) > len(importance):
                    feature_names = feature_names[:len(importance)]
                else:
                    for i in range(len(feature_names), len(importance)):
                        feature_names.append(f"unknown_{i+1}")
            
            # Create importance dictionary
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
            
            # Group features by model
            model_importance = defaultdict(float)
            pattern_importance = 0
            
            for name, imp in importance_dict.items():
                if 'pattern_' in name:
                    pattern_importance += imp
                else:
                    model_name = name.split('_')[0]
                    model_importance[model_name] += imp
            
            # Sort by importance
            model_importance = {k: v for k, v in sorted(model_importance.items(), 
                                                     key=lambda item: item[1], 
                                                     reverse=True)}
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            
            # Plot 1: Feature importance
            plt.subplot(2, 1, 1)
            
            # Sort features by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:15]  # Show top 15
            
            feature_names = [f[0] for f in top_features]
            feature_values = [f[1] for f in top_features]
            
            # Create bar chart
            colors = []
            for name in feature_names:
                if 'pattern_' in name:
                    colors.append('#8B0000')  # Dark red for pattern features
                else:
                    model_name = name.split('_')[0]
                    colors.append(MODEL_COLORS.get(model_name, '#777777'))
            
            plt.barh(range(len(feature_names)), feature_values, color=colors, alpha=0.7)
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('Importance (Mean Absolute Coefficient)', fontsize=12)
            plt.title('Top Feature Importance in Stacking Ensemble', fontsize=14)
            plt.grid(axis='x', alpha=0.3)
            
            # Plot 2: Model importance
            plt.subplot(2, 1, 2)
            
            # Combine model and pattern importance for pie chart
            all_importance = dict(model_importance)
            all_importance['pattern_features'] = pattern_importance
            
            # Calculate percentages
            total = sum(all_importance.values())
            percentages = {k: (v/total)*100 for k, v in all_importance.items()}
            
            # Create pie chart
            labels = list(percentages.keys())
            sizes = list(percentages.values())
            colors = [MODEL_COLORS.get(m, '#777777') if m != 'pattern_features' else '#8B0000' for m in labels]
            
            patches, texts, autotexts = plt.pie(
                sizes, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%', 
                startangle=90,
                explode=[0.05 if s == max(sizes) else 0 for s in sizes]
            )
            
            # Enhance text visibility
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')
                
            plt.axis('equal')
            plt.title('Contribution to Stacking Ensemble Decision', fontsize=14)
            
            plt.tight_layout()
            plt.savefig('data/images/stacking_weights_analysis.png', dpi=300)
            plt.show()
            
            return {
                'feature_importance': dict(sorted_features),
                'model_importance': dict(model_importance),
                'pattern_importance': pattern_importance,
                'percentages': percentages,
                'meta_examples': len(stacking_model.meta_X) if hasattr(stacking_model, 'meta_X') else 0
            }
        else:
            print("Stacking model doesn't have meta-training data.")
            return None
            
    except Exception as e:
        print(f"Error analyzing ensemble weights: {e}")
        import traceback
        traceback.print_exc()
        return None

def track_model_drift(lookback_days=None, window_size=10):
    """
    Track how models have drifted over time in terms of performance and contributions.
    
    Args:
        lookback_days: Number of days to look back (None for all data)
        window_size: Size of window for rolling performance calculation
        
    Returns:
        dict: Model drift analysis results
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check for required columns
        if 'model_contributions' not in log_df.columns or 'Correct' not in log_df.columns:
            print("Log file doesn't contain required columns")
            return None
        
        # Apply time filtering if timestamp is available
        if lookback_days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {lookback_days} days ({len(log_df)} predictions)")
        
        # Sort by timestamp if available
        if 'Timestamp' in log_df.columns:
            log_df = log_df.sort_values('Timestamp')
        
        # Extract model-specific predictions for each data point
        predictions_by_model = defaultdict(list)
        
        for i, row in log_df.iterrows():
            # Parse model contributions
            if isinstance(row['model_contributions'], str):
                try:
                    contribs = json.loads(row['model_contributions'])
                except:
                    continue
            elif isinstance(row['model_contributions'], dict):
                contribs = row['model_contributions']
            else:
                continue
                
            # Get prediction outcome
            correct = row['Correct'] if 'Correct' in row else (row['Predicted'] == row['Actual'])
            timestamp = row.get('Timestamp', i)
            
            # Record for each model
            for model, weight in contribs.items():
                if weight > 0:  # Only include models that contributed
                    predictions_by_model[model].append({
                        'timestamp': timestamp,
                        'correct': 1 if correct else 0,
                        'weight': weight,
                        'index': i
                    })
        
        # Calculate rolling performance for each model
        rolling_performance = {}
        
        for model, predictions in predictions_by_model.items():
            if len(predictions) < window_size:
                continue  # Skip models with too few predictions
                
            # Create rolling window performance
            rolling_acc = []
            timestamps = []
            indices = []
            
            for i in range(len(predictions) - window_size + 1):
                window = predictions[i:i+window_size]
                correct = sum(p['correct'] for p in window)
                accuracy = (correct / window_size) * 100
                
                rolling_acc.append(accuracy)
                timestamps.append(window[-1]['timestamp'])  # Use last timestamp in window
                indices.append(window[-1]['index'])  # Use last index in window
            
            rolling_performance[model] = {
                'accuracy': rolling_acc,
                'timestamps': timestamps,
                'indices': indices
            }
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Plot rolling accuracy for each model
        if all(isinstance(t, pd.Timestamp) for data in rolling_performance.values() for t in data['timestamps']):
            # Time-based x-axis
            for model, data in rolling_performance.items():
                plt.plot(
                    data['timestamps'],
                    data['accuracy'],
                    label=model,
                    color=MODEL_COLORS.get(model, None),
                    alpha=0.7,
                    linewidth=2
                )
            plt.gcf().autofmt_xdate()
            plt.xlabel('Date', fontsize=12)
        else:
            # Index-based x-axis
            for model, data in rolling_performance.items():
                plt.plot(
                    data['indices'],
                    data['accuracy'],
                    label=model,
                    color=MODEL_COLORS.get(model, None),
                    alpha=0.7,
                    linewidth=2
                )
            plt.xlabel('Prediction Index', fontsize=12)
        
        plt.ylabel('Rolling Accuracy (%)', fontsize=12)
        plt.title(f'{window_size}-Prediction Rolling Accuracy by Model', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('data/images/model_drift_analysis.png', dpi=300)
        plt.show()
        
        # Calculate trend metrics
        drift_metrics = {}
        
        for model, data in rolling_performance.items():
            if len(data['accuracy']) >= 2:
                start = data['accuracy'][0]
                end = data['accuracy'][-1]
                change = end - start
                
                drift_metrics[model] = {
                    'start_accuracy': start,
                    'end_accuracy': end,
                    'change': change,
                    'trend': 'improving' if change > 5 else 'declining' if change < -5 else 'stable'
                }
        
        return {
            'rolling_performance': rolling_performance,
            'drift_metrics': drift_metrics,
            'window_size': window_size
        }
        
    except Exception as e:
        print(f"Error tracking model drift: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_base_models(test_set=None):
    """
    Perform a direct comparison of base model performance on either historical data
    or a provided test set.
    
    Args:
        test_set: Optional test data for evaluation
        
    Returns:
        dict: Comparative performance metrics
    """
    ensure_image_directory()
    
    try:
        # Load models from registry
        registry_path = MODEL_REGISTRY_PATH
        registry_file = os.path.join(registry_path, "registry.json")
        
        if not os.path.exists(registry_file):
            print("Model registry not found.")
            return None
            
        # Load registry data
        with open(registry_file, 'r') as f:
            registry_data = json.load(f)
            
        # Get model IDs
        model_ids = registry_data.get("model_ids", [])
        
        # Filter to base models (exclude stacking ensemble)
        base_model_ids = [m for m in model_ids if m != "stacking_ensemble"]
        
        if not base_model_ids:
            print("No base models found in registry.")
            return None
            
        # Prepare for evaluation
        model_performance = {}
        
        # If test set is provided, use it for evaluation
        if test_set is not None:
            X_test = test_set['X']
            y_test = test_set['y']
            
            # Evaluate each model on test set
            for model_id in base_model_ids:
                model_file = os.path.join(registry_path, f"{model_id}.pkl")
                
                if os.path.exists(model_file):
                    # Load model
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                        
                    # Evaluate model
                    try:
                        y_pred = model.predict(X_test)
                        accuracy = np.mean(y_pred == y_test) * 100
                        
                        # Get class-specific accuracy
                        class_acc = {}
                        for cls in [0, 1, 2]:  # Banker, Player, Tie
                            mask = (y_test == cls)
                            if mask.any():
                                class_acc[cls] = np.mean(y_pred[mask] == y_test[mask]) * 100
                            else:
                                class_acc[cls] = 0
                                
                        model_performance[model_id] = {
                            'accuracy': accuracy,
                            'class_accuracy': class_acc,
                            'samples': len(y_test)
                        }
                    except Exception as e:
                        print(f"Error evaluating {model_id}: {e}")
            
        else:
            # Use historical performance from logs
            if not os.path.exists(LOG_FILE):
                print("No prediction log found for historical evaluation.")
                return None
                
            log_df = pd.read_csv(LOG_FILE)
            
            # Check for model_contributions column
            if 'model_contributions' not in log_df.columns:
                print("Log file doesn't contain model contribution data")
                return None
                
            # Extract performance from logs
            for _, row in log_df.iterrows():
                # Parse model contributions
                if isinstance(row['model_contributions'], str):
                    try:
                        contribs = json.loads(row['model_contributions'])
                    except:
                        continue
                elif isinstance(row['model_contributions'], dict):
                    contribs = row['model_contributions']
                else:
                    continue
                    
                # Get prediction correctness
                correct = row['Correct'] if 'Correct' in row else (row['Predicted'] == row['Actual'])
                
                # Get actual outcome for class-specific tracking
                actual = row['Actual']
                
                # Update each model's stats
                for model_id, weight in contribs.items():
                    if model_id in base_model_ids:
                        if model_id not in model_performance:
                            model_performance[model_id] = {
                                'correct': 0,
                                'total': 0,
                                'class_correct': {0: 0, 1: 0, 2: 0},
                                'class_total': {0: 0, 1: 0, 2: 0}
                            }
                            
                        stats = model_performance[model_id]
                        stats['total'] += weight
                        
                        # Update class-specific counts
                        stats['class_total'][actual] = stats['class_total'].get(actual, 0) + weight
                        
                        if correct:
                            stats['correct'] += weight
                            stats['class_correct'][actual] = stats['class_correct'].get(actual, 0) + weight
            
            # Calculate final metrics
            for model_id, stats in model_performance.items():
                if stats['total'] > 0:
                    # Overall accuracy
                    accuracy = (stats['correct'] / stats['total']) * 100
                    
                    # Class-specific accuracy
                    class_acc = {}
                    for cls in [0, 1, 2]:
                        if stats['class_total'].get(cls, 0) > 0:
                            class_acc[cls] = (stats['class_correct'].get(cls, 0) / stats['class_total'][cls]) * 100
                        else:
                            class_acc[cls] = 0
                            
                    model_performance[model_id] = {
                        'accuracy': accuracy,
                        'class_accuracy': class_acc,
                        'samples': stats['total']
                    }
        
        # Create comparative visualization
        if model_performance:
            plt.figure(figsize=(14, 10))
            
            # Plot 1: Overall accuracy
            plt.subplot(2, 1, 1)
            
            # Sort models by accuracy
            sorted_models = sorted(model_performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            model_names = [m[0] for m in sorted_models]
            accuracies = [m[1]['accuracy'] for m in sorted_models]
            
            bars = plt.bar(
                range(len(model_names)),
                accuracies,
                color=[MODEL_COLORS.get(m, '#777777') for m in model_names],
                alpha=0.7
            )
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{accuracies[i]:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            
            plt.axhline(y=33.33, color='gray', linestyle='--', alpha=0.7, label='Random Guess (33.33%)')
            
            plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title('Base Model Accuracy Comparison', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            
            # Plot 2: Class-specific accuracy
            plt.subplot(2, 1, 2)
            
            # Prepare data for grouped bar chart
            class_labels = ['Banker', 'Player', 'Tie']
            x = np.arange(len(model_names))
            width = 0.25
            
            # Plot bars for each class
            for i, cls in enumerate([0, 1, 2]):
                class_accs = [m[1]['class_accuracy'].get(cls, 0) for m in sorted_models]
                
                plt.bar(
                    x + (i - 1) * width,
                    class_accs,
                    width,
                    label=class_labels[cls],
                    color=['green', 'blue', 'purple'][cls],
                    alpha=0.7
                )
            
            plt.xticks(x, model_names, rotation=45, ha='right')
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Class Accuracy (%)', fontsize=12)
            plt.title('Model Accuracy by Outcome Class', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('data/images/base_model_comparison.png', dpi=300)
            plt.show()
            
            return {
                'model_performance': model_performance,
                'best_model': sorted_models[0][0],
                'best_accuracy': sorted_models[0][1]['accuracy']
            }
        else:
            print("No performance data available for models")
            return None
            
    except Exception as e:
        print(f"Error comparing base models: {e}")
        import traceback
        traceback.print_exc()
        return None

def profile_memory_usage():
    """
    Profile memory usage of different models in the registry.
    
    Returns:
        dict: Memory usage statistics
    """
    ensure_image_directory()
    
    try:
        # Load models from registry
        registry_path = MODEL_REGISTRY_PATH
        registry_file = os.path.join(registry_path, "registry.json")
        
        if not os.path.exists(registry_file):
            print("Model registry not found.")
            return None
            
        # Load registry data
        with open(registry_file, 'r') as f:
            registry_data = json.load(f)
            
        # Get model IDs
        model_ids = registry_data.get("model_ids", [])
        
        if not model_ids:
            print("No models found in registry.")
            return None
            
        # Measure memory usage for each model
        memory_usage = {}
        
        for model_id in model_ids:
            model_file = os.path.join(registry_path, f"{model_id}.pkl")
            
            if os.path.exists(model_file):
                # Get file size
                file_size = os.path.getsize(model_file) / (1024 * 1024)  # Convert to MB
                
                # Load model for in-memory size measurement
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    
                # Estimate in-memory size using pickle
                import io
                buffer = io.BytesIO()
                pickle.dump(model, buffer)
                memory_size = buffer.getbuffer().nbytes / (1024 * 1024)  # Convert to MB
                
                memory_usage[model_id] = {
                    'file_size_mb': file_size,
                    'memory_size_mb': memory_size
                }
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Sort models by memory usage
        sorted_models = sorted(memory_usage.items(), key=lambda x: x[1]['memory_size_mb'], reverse=True)
        model_names = [m[0] for m in sorted_models]
        file_sizes = [m[1]['file_size_mb'] for m in sorted_models]
        memory_sizes = [m[1]['memory_size_mb'] for m in sorted_models]
        
        # Create grouped bar chart
        x = np.arange(len(model_names))
        width = 0.35
        
        # Plot bars
        plt.bar(x - width/2, file_sizes, width, label='File Size (MB)', color='cornflowerblue')
        plt.bar(x + width/2, memory_sizes, width, label='Memory Size (MB)', color='darkorange')
        
        # Add labels
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Size (MB)', fontsize=12)
        plt.title('Model Memory Usage', fontsize=14)
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        # Add annotation with total memory usage
        total_memory = sum(memory_sizes)
        plt.figtext(0.5, 0.01, 
                   f"Total Memory Usage of All Models: {total_memory:.2f} MB", 
                   ha="center", fontsize=12, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig('data/images/model_memory_usage.png', dpi=300)
        plt.show()
        
        return {
            'memory_usage': memory_usage,
            'total_file_size_mb': sum(file_sizes),
            'total_memory_size_mb': total_memory
        }
        
    except Exception as e:
        print(f"Error profiling memory usage: {e}")
        import traceback
        traceback.print_exc()
        return None