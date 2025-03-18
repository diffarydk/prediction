"""
Implementation of pattern-specific feature analysis for the Baccarat prediction system.
This module provides functions to analyze the effectiveness of different pattern types
and how they contribute to the stacking ensemble decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import matplotlib.dates as mdates

from config import LOG_FILE, REALTIME_FILE, MODEL_REGISTRY_PATH
from analytics.analytics import ensure_image_directory

# Define standard pattern colors for consistency
PATTERN_COLORS = {
    'no_pattern': '#7F7F7F',    # Gray
    'streak': '#E31A1C',        # Red
    'alternating': '#1F78B4',   # Blue
    'tie': '#33A02C',           # Green
    'other_pattern': '#9467BD'  # Purple
}

def pattern_feature_impact(days=None, top_models=3):
    """
    Analyze the impact of pattern features on prediction accuracy.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        top_models: Number of top models to include in the analysis
        
    Returns:
        dict: Pattern impact analysis results
    """
    ensure_image_directory()
    
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check for required columns
        if 'pattern_type' not in log_df.columns or 'Correct' not in log_df.columns:
            print("Log file missing pattern or correctness data")
            return None
        
        # Apply time filtering if requested
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {days} days ({len(log_df)} predictions)")
        
        # Get overall accuracy for reference
        overall_accuracy = log_df['Correct'].mean() * 100
        
        # Calculate pattern-specific accuracy
        pattern_accuracy = {}
        pattern_counts = {}
        
        # Process each pattern type
        for pattern in log_df['pattern_type'].unique():
            if pd.isna(pattern):
                continue
                
            pattern_df = log_df[log_df['pattern_type'] == pattern]
            
            # Skip if too few examples
            if len(pattern_df) < 5:
                continue
                
            # Calculate accuracy
            accuracy = pattern_df['Correct'].mean() * 100
            
            pattern_accuracy[pattern] = accuracy
            pattern_counts[pattern] = len(pattern_df)
        
        # Analyze model-specific performance on different patterns
        if 'model_contributions' in log_df.columns:
            # Extract model data by pattern
            model_pattern_data = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
            
            for _, row in log_df.iterrows():
                pattern = row.get('pattern_type')
                if pd.isna(pattern):
                    continue
                    
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
                    model_pattern_data[model][pattern]['total'] += weight
                    if correct:
                        model_pattern_data[model][pattern]['correct'] += weight
            
            # Calculate model-specific accuracy for each pattern
            model_pattern_accuracy = {}
            
            for model, patterns in model_pattern_data.items():
                model_pattern_accuracy[model] = {}
                
                for pattern, stats in patterns.items():
                    if stats['total'] >= 5:  # Minimum sample size
                        accuracy = (stats['correct'] / stats['total']) * 100
                        model_pattern_accuracy[model][pattern] = {
                            'accuracy': accuracy,
                            'samples': stats['total'],
                            'vs_overall': accuracy - overall_accuracy
                        }
            
            # Find top models for each pattern
            top_pattern_models = {}
            
            for pattern in pattern_accuracy.keys():
                pattern_models = []
                
                for model, patterns in model_pattern_accuracy.items():
                    if pattern in patterns:
                        pattern_models.append((model, patterns[pattern]['accuracy']))
                
                # Sort by accuracy
                pattern_models.sort(key=lambda x: x[1], reverse=True)
                
                # Take top N models
                top_pattern_models[pattern] = pattern_models[:top_models]
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Pattern accuracy comparison
        plt.subplot(2, 1, 1)
        
        # Sort patterns by accuracy
        sorted_patterns = sorted(pattern_accuracy.items(), key=lambda x: x[1], reverse=True)
        patterns = [p[0] for p in sorted_patterns]
        accuracies = [p[1] for p in sorted_patterns]
        counts = [pattern_counts[p] for p in patterns]
        
        # Create bar chart
        bars = plt.bar(
            range(len(patterns)),
            accuracies,
            color=[PATTERN_COLORS.get(p, '#777777') for p in patterns],
            alpha=0.7
        )
        
        # Add count labels
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"n={counts[i]}",
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # Add overall accuracy reference line
        plt.axhline(y=overall_accuracy, color='red', linestyle='--', 
                   label=f'Overall: {overall_accuracy:.1f}%')
        
        # Add random baseline
        plt.axhline(y=33.33, color='gray', linestyle=':', alpha=0.7, 
                   label='Random: 33.3%')
        
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Prediction Accuracy by Pattern Type', fontsize=14)
        plt.xticks(range(len(patterns)), patterns)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        # Plot 2: Best models for each pattern
        if 'top_pattern_models' in locals():
            plt.subplot(2, 1, 2)
            
            # Create grouped bar chart for top models by pattern
            bar_width = 0.8 / top_models
            
            for i, pattern in enumerate(patterns):
                if pattern in top_pattern_models:
                    for j, (model, accuracy) in enumerate(top_pattern_models[pattern]):
                        pos = i + (j - top_models/2 + 0.5) * bar_width
                        plt.bar(
                            pos,
                            accuracy,
                            bar_width,
                            color=plt.cm.viridis(j / top_models),
                            alpha=0.7,
                            label=model if i == 0 else ""
                        )
            
            plt.xlabel('Pattern Type', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title(f'Top {top_models} Models for Each Pattern Type', fontsize=14)
            plt.xticks(range(len(patterns)), patterns)
            plt.grid(axis='y', alpha=0.3)
            plt.legend(title='Model', title_fontsize=10)
        
        plt.tight_layout()
        plt.savefig('data/images/pattern_feature_impact.png', dpi=300)
        plt.show()
        
        # Return analysis results
        results = {
            'pattern_accuracy': pattern_accuracy,
            'pattern_counts': pattern_counts,
            'overall_accuracy': overall_accuracy,
            'sample_count': len(log_df)
        }
        
        # Add model-specific pattern performance if available
        if 'model_pattern_accuracy' in locals():
            results['model_pattern_accuracy'] = model_pattern_accuracy
            results['top_pattern_models'] = top_pattern_models
        
        return results
        
    except Exception as e:
        print(f"Error analyzing pattern feature impact: {e}")
        import traceback
        traceback.print_exc()
        return None

def temporal_feature_analysis(days=None, window_size=20):
    """
    Analyze how pattern effectiveness changes over time.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        window_size: Window size for rolling pattern analysis
        
    Returns:
        dict: Temporal analysis of pattern effectiveness
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check for required columns
        if 'pattern_type' not in log_df.columns or 'Correct' not in log_df.columns:
            print("Log file missing pattern or correctness data")
            return None
        
        # Apply time filtering if requested
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {days} days ({len(log_df)} predictions)")
        
        # Sort by time if available, otherwise by index
        if 'Timestamp' in log_df.columns:
            log_df = log_df.sort_values('Timestamp')
        
        # Get pattern types
        patterns = [p for p in log_df['pattern_type'].unique() if not pd.isna(p)]
        
        # Calculate rolling accuracy for each pattern
        rolling_pattern_accuracy = {}
        
        for pattern in patterns:
            pattern_indices = log_df[log_df['pattern_type'] == pattern].index
            
            # Skip if too few examples
            if len(pattern_indices) < window_size:
                continue
                
            # Create windows for this pattern
            windows = []
            
            for i in range(len(pattern_indices) - window_size + 1):
                window_indices = pattern_indices[i:i+window_size]
                window_df = log_df.loc[window_indices]
                
                # Calculate accuracy for this window
                accuracy = window_df['Correct'].mean() * 100
                
                # Get timestamp or index for x-axis
                if 'Timestamp' in window_df.columns:
                    x_value = window_df['Timestamp'].iloc[-1]  # Last timestamp in window
                else:
                    x_value = window_indices[-1]  # Last index in window
                
                windows.append({
                    'x_value': x_value,
                    'accuracy': accuracy,
                    'window_size': len(window_df)
                })
            
            rolling_pattern_accuracy[pattern] = windows
        
        # Calculate rolling overall accuracy for reference
        overall_windows = []
        
        for i in range(0, len(log_df) - window_size + 1, max(1, window_size // 5)):  # Step size for efficiency
            window_df = log_df.iloc[i:i+window_size]
            
            accuracy = window_df['Correct'].mean() * 100
            
            if 'Timestamp' in window_df.columns:
                x_value = window_df['Timestamp'].iloc[-1]
            else:
                x_value = i + window_size - 1
                
            overall_windows.append({
                'x_value': x_value,
                'accuracy': accuracy,
                'window_size': len(window_df)
            })
            
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Check if x values are timestamps
        is_time_based = all(isinstance(w['x_value'], (pd.Timestamp, datetime)) 
                        for windows in rolling_pattern_accuracy.values() 
                        for w in windows)
        
        # Plot pattern accuracy trends
        for pattern, windows in rolling_pattern_accuracy.items():
            x_values = [w['x_value'] for w in windows]
            accuracies = [w['accuracy'] for w in windows]
            
            plt.plot(
                x_values,
                accuracies,
                label=pattern,
                color=PATTERN_COLORS.get(pattern, '#777777'),
                alpha=0.7,
                linewidth=2
            )
        
        # Plot overall accuracy
        x_values = [w['x_value'] for w in overall_windows]
        accuracies = [w['accuracy'] for w in overall_windows]
        
        plt.plot(
            x_values,
            accuracies,
            label='Overall',
            color='black',
            linestyle='--',
            alpha=0.7,
            linewidth=2
        )
        
        # Add random baseline
        if is_time_based:
            plt.axhline(y=33.33, color='gray', linestyle=':', alpha=0.5, label='Random: 33.3%')
        else:
            plt.plot(x_values, [33.33] * len(x_values), 'gray:', alpha=0.5, label='Random: 33.3%')
        
        # Format axes
        if is_time_based:
            plt.gcf().autofmt_xdate()
            plt.xlabel('Date', fontsize=12)
        else:
            plt.xlabel('Prediction Index', fontsize=12)
            
        plt.ylabel('Rolling Accuracy (%)', fontsize=12)
        plt.title(f'{window_size}-Prediction Rolling Accuracy by Pattern Type', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Pattern Type', title_fontsize=10)
        
        plt.tight_layout()
        plt.savefig('data/images/temporal_pattern_effectiveness.png', dpi=300)
        plt.show()
        
        # Calculate trend metrics
        trend_metrics = {}
        
        for pattern, windows in rolling_pattern_accuracy.items():
            if len(windows) >= 2:
                start = windows[0]['accuracy']
                end = windows[-1]['accuracy']
                change = end - start
                
                trend_metrics[pattern] = {
                    'start_accuracy': start,
                    'end_accuracy': end,
                    'change': change,
                    'trend': 'improving' if change > 5 else 'declining' if change < -5 else 'stable'
                }
        
        return {
            'rolling_pattern_accuracy': rolling_pattern_accuracy,
            'overall_accuracy_trend': overall_windows,
            'trend_metrics': trend_metrics,
            'window_size': window_size
        }
        
    except Exception as e:
        print(f"Error analyzing temporal pattern features: {e}")
        import traceback
        traceback.print_exc()
        return None

def feature_correlation_matrix(days=None):
    """
    Create a correlation matrix showing relationships between different pattern types
    and prediction outcomes.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        
    Returns:
        dict: Correlation analysis results
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check for required columns
        if 'pattern_type' not in log_df.columns or 'Actual' not in log_df.columns:
            print("Log file missing pattern type or outcome data")
            return None
        
        # Apply time filtering if requested
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            print(f"Analyzing data from the last {days} days ({len(log_df)} predictions)")
        
        # Create dummy variables for pattern types and outcomes
        pattern_dummies = pd.get_dummies(log_df['pattern_type'], prefix='pattern')
        outcome_dummies = pd.get_dummies(log_df['Actual'], prefix='outcome')
        
        # Rename outcome columns for clarity
        outcome_mapping = {
            'outcome_0': 'outcome_banker',
            'outcome_1': 'outcome_player',
            'outcome_2': 'outcome_tie'
        }
        outcome_dummies = outcome_dummies.rename(columns=outcome_mapping)
        
        # Add correctness column
        analysis_df = pd.concat([pattern_dummies, outcome_dummies, log_df['Correct']], axis=1)
        
        # Calculate correlation matrix
        corr_matrix = analysis_df.corr()
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Customize colormap for better contrast
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Create heatmap with annotations
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Correlation between Pattern Types and Outcomes', fontsize=16)
        
        plt.tight_layout()
        plt.savefig('data/images/pattern_correlation_matrix.png', dpi=300)
        plt.show()
        
        # Extract key correlations
        # Pattern-outcome correlations
        pattern_outcome_corr = {}
        for pattern_col in pattern_dummies.columns:
            pattern = pattern_col.replace('pattern_', '')
            
            for outcome_col in outcome_dummies.columns:
                outcome = outcome_col.replace('outcome_', '')
                
                if pattern_col in corr_matrix.index and outcome_col in corr_matrix.columns:
                    correlation = corr_matrix.loc[pattern_col, outcome_col]
                    
                    if pattern not in pattern_outcome_corr:
                        pattern_outcome_corr[pattern] = {}
                        
                    pattern_outcome_corr[pattern][outcome] = correlation
        
        # Pattern-correct correlations
        pattern_correct_corr = {}
        for pattern_col in pattern_dummies.columns:
            pattern = pattern_col.replace('pattern_', '')
            
            if pattern_col in corr_matrix.index and 'Correct' in corr_matrix.columns:
                correlation = corr_matrix.loc[pattern_col, 'Correct']
                pattern_correct_corr[pattern] = correlation
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'pattern_outcome_correlation': pattern_outcome_corr,
            'pattern_correct_correlation': pattern_correct_corr,
            'sample_count': len(log_df)
        }
        
    except Exception as e:
        print(f"Error creating feature correlation matrix: {e}")
        import traceback
        traceback.print_exc()
        return None