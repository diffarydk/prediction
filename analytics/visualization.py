"""
Advanced visualization tools for Baccarat prediction system with interactive capabilities.
This module provides high-quality visualizations for model performance, prediction patterns,
and data exploration with enhanced readability and interactivity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from collections import defaultdict, Counter
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from colorama import Fore, Style 

from config import LOG_FILE, REALTIME_FILE, MODEL_REGISTRY_PATH
from analytics.analytics import ensure_image_directory

# Define standard model and pattern colors for consistency
MODEL_COLORS = {
    'baccarat_rf': '#4285F4',       # Google Blue
    'markov_1': '#EA4335',          # Google Red
    'markov_2': '#FBBC05',          # Google Yellow
    'xgboost_base': '#34A853',      # Google Green
    'stacking_ensemble': '#9C27B0', # Purple
    'xgb_variant': '#FF9800',       # Orange
    'markov_3': '#00BCD4',          # Cyan
}

PATTERN_COLORS = {
    'no_pattern': '#7F7F7F',    # Gray
    'streak': '#E31A1C',        # Red
    'alternating': '#1F78B4',   # Blue
    'tie': '#33A02C',           # Green
}

def plot_model_comparison(days=None, metric='accuracy'):
    """
    Create enhanced comparative visualization of model performance.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        metric: Performance metric to visualize ('accuracy', 'calibration', 'confidence')
        
    Returns:
        dict: Visualization metadata
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
        model_data = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidence': []})
        
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
            confidence = row.get('Confidence', 50.0)
            
            # Update metrics for each model
            for model, weight in contribs.items():
                model_data[model]['total'] += weight
                if correct:
                    model_data[model]['correct'] += weight
                model_data[model]['confidence'].append((confidence, weight))
        
        # Calculate performance metrics
        model_metrics = {}
        
        for model, data in model_data.items():
            if data['total'] > 0:
                # Calculate accuracy
                accuracy = (data['correct'] / data['total']) * 100
                
                # Calculate weighted confidence
                weighted_confidence = sum(conf * weight for conf, weight in data['confidence']) / sum(weight for _, weight in data['confidence'])
                
                # Calculate calibration error (|confidence - accuracy|)
                calibration_error = abs(weighted_confidence - accuracy)
                
                model_metrics[model] = {
                    'accuracy': accuracy,
                    'confidence': weighted_confidence,
                    'calibration_error': calibration_error,
                    'samples': data['total']
                }
        
        # Create visualization based on selected metric
        plt.figure(figsize=(14, 10))
        
        if metric == 'accuracy':
            # Sort models by accuracy
            sorted_models = sorted(
                [(model, metrics['accuracy']) for model, metrics in model_metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            models = [m[0] for m in sorted_models]
            values = [m[1] for m in sorted_models]
            
            # Create enhanced bar chart
            bars = plt.bar(
                range(len(models)),
                values,
                color=[MODEL_COLORS.get(m, '#777777') for m in models],
                alpha=0.7,
                width=0.6
            )
            
            # Add data point markers for more precise reading
            plt.plot(range(len(models)), values, 'ko', alpha=0.7)
            
            # Add sample count and exact value labels
            for i, bar in enumerate(bars):
                # Show sample count above bar
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f"n={model_metrics[models[i]]['samples']:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
                
                # Show exact value inside bar
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() / 2,
                    f"{values[i]:.2f}%",
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='white',
                    fontweight='bold'
                )
            
            # Add overall accuracy reference line
            overall_acc = log_df['Correct'].mean() * 100
            plt.axhline(
                y=overall_acc,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Overall: {overall_acc:.2f}%'
            )
            
            # Add random baseline (33.33% for 3-class problem)
            plt.axhline(
                y=33.33,
                color='gray',
                linestyle=':',
                alpha=0.7,
                label='Random: 33.33%'
            )
            
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Accuracy (%)', fontsize=14)
            plt.title('Model Accuracy Comparison', fontsize=16)
            plt.ylim(0, 100)
            
        elif metric == 'calibration':
            # Sort models by calibration error (lower is better)
            sorted_models = sorted(
                [(model, metrics['calibration_error']) for model, metrics in model_metrics.items()],
                key=lambda x: x[1]
            )
            
            models = [m[0] for m in sorted_models]
            values = [m[1] for m in sorted_models]
            
            # Create enhanced bar chart
            bars = plt.bar(
                range(len(models)),
                values,
                color=[plt.cm.RdYlGn_r(min(v / 30, 1.0)) for v in values],  # Color by error magnitude
                alpha=0.8,
                width=0.6
            )
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{values[i]:.2f}%",
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
            
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Calibration Error (|confidence - accuracy|)', fontsize=14)
            plt.title('Model Calibration Comparison (Lower Is Better)', fontsize=16)
            
        elif metric == 'confidence':
            # Create scatter plot of confidence vs accuracy
            plt.figure(figsize=(12, 10))
            
            models = list(model_metrics.keys())
            accuracies = [model_metrics[m]['accuracy'] for m in models]
            confidences = [model_metrics[m]['confidence'] for m in models]
            sizes = [model_metrics[m]['samples'] * 0.5 for m in models]  # Scale by sample count
            
            # Plot scatter points
            plt.scatter(
                confidences,
                accuracies,
                s=sizes,
                c=[MODEL_COLORS.get(m, '#777777') for m in models],
                alpha=0.7,
                edgecolors='black'
            )
            
            # Add model labels
            for i, model in enumerate(models):
                plt.annotate(
                    model,
                    (confidences[i], accuracies[i]),
                    xytext=(7, 0),
                    textcoords='offset points',
                    fontsize=10
                )
            
            # Add perfect calibration line
            max_val = max(max(confidences), max(accuracies)) + 5
            plt.plot(
                [0, max_val],
                [0, max_val],
                'r--',
                alpha=0.5,
                label='Perfect Calibration'
            )
            
            # Add explanatory regions
            plt.fill_between(
                [0, max_val],
                [0, max_val],
                [max_val, max_val],
                alpha=0.1,
                color='blue',
                label='Underconfident'
            )
            
            plt.fill_between(
                [0, max_val],
                [0, 0],
                [0, max_val],
                alpha=0.1,
                color='red',
                label='Overconfident'
            )
            
            plt.xlabel('Average Confidence (%)', fontsize=14)
            plt.ylabel('Actual Accuracy (%)', fontsize=14)
            plt.title('Model Calibration: Confidence vs. Accuracy', fontsize=16)
            plt.grid(True, alpha=0.3)
            
        # Enhance all visualizations
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save visualization
        filename = f'data/images/model_{metric}_comparison.png'
        plt.savefig(filename, dpi=300)
        plt.show()
        
        return {
            'metric': metric,
            'models': models,
            'model_metrics': model_metrics,
            'image_path': filename
        }
        
    except Exception as e:
        print(f"Error creating model comparison visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_feature_importance_heatmap():
    """
    Create a heatmap visualization of feature importance across model outcomes.
    
    Returns:
        dict: Visualization metadata
    """
    ensure_image_directory()
    
    try:
        # Load stacking model to extract coefficients
        stacking_file = os.path.join(MODEL_REGISTRY_PATH, "stacking_ensemble.pkl")
        if not os.path.exists(stacking_file):
            print("Stacking model file not found.")
            return None
            
        with open(stacking_file, 'rb') as f:
            import pickle
            stacking_model = pickle.load(f)
            
        # Check if model has coefficients
        if not hasattr(stacking_model, 'meta_model') or not hasattr(stacking_model.meta_model, 'coef_'):
            print("Stacking model doesn't have accessible coefficients.")
            return None
            
        # Extract coefficients
        coef = stacking_model.meta_model.coef_
        
        # Generate feature names
        with open(os.path.join(MODEL_REGISTRY_PATH, "registry.json"), 'r') as f:
            registry_data = json.load(f)
            
        base_models = [m for m in registry_data.get("model_ids", []) if m != "stacking_ensemble"]
        
        feature_names = []
        for model in base_models:
            for outcome in ['Banker', 'Player', 'Tie']:
                feature_names.append(f"{model}_{outcome}")
                
        # Add pattern feature names
        for pattern in ['no_pattern', 'streak', 'alternating', 'tie']:
            feature_names.append(f"pattern_{pattern}")
            
        # Adjust feature names length to match coefficients
        if len(feature_names) != coef.shape[1]:
            print(f"Warning: Feature name count ({len(feature_names)}) doesn't match coefficient count ({coef.shape[1]})")
            
            if len(feature_names) > coef.shape[1]:
                feature_names = feature_names[:coef.shape[1]]
            else:
                for i in range(len(feature_names), coef.shape[1]):
                    feature_names.append(f"Feature_{i+1}")
        
        # Create enhanced heatmap visualization
        plt.figure(figsize=(16, 12))
        
        # Set up a grid layout with main heatmap and two marginal histograms
        gs = GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 12])
        
        # Main heatmap
        ax_heatmap = plt.subplot(gs[1, 0])
        
        # Create class-specific labels (positive/negative effect on each class)
        class_names = ['Banker', 'Player', 'Tie']
        expanded_class_names = []
        for name in class_names:
            expanded_class_names.append(f"+{name}")  # Positive effect
            expanded_class_names.append(f"-{name}")  # Negative effect
        
        # Prepare data for visualization (separate positive and negative effects)
        viz_data = np.zeros((len(feature_names), len(expanded_class_names)))
        
        for i in range(coef.shape[0]):  # For each class
            for j in range(coef.shape[1]):  # For each feature
                value = coef[i, j]
                if value >= 0:
                    viz_data[j, i*2] = value  # Positive effect
                else:
                    viz_data[j, i*2+1] = -value  # Negative effect (make positive for visualization)
        
        # Create custom colormap with different colors for positive/negative
        pos_cmap = plt.cm.Greens
        neg_cmap = plt.cm.Reds
        
        # Plot the main heatmap with custom colors
        im = ax_heatmap.imshow(
            viz_data,
            aspect='auto',
            cmap='coolwarm',
            interpolation='nearest'
        )
        
        # Add colorbar
        plt.colorbar(im, cax=plt.subplot(gs[1, 1]))
        
        # Add feature labels
        ax_heatmap.set_yticks(range(len(feature_names)))
        ax_heatmap.set_yticklabels(feature_names)
        
        # Add class labels
        ax_heatmap.set_xticks(range(len(expanded_class_names)))
        ax_heatmap.set_xticklabels(expanded_class_names, rotation=45, ha='right')
        
        # Add feature type colors to y-tick labels
        for i, tick in enumerate(ax_heatmap.get_yticklabels()):
            feature = feature_names[i]
            if 'pattern_' in feature:
                tick.set_color('#8B0000')  # Dark red for pattern features
            else:
                model_name = feature.split('_')[0]
                tick.set_color(MODEL_COLORS.get(model_name, '#333333'))
        
        # Add feature sum histogram on top
        ax_top = plt.subplot(gs[0, 0], sharex=ax_heatmap)
        feature_importance = np.abs(coef).mean(axis=0)
        ax_top.bar(
            range(len(feature_names)),
            feature_importance,
            color=[
                '#8B0000' if 'pattern_' in f else MODEL_COLORS.get(f.split('_')[0], '#333333')
                for f in feature_names
            ],
            alpha=0.7
        )
        ax_top.set_ylabel('Importance')
        ax_top.set_title('Feature Importance in Stacking Ensemble', fontsize=16)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        
        plt.tight_layout()
        plt.savefig('data/images/feature_importance_heatmap.png', dpi=300)
        plt.show()
        
        return {
            'feature_names': feature_names,
            'class_names': class_names,
            'image_path': 'data/images/feature_importance_heatmap.png'
        }
        
    except Exception as e:
        print(f"Error creating feature importance heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_confidence_distribution(by_pattern=False):
    """
    Visualize the distribution of prediction confidence and its relationship with accuracy.
    
    Args:
        by_pattern: If True, break down by pattern type
        
    Returns:
        dict: Visualization metadata
    """
    ensure_image_directory()
    if not os.path.exists(LOG_FILE):
        print("No prediction log found.")
        return None
    
    try:
        log_df = pd.read_csv(LOG_FILE)
        
        # Check required columns
        if 'Confidence' not in log_df.columns or 'Correct' not in log_df.columns:
            print("Log file missing confidence or correctness data")
            return None
        
        # Create confidence bins
        log_df['ConfidenceBin'] = pd.cut(
            log_df['Confidence'],
            bins=[0, 40, 50, 60, 70, 80, 90, 100],
            labels=['0-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        )
        
        # Calculate accuracy by confidence bin
        bin_results = log_df.groupby('ConfidenceBin')['Correct'].agg(['mean', 'count']).reset_index()
        bin_results['accuracy'] = bin_results['mean'] * 100
        
        # Overall visualization
        plt.figure(figsize=(14, 10))
        
        # If breaking down by pattern
        if by_pattern and 'pattern_type' in log_df.columns:
            # Create a grid of subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Confidence vs. Accuracy by Pattern Type', fontsize=18)
            axes = axes.flatten()
            
            # Overall plot in first position
            ax = axes[0]
            bar_positions = range(len(bin_results))
            bars = ax.bar(
                bar_positions,
                bin_results['accuracy'],
                color=plt.cm.viridis(bin_results['accuracy']/100),
                alpha=0.7
            )
            
            # Add perfect calibration reference line
            conf_values = [30, 45, 55, 65, 75, 85, 95]  # Middle of each bin
            ax.plot(
                bar_positions,
                conf_values[:len(bin_results)],
                'r--',
                label='Perfect Calibration'
            )
            
            # Add count labels
            for i, bar in enumerate(bars):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 2,
                    f"{int(bin_results.iloc[i]['count'])}",
                    ha='center',
                    fontsize=9
                )
            
            ax.set_title('Overall', fontsize=14)
            ax.set_xlabel('Confidence Range', fontsize=12)
            ax.set_ylabel('Actual Accuracy (%)', fontsize=12)
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(bin_results['ConfidenceBin'], rotation=45, ha='right')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            
            # Plot for each pattern type
            pattern_types = log_df['pattern_type'].unique()
            for i, pattern in enumerate(pattern_types[:3], 1):  # Limit to 3 patterns
                if pd.isna(pattern):
                    continue
                    
                ax = axes[i]
                
                # Filter by pattern
                pattern_df = log_df[log_df['pattern_type'] == pattern]
                
                # Skip if too little data
                if len(pattern_df) < 10:
                    ax.text(
                        0.5, 0.5,
                        f"Insufficient data for\n{pattern} pattern",
                        ha='center',
                        va='center',
                        fontsize=12,
                        transform=ax.transAxes
                    )
                    continue
                
                # Calculate accuracy by confidence bin for this pattern
                pattern_bins = pattern_df.groupby('ConfidenceBin')['Correct'].agg(['mean', 'count']).reset_index()
                pattern_bins['accuracy'] = pattern_bins['mean'] * 100
                
                # Plot bars
                bar_positions = range(len(pattern_bins))
                bars = ax.bar(
                    bar_positions,
                    pattern_bins['accuracy'],
                    color=PATTERN_COLORS.get(pattern, '#777777'),
                    alpha=0.7
                )
                
                # Add perfect calibration line
                conf_values = [30, 45, 55, 65, 75, 85, 95]  # Middle of each bin
                ax.plot(
                    bar_positions,
                    conf_values[:len(pattern_bins)],
                    'r--',
                    label='Perfect Calibration'
                )
                
                # Add count labels
                for i, bar in enumerate(bars):
                    if i < len(pattern_bins):
                        ax.text(
                            bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 2,
                            f"{int(pattern_bins.iloc[i]['count'])}",
                            ha='center',
                            fontsize=9
                        )
                
                ax.set_title(f"{pattern.capitalize()} Pattern", fontsize=14)
                ax.set_xlabel('Confidence Range', fontsize=12)
                ax.set_ylabel('Actual Accuracy (%)', fontsize=12)
                ax.set_xticks(bar_positions)
                ax.set_xticklabels(pattern_bins['ConfidenceBin'], rotation=45, ha='right')
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=0.3)
                ax.legend()
            
            # Add explanatory text
            plt.figtext(
                0.5, 0.02,
                "Bars above the calibration line indicate underconfidence (system is more accurate than confident)\n"
                "Bars below the line indicate overconfidence (system is less accurate than confident)",
                ha="center",
                fontsize=11,
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
            )
            
        else:
            # Simple single plot for overall confidence calibration
            bar_positions = range(len(bin_results))
            bars = plt.bar(
                bar_positions,
                bin_results['accuracy'],
                color=plt.cm.viridis(bin_results['accuracy']/100)
            )
            
            # Add count labels
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 2,
                    f"{int(bin_results.iloc[i]['count'])} pred.",
                    ha='center',
                    fontsize=10
                )
            
            # Add perfect calibration reference line
            conf_values = [30, 45, 55, 65, 75, 85, 95]  # Middle of each bin
            plt.plot(
                bar_positions,
                conf_values[:len(bin_results)],
                'r--',
                label='Perfect Calibration'
            )
            
            plt.xlabel('Confidence Range', fontsize=14)
            plt.ylabel('Actual Accuracy (%)', fontsize=14)
            plt.title('Prediction Confidence vs. Actual Accuracy', fontsize=16)
            plt.xticks(bar_positions, bin_results['ConfidenceBin'])
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            
            # Add explanatory text
            plt.figtext(
                0.5, 0.01,
                "Bars above the calibration line indicate underconfidence; below the line indicate overconfidence",
                ha="center",
                fontsize=11,
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
            )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('data/images/confidence_distribution.png', dpi=300)
        plt.show()
        
        # Add calibration metrics
        from analytics.analytics import calculate_calibration_metrics
        calibration_metrics = calculate_calibration_metrics(log_df)
        
        return {
            'bin_results': bin_results.to_dict(),
            'calibration_metrics': calibration_metrics,
            'image_path': 'data/images/confidence_distribution.png'
        }
        
    except Exception as e:
        print(f"Error visualizing confidence distribution: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_interactive_dashboard():
    """
    Create an interactive HTML dashboard with all key visualizations.
    This is a simplified version that combines multiple static visualizations.
    
    Returns:
        str: Path to generated HTML dashboard
    """
    ensure_image_directory()
    
    try:
        # Generate all key visualizations
        plot_model_comparison(metric='accuracy')
        plot_feature_importance_heatmap()
        visualize_confidence_distribution()
        
        # Create simple HTML dashboard
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Baccarat Prediction Analytics Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .dashboard {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1, h2 {
                    color: #333;
                }
                .visualization {
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                    border: 1px solid #ddd;
                }
                .timestamp {
                    text-align: right;
                    color: #666;
                    font-size: 0.9em;
                    margin-top: 30px;
                }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1>Baccarat Prediction Analytics Dashboard</h1>
                <p>Comprehensive visualization of model performance, feature importance, and prediction patterns.</p>
                
                <div class="visualization">
                    <h2>Model Performance Comparison</h2>
                    <img src="../images/model_accuracy_comparison.png" alt="Model Accuracy Comparison">
                    <p>This chart compares the accuracy of different models in the prediction system.</p>
                </div>
                
                <div class="visualization">
                    <h2>Feature Importance Heatmap</h2>
                    <img src="../images/feature_importance_heatmap.png" alt="Feature Importance Heatmap">
                    <p>This heatmap shows the importance of different features in the stacking ensemble model.</p>
                </div>
                
                <div class="visualization">
                    <h2>Confidence Calibration</h2>
                    <img src="../images/confidence_distribution.png" alt="Confidence Distribution">
                    <p>This chart compares prediction confidence with actual accuracy to evaluate calibration.</p>
                </div>
                
                <div class="timestamp">
                    Generated on: %s
                </div>
            </div>
        </body>
        </html>
        """ % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create output directory
        os.makedirs("data/dashboard", exist_ok=True)
        
        # Save HTML file
        dashboard_path = "data/dashboard/analytics_dashboard.html"
        with open(dashboard_path, "w") as f:
            f.write(html_content)
        
        print(f"Interactive dashboard created at {dashboard_path}")
        return dashboard_path
        
    except Exception as e:
        print(f"Error creating interactive dashboard: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_visualizations_to_html(include_all=False):
    """
    Export all visualizations to an HTML report.
    
    Args:
        include_all: If True, include all visualizations, otherwise just key ones
    
    Returns:
        str: Path to generated HTML report
    """
    ensure_image_directory()
    
    try:
        # Determine which visualizations to include
        visualizations = [
            {
                'title': 'Model Accuracy Comparison',
                'filename': 'model_accuracy_comparison.png',
                'description': 'Comparison of accuracy across different prediction models.'
            },
            {
                'title': 'Prediction Distribution',
                'filename': 'prediction_distribution.png',
                'description': 'Distribution of predicted vs actual outcomes.'
            },
            {
                'title': 'Confidence Calibration',
                'filename': 'confidence_distribution.png',
                'description': 'Relationship between prediction confidence and actual accuracy.'
            }
        ]
        
        if include_all:
            # Add more visualizations
            additional_viz = [
                {
                    'title': 'Feature Importance',
                    'filename': 'meta_feature_importance.png',
                    'description': 'Importance of different features in the stacking ensemble model.'
                },
                {
                    'title': 'Temporal Pattern',
                    'filename': 'temporal_pattern.png',
                    'description': 'Analysis of temporal patterns in outcome distributions.'
                },
                {
                    'title': 'Pattern Effectiveness',
                    'filename': 'pattern_effectiveness.png',
                    'description': 'Effectiveness of different pattern types in predictions.'
                },
                {
                    'title': 'Model Pattern Breakdown',
                    'filename': 'model_pattern_breakdown.png',
                    'description': 'Breakdown of model performance by pattern type.'
                },
                {
                    'title': 'Markov Transitions',
                    'filename': 'markov_transitions.png',
                    'description': 'Transition probabilities between different outcomes.'
                }
            ]
            
            visualizations.extend(additional_viz)
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Baccarat Prediction Visualization Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .report {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1, h2 {
                    color: #333;
                }
                .visualization {
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                    border: 1px solid #ddd;
                }
                .timestamp {
                    text-align: right;
                    color: #666;
                    font-size: 0.9em;
                    margin-top: 30px;
                }
                .nav {
                    position: sticky;
                    top: 0;
                    background-color: white;
                    padding: 10px 0;
                    margin-bottom: 20px;
                    display: flex;
                    gap: 10px;
                    overflow-x: auto;
                }
                .nav-item {
                    padding: 8px 12px;
                    background-color: #eee;
                    border-radius: 4px;
                    text-decoration: none;
                    color: #333;
                    white-space: nowrap;
                }
                .nav-item:hover {
                    background-color: #ddd;
                }
            </style>
        </head>
        <body>
            <div class="report">
                <h1>Baccarat Prediction Visualization Report</h1>
                <p>Generated on: %s</p>
                
                <div class="nav">
        """ % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add navigation items
        for i, viz in enumerate(visualizations):
            html_content += f'<a href="#viz{i}" class="nav-item">{viz["title"]}</a>\n'
        
        html_content += """
                </div>
        """
        
        # Add visualizations
        for i, viz in enumerate(visualizations):
            html_content += f"""
                <div id="viz{i}" class="visualization">
                    <h2>{viz['title']}</h2>
                    <img src="../images/{viz['filename']}" alt="{viz['title']}">
                    <p>{viz['description']}</p>
                </div>
            """
        
        html_content += """
                <div class="timestamp">
                    Generated on: %s
                </div>
            </div>
        </body>
        </html>
        """ % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create output directory
        os.makedirs("data/reports", exist_ok=True)
        
        # Save HTML file
        report_path = f"data/reports/visualization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, "w") as f:
            f.write(html_content)
        
        print(f"Visualization report created at {report_path}")
        return report_path
        
    except Exception as e:
        print(f"Error exporting visualizations: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_temporal_pattern(window_size=10):
    """
    Plot enhanced visualization of temporal patterns in outcomes.
    
    Args:
        window_size: Size of the rolling window for smoothing
        
    Returns:
        dict: Visualization metadata
    """
    ensure_image_directory()
    if not os.path.exists(REALTIME_FILE):
        print("No real-time data found.")
        return None
    
    try:
        df = pd.read_csv(REALTIME_FILE)
        
        if 'Target' not in df.columns:
            print("Data doesn't have Target column")
            return None
            
        # Create a time-series representation of outcomes
        sequence = df['Target'].tolist()
        
        if len(sequence) < window_size:
            print(f"Not enough data for temporal analysis (minimum {window_size} points needed)")
            return None
        
        # Calculate rolling proportions for each outcome
        banker_ratio = []
        player_ratio = []
        tie_ratio = []
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            banker_ratio.append(window.count(0) / window_size * 100)
            player_ratio.append(window.count(1) / window_size * 100)
            tie_ratio.append(window.count(2) / window_size * 100)
        
        # Create enhanced visualization with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        
        # Plot 1: Rolling percentages
        ax1.plot(range(len(banker_ratio)), banker_ratio, 'g-', label='Banker %', linewidth=2)
        ax1.plot(range(len(player_ratio)), player_ratio, 'b-', label='Player %', linewidth=2)
        ax1.plot(range(len(tie_ratio)), tie_ratio, 'r-', label='Tie %', linewidth=2, alpha=0.7)
        
        # Add reference lines
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% line')
        
        # Calculate and show average proportions
        avg_banker = sum(banker_ratio) / len(banker_ratio)
        avg_player = sum(player_ratio) / len(player_ratio)
        avg_tie = sum(tie_ratio) / len(tie_ratio)
        
        ax1.axhline(y=avg_banker, color='g', linestyle=':', alpha=0.5, 
                   label=f'Avg Banker: {avg_banker:.1f}%')
        ax1.axhline(y=avg_player, color='b', linestyle=':', alpha=0.5, 
                   label=f'Avg Player: {avg_player:.1f}%')
        
        # Calculate standard deviations
        std_banker = np.std(banker_ratio)
        std_player = np.std(player_ratio)
        
        # Add volatility bands (±1 standard deviation)
        ax1.fill_between(
            range(len(banker_ratio)),
            [avg_banker - std_banker] * len(banker_ratio),
            [avg_banker + std_banker] * len(banker_ratio),
            color='g', alpha=0.1
        )
        
        ax1.fill_between(
            range(len(player_ratio)),
            [avg_player - std_player] * len(player_ratio),
            [avg_player + std_player] * len(player_ratio),
            color='b', alpha=0.1
        )
        
        # Enhance the plot
        ax1.set_title('Temporal Pattern of Outcomes', fontsize=16)
        ax1.set_ylabel('Percentage in Window (%)', fontsize=14)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Plot 2: Actual outcome sequence (heatmap-style)
        cmap = mcolors.ListedColormap(['green', 'blue', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Create sequence heatmap (one cell per outcome)
        sequence_img = ax2.imshow(
            [sequence],
            aspect='auto',
            cmap=cmap,
            norm=norm,
            interpolation='nearest'
        )
        
        # Add colorbar with outcome labels
        cbar = plt.colorbar(sequence_img, ax=ax2, ticks=[0, 1, 2], orientation='horizontal', pad=0.05)
        cbar.set_label('Outcome')
        cbar.set_ticklabels(['Banker', 'Player', 'Tie'])
        
        ax2.set_yticks([])
        ax2.set_xlabel('Game Sequence', fontsize=14)
        
        # Add annotations for longest streaks
        banker_streak, banker_pos = find_longest_streak(sequence, 0)
        player_streak, player_pos = find_longest_streak(sequence, 1)
        tie_streak, tie_pos = find_longest_streak(sequence, 2)
        
        if banker_streak >= 3:
            ax2.annotate(
                f'Banker streak: {banker_streak}',
                xy=(banker_pos, 0),
                xytext=(banker_pos, -0.5),
                arrowprops=dict(facecolor='green', shrink=0.05),
                fontsize=10,
                ha='center'
            )
            
        if player_streak >= 3:
            ax2.annotate(
                f'Player streak: {player_streak}',
                xy=(player_pos, 0),
                xytext=(player_pos, -0.5),
                arrowprops=dict(facecolor='blue', shrink=0.05),
                fontsize=10,
                ha='center'
            )
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                   f"Analysis based on {window_size}-game rolling window. "
                   f"Volatility bands show ±1 standard deviation from average.", 
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('data/images/temporal_pattern_enhanced.png', dpi=300)
        plt.show()
        
        return {
            'avg_banker': avg_banker,
            'avg_player': avg_player,
            'avg_tie': avg_tie,
            'std_banker': std_banker,
            'std_player': std_player,
            'banker_streak': banker_streak,
            'player_streak': player_streak,
            'image_path': 'data/images/temporal_pattern_enhanced.png'
        }
        
    except Exception as e:
        print(f"Error plotting temporal pattern: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_longest_streak(sequence, value):
    """
    Find the longest streak of a specific value in a sequence.
    
    Args:
        sequence: List of values
        value: The value to find streaks of
        
    Returns:
        tuple: (streak_length, streak_position)
    """
    max_streak = 0
    max_pos = 0
    current_streak = 0
    
    for i, val in enumerate(sequence):
        if val == value:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                max_pos = i - (current_streak - 1) + (current_streak // 2)  # Middle of streak
        else:
            current_streak = 0
            
    return max_streak, max_pos

def plot_markov_transitions(order=1, colormap='YlGnBu'):
    """
    Create enhanced visualization of Markov transition probabilities.
    
    Args:
        order: Markov model order (1 or 2)
        colormap: Matplotlib colormap to use
        
    Returns:
        dict: Visualization metadata
    """
    ensure_image_directory()
    if not os.path.exists(REALTIME_FILE):
        print("No real-time data found.")
        return None
    
    try:
        df = pd.read_csv(REALTIME_FILE)
        
        if 'Target' not in df.columns:
            print("Data doesn't have Target column")
            return None
            
        # Create a sequence of outcomes
        sequence = df['Target'].tolist()
        
        if len(sequence) < 10:
            print("Not enough data for Markov analysis (minimum 10 points needed)")
            return None
        
        if order == 1:
            # First-order Markov: current outcome → next outcome
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
            
            # Create enhanced heatmap visualization
            plt.figure(figsize=(12, 10))
            
            # Subplot 1: Transition probabilities heatmap
            plt.subplot(2, 1, 1)
            sns.heatmap(
                transition_probs,
                annot=True,
                fmt='.2f',
                cmap=colormap,
                xticklabels=['Banker', 'Player', 'Tie'],
                yticklabels=['Banker', 'Player', 'Tie'],
                cbar_kws={'label': 'Transition Probability'}
            )
            
            plt.title('First-Order Markov Transition Probabilities', fontsize=16)
            plt.xlabel('Next Outcome', fontsize=14)
            plt.ylabel('Current Outcome', fontsize=14)
            
            # Subplot 2: Visualization of state frequencies
            plt.subplot(2, 1, 2)
            
            # Create bar chart of state frequencies
            state_percentages = [count / sum(state_counts) * 100 for count in state_counts]
            bars = plt.bar(
                ['Banker', 'Player', 'Tie'],
                state_percentages,
                color=['green', 'blue', 'red'],
                alpha=0.7
            )
            
            # Add count and percentage labels
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f"{state_counts[i]} ({state_percentages[i]:.1f}%)",
                    ha='center',
                    fontsize=10
                )
            
            plt.ylabel('Frequency (%)', fontsize=14)
            plt.title('Outcome Distribution', fontsize=16)
            plt.grid(axis='y', alpha=0.3)
            
            # Highlight strongest transitions in text form
            plt.figtext(0.5, 0.01, 
                       "Key Transitions:\n" + 
                       "\n".join([f"After {['Banker', 'Player', 'Tie'][i]}, most likely next is {['Banker', 'Player', 'Tie'][np.argmax(transition_probs[i])]} ({transition_probs[i][np.argmax(transition_probs[i])]:.2f})" 
                                 for i in range(3)]), 
                       ha="center", fontsize=12, 
                       bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])
            plt.savefig('data/images/markov_transitions_enhanced.png', dpi=300)
            plt.show()
            
            return {
                'transition_probs': transition_probs.tolist(),
                'state_counts': state_counts,
                'strongest_transitions': [
                    {
                        'from': ['Banker', 'Player', 'Tie'][i],
                        'to': ['Banker', 'Player', 'Tie'][np.argmax(transition_probs[i])],
                        'probability': float(transition_probs[i][np.argmax(transition_probs[i])])
                    } for i in range(3)
                ],
                'image_path': 'data/images/markov_transitions_enhanced.png'
            }
            
        elif order == 2:
            # Calculate second-order transitions (future implementation)
            print("Second-order Markov visualization not implemented in this version.")
            return None
        
    except Exception as e:
        print(f"Error plotting Markov transitions: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_report_with_visualizations(days=None):
    """
    Generate a comprehensive report with all key visualizations and analysis.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        
    Returns:
        str: Path to generated report
    """
    ensure_image_directory()
    
    print(f"{Fore.CYAN}=== Generating Comprehensive Visualization Report ===")
    start_time = datetime.now()
    
    # Generate all visualizations
    print("Creating model comparison visualizations...")
    try:
        plot_model_comparison(days=days, metric='accuracy')
    except Exception as e:
        print(f"Error creating model comparison: {e}")
    
    print("Creating feature importance visualization...")
    try:
        plot_feature_importance_heatmap()
    except Exception as e:
        print(f"Error creating feature importance heatmap: {e}")
    
    print("Creating confidence distribution visualization...")
    try:
        visualize_confidence_distribution(by_pattern=True)
    except Exception as e:
        print(f"Error creating confidence distribution: {e}")
    
    print("Creating temporal pattern visualization...")
    try:
        plot_temporal_pattern()
    except Exception as e:
        print(f"Error creating temporal pattern: {e}")
    
    print("Creating Markov transition visualization...")
    try:
        plot_markov_transitions()
    except Exception as e:
        print(f"Error creating Markov transitions: {e}")
    
    # Export combined report
    print("Generating HTML report...")
    report_path = export_visualizations_to_html(include_all=True)
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"{Fore.GREEN}Report generation completed in {duration:.1f} seconds.")
    print(f"Report saved to: {report_path}")
    
    return report_path                            