"""
Analysis of meta-features in the stacking ensemble model for Baccarat prediction.
This module provides specialized tools to understand feature importance, correlations
between features and outcomes, and how meta-features contribute to stacking decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from datetime import datetime, timedelta  # Add datetime import
from collections import defaultdict, Counter

from config import LOG_FILE, MODEL_REGISTRY_PATH
from analytics.analytics import ensure_image_directory

# Define consistent model colors from analytics module
MODEL_COLORS = {
    'baccarat_rf': '#4285F4',       # Google Blue
    'markov_1': '#EA4335',          # Google Red
    'markov_2': '#FBBC05',          # Google Yellow
    'xgboost_base': '#34A853',      # Google Green
    'stacking_ensemble': '#9C27B0', # Purple
    'xgb_variant': '#FF9800',       # Orange
    'markov_3': '#00BCD4',          # Cyan
}

# Pattern type colors
PATTERN_COLORS = {
    'no_pattern': '#7F7F7F',    # Gray
    'streak': '#E31A1C',        # Red
    'alternating': '#1F78B4',   # Blue
    'tie': '#33A02C',           # Green
}

def analyze_feature_importance(detailed=False):
    """
    Analyze which meta-features are most important in the stacking ensemble.
    Requires access to the stacking model in registry.
    
    Args:
        detailed: If True, perform more detailed analysis
        
    Returns:
        dict: Feature importance metrics
    """
    ensure_image_directory()
    
    try:
        # Load model registry to get stacking model
        registry_file = os.path.join(MODEL_REGISTRY_PATH, "registry.json")
        if not os.path.exists(registry_file):
            print("Model registry not found.")
            return None
            
        # Try to load registry data
        with open(registry_file, 'r') as f:
            registry_data = json.load(f)
            
        # Check if stacking ensemble exists
        if "stacking_ensemble" not in registry_data.get("model_ids", []):
            print("Stacking ensemble not found in registry.")
            return None
            
        # Try to load stacking model
        stacking_file = os.path.join(MODEL_REGISTRY_PATH, "stacking_ensemble.pkl")
        if not os.path.exists(stacking_file):
            print("Stacking model file not found.")
            return None
            
        with open(stacking_file, 'rb') as f:
            stacking_model = pickle.load(f)
            
        # Check if model has LogisticRegression meta_model with coefficients
        if not hasattr(stacking_model, 'meta_model') or not hasattr(stacking_model.meta_model, 'coef_'):
            print("Stacking model doesn't have accessible coefficients.")
            return None
            
        # Check if model was trained
        if not getattr(stacking_model, 'is_trained', False):
            print("Stacking model hasn't been trained yet.")
            return None
            
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
        
        # If detailed analysis requested, show class-specific coefficients
        if detailed and hasattr(stacking_model.meta_model, 'coef_'):
            # Create class-specific importance visualization
            plt.figure(figsize=(15, 10))
            
            # Get coefficients for each class
            classes = ['Banker', 'Player', 'Tie']
            for i, class_name in enumerate(classes):
                plt.subplot(3, 1, i+1)
                
                # Get class-specific coefficients (top 10)
                class_coef = coef[i, :]
                sorted_indices = np.argsort(np.abs(class_coef))[::-1][:10]
                
                # Extract names and values
                class_features = [feature_names[j] for j in sorted_indices]
                class_values = [class_coef[j] for j in sorted_indices]
                
                # Create bar chart
                bars = plt.barh(range(len(class_features)), class_values, 
                               color=[MODEL_COLORS.get(f.split('_')[0], '#333333') 
                                     if not f.startswith('pattern_') else '#8B0000' 
                                     for f in class_features])
                
                # Colors based on positive/negative
                bar_colors = ['green' if x > 0 else 'red' for x in class_values]
                for j, (bar, color) in enumerate(zip(bars, bar_colors)):
                    bar.set_color(color)
                    bar.set_alpha(0.7)
                    
                plt.yticks(range(len(class_features)), class_features)
                plt.title(f'Features Predicting {class_name} Outcome', fontsize=12)
                plt.xlabel('Coefficient value', fontsize=10)
                plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                plt.grid(axis='x', alpha=0.3)
                
            plt.tight_layout()
            plt.savefig('data/images/class_specific_features.png', dpi=300)
            plt.show()
        
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

def analyze_meta_feature_contribution(days=None):
    """
    Analyze how meta-features contribute to specific outcomes.
    
    Args:
        days: Number of days to analyze, or None for all data
        
    Returns:
        dict: Feature contribution analysis
    """
    # This analysis requires both the stacking model and detailed prediction logs
    ensure_image_directory()
    
    try:
        # First get the stacking model
        registry_file = os.path.join(MODEL_REGISTRY_PATH, "registry.json")
        if not os.path.exists(registry_file) or not os.path.exists(LOG_FILE):
            print("Required files not found.")
            return None
            
        # Load registry data
        with open(registry_file, 'r') as f:
            registry_data = json.load(f)
            
        if "stacking_ensemble" not in registry_data.get("model_ids", []):
            print("Stacking ensemble not found in registry.")
            return None
            
        # Load prediction log
        log_df = pd.read_csv(LOG_FILE)
        
        # Apply time filter if specified
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            
        # Check if pattern_type column exists for analysis
        if 'pattern_type' not in log_df.columns:
            print("Log file doesn't have pattern type information.")
            return None
            
        # Analyze feature effectiveness by pattern type
        pattern_results = {}
        
        for pattern in log_df['pattern_type'].unique():
            if pd.isna(pattern):
                continue
                
            pattern_df = log_df[log_df['pattern_type'] == pattern]
            
            # Skip patterns with too few examples
            if len(pattern_df) < 5:
                continue
                
            # Calculate accuracy for this pattern
            pattern_accuracy = pattern_df['Correct'].mean() * 100
            
            # Analyze model contributions for this pattern
            model_contribs = []
            for _, row in pattern_df.iterrows():
                if isinstance(row['model_contributions'], str):
                    try:
                        contribs = json.loads(row['model_contributions'])
                        model_contribs.append((row['Correct'], contribs))
                    except:
                        pass
                elif isinstance(row['model_contributions'], dict):
                    model_contribs.append((row['Correct'], row['model_contributions']))
            
            # Calculate most effective models for this pattern
            if model_contribs:
                model_effectiveness = defaultdict(lambda: {'correct': 0, 'total': 0})
                
                for correct, contribs in model_contribs:
                    for model, weight in contribs.items():
                        model_effectiveness[model]['total'] += weight
                        if correct:
                            model_effectiveness[model]['correct'] += weight
                
                # Calculate accuracy by model for this pattern
                model_accuracy = {}
                for model, stats in model_effectiveness.items():
                    if stats['total'] > 0:
                        model_accuracy[model] = (stats['correct'] / stats['total']) * 100
                
                pattern_results[pattern] = {
                    'accuracy': pattern_accuracy,
                    'sample_count': len(pattern_df),
                    'model_accuracy': model_accuracy
                }
        
        # Visualize pattern-specific model effectiveness
        if pattern_results:
            plt.figure(figsize=(15, 10))
            
            # Get all models that appear in any pattern
            all_models = set()
            for pattern_data in pattern_results.values():
                all_models.update(pattern_data['model_accuracy'].keys())
            all_models = sorted(all_models)
            
            # Prepare data for grouped bar chart
            patterns = list(pattern_results.keys())
            
            # Set up plot
            bar_width = 0.8 / len(all_models)
            x = np.arange(len(patterns))
            
            # Plot each model's accuracy by pattern
            for i, model in enumerate(all_models):
                accuracies = []
                for pattern in patterns:
                    pattern_data = pattern_results[pattern]
                    acc = pattern_data['model_accuracy'].get(model, 0)
                    accuracies.append(acc)
                
                position = x + (i - len(all_models)/2 + 0.5) * bar_width
                plt.bar(position, accuracies, bar_width, label=model, 
                       color=MODEL_COLORS.get(model, f'C{i}'), alpha=0.7)
            
            # Add overall accuracy line
            overall_acc = log_df['Correct'].mean() * 100
            plt.axhline(y=overall_acc, color='red', linestyle='--', 
                       label=f'Overall: {overall_acc:.1f}%')
            
            plt.xlabel('Pattern Type', fontsize=14)
            plt.ylabel('Accuracy (%)', fontsize=14)
            plt.title('Model Effectiveness by Pattern Type', fontsize=16)
            plt.xticks(x, patterns)
            plt.legend(title='Model', title_fontsize=12)
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)
            
            # Add sample size annotations
            for i, pattern in enumerate(patterns):
                sample_count = pattern_results[pattern]['sample_count']
                plt.text(i, 5, f"n={sample_count}", ha='center', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7))
                
            plt.tight_layout()
            plt.savefig('data/images/feature_effectiveness_by_pattern.png', dpi=300)
            plt.show()
            
        return {
            'pattern_results': pattern_results,
            'total_samples': len(log_df)
        }
        
    except Exception as e:
        print(f"Error analyzing meta-feature contribution: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_key_pattern_features():
    """
    Detect which pattern features are most significant for each outcome.
    
    Returns:
        dict: Pattern feature significance analysis
    """
    ensure_image_directory()
    
    try:
        # Load the stacking model for analysis
        stacking_file = os.path.join(MODEL_REGISTRY_PATH, "stacking_ensemble.pkl")
        if not os.path.exists(stacking_file):
            print("Stacking model file not found.")
            return None
            
        with open(stacking_file, 'rb') as f:
            stacking_model = pickle.load(f)
            
        # Check if model was trained and has coefficients
        if not hasattr(stacking_model, 'meta_model') or not hasattr(stacking_model.meta_model, 'coef_'):
            print("Stacking model doesn't have accessible coefficients.")
            return None
            
        # Extract pattern feature coefficients
        coef = stacking_model.meta_model.coef_
        
        # Get meta-feature dimensions
        if hasattr(stacking_model, 'meta_X') and stacking_model.meta_X:
            n_features = len(stacking_model.meta_X[0])
            
            # The last 4 features are usually pattern features
            pattern_start_idx = n_features - 4
            
            # Extract pattern feature coefficients for each class
            pattern_coef = coef[:, pattern_start_idx:]
            
            # Pattern feature names
            pattern_names = ['no_pattern', 'streak', 'alternating', 'tie']
            
            # Class names
            class_names = ['Banker', 'Player', 'Tie']
            
            # Create heatmap visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                pattern_coef, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                xticklabels=pattern_names,
                yticklabels=class_names
            )
            
            plt.title('Pattern Feature Importance by Outcome Class', fontsize=16)
            plt.tight_layout()
            plt.savefig('data/images/pattern_feature_importance.png', dpi=300)
            plt.show()
            
            # Find key pattern-outcome relationships
            key_patterns = []
            
            for i, class_name in enumerate(class_names):
                # Get pattern with highest positive coefficient (supporting this outcome)
                max_idx = np.argmax(pattern_coef[i])
                max_coef = pattern_coef[i, max_idx]
                
                # Get pattern with most negative coefficient (opposing this outcome)
                min_idx = np.argmin(pattern_coef[i])
                min_coef = pattern_coef[i, min_idx]
                
                key_patterns.append({
                    'outcome': class_name,
                    'supporting_pattern': {
                        'pattern': pattern_names[max_idx],
                        'coefficient': float(max_coef)
                    },
                    'opposing_pattern': {
                        'pattern': pattern_names[min_idx],
                        'coefficient': float(min_coef)
                    }
                })
            
            return {
                'pattern_coefficients': pattern_coef.tolist(),
                'pattern_names': pattern_names,
                'class_names': class_names,
                'key_patterns': key_patterns
            }
        else:
            print("Stacking model doesn't have meta-features.")
            return None
        
    except Exception as e:
        print(f"Error detecting key pattern features: {e}")
        import traceback
        traceback.print_exc()
        return None
def pattern_feature_impact(days=None, top_models=3):
    """
    Analyze the impact of pattern features on prediction accuracy.
    
    Args:
        days: Number of recent days to analyze (None for all data)
        top_models: Number of top models to include in the analysis
        
    Returns:
        dict: Pattern impact analysis results
    """
    # Redirect to the dedicated function in pattern_feature_analysis module
    from analytics.pattern_feature_analysis import pattern_feature_impact as pfi
    return pfi(days=days, top_models=top_models)

def track_feature_stability(timewindow=None):
    """
    Track the stability of feature importance over time.
    
    Args:
        timewindow: Number of days or predictions to analyze in each window
        
    Returns:
        dict: Feature stability analysis
    """
    # This requires stacking ensemble with history of coefficients
    pass

def track_feature_stability(timewindow=None):
    """
    Track the stability of feature importance over time.
    
    Args:
        timewindow: Number of days or predictions to analyze in each window
        
    Returns:
        dict: Feature stability analysis
    """
    ensure_image_directory()
    
    try:
        # Load prediction log for temporal analysis
        if not os.path.exists(LOG_FILE):
            print("No prediction log found.")
            return None
            
        log_df = pd.read_csv(LOG_FILE)
        
        # Check if prediction log has timestamp
        if 'Timestamp' not in log_df.columns:
            print("Log file doesn't have timestamp information.")
            return None
            
        # Convert timestamp to datetime
        log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
        
        # Check for meta-feature importance data in logs
        if not hasattr(log_df, 'meta_features') and 'meta_features' not in log_df.columns:
            print("Log file doesn't have meta-feature information.")
            return None
            
        # For each time window, calculate feature importance
        if timewindow is None:
            # Default to 4 equal windows
            windows = 4
            window_size = len(log_df) // windows
            
            # Create time windows
            window_dfs = []
            for i in range(windows):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if i < windows - 1 else len(log_df)
                window_dfs.append(log_df.iloc[start_idx:end_idx])
        else:
            # Use specified time window in days
            window_dfs = []
            
            # Sort by timestamp
            log_df = log_df.sort_values('Timestamp')
            
            # Get date range
            start_date = log_df['Timestamp'].min()
            end_date = log_df['Timestamp'].max()
            
            # Create windows of specified days
            current_date = start_date
            while current_date < end_date:
                next_date = current_date + pd.Timedelta(days=timewindow)
                window_df = log_df[(log_df['Timestamp'] >= current_date) & 
                                  (log_df['Timestamp'] < next_date)]
                
                if len(window_df) > 10:  # Only include windows with sufficient data
                    window_dfs.append(window_df)
                    
                current_date = next_date
        
        # Calculate feature importance for each window
        window_importance = []
        
        for i, window_df in enumerate(window_dfs):
            # Load the registry for this time period's end
            end_time = window_df['Timestamp'].max()
            
            # Use main feature importance analysis for this window
            result = analyze_feature_importance()
            
            if result:
                window_importance.append({
                    'window': i + 1,
                    'start_time': window_df['Timestamp'].min(),
                    'end_time': end_time,
                    'sample_count': len(window_df),
                    'feature_importance': result['feature_importance'][:10]  # Top 10 features
                })
        
        # Analyze stability of features across windows
        if len(window_importance) > 1:
            # Track top features across windows
            feature_ranks = defaultdict(list)
            
            # For each window, record feature ranks
            for window in window_importance:
                for rank, (feature, _) in enumerate(window['feature_importance']):
                    feature_ranks[feature].append(rank + 1)  # Convert to 1-based rank
            
            # Calculate stability metrics
            stability_metrics = {}
            for feature, ranks in feature_ranks.items():
                # Only consider features that appear in at least half the windows
                if len(ranks) >= len(window_importance) / 2:
                    # Calculate rank stability (lower variation is more stable)
                    avg_rank = np.mean(ranks)
                    rank_std = np.std(ranks)
                    
                    stability_metrics[feature] = {
                        'avg_rank': avg_rank,
                        'rank_std': rank_std,
                        'appearances': len(ranks),
                        'stability_score': 1.0 / (1.0 + rank_std)  # Higher for more stable features
                    }
            
            # Visualize feature stability
            plt.figure(figsize=(14, 8))
            
            # Sort features by stability score
            sorted_features = sorted(
                stability_metrics.items(),
                key=lambda x: x[1]['stability_score'],
                reverse=True
            )
            
            feature_names = [f[0] for f in sorted_features[:15]]  # Top 15 most stable
            stability_scores = [f[1]['stability_score'] for f in sorted_features[:15]]
            avg_ranks = [f[1]['avg_rank'] for f in sorted_features[:15]]
            
            # Create bar chart with dual axis
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Stability score bars
            bars = ax1.bar(
                range(len(feature_names)), 
                stability_scores,
                color='skyblue',
                alpha=0.7,
                label='Stability Score'
            )
            
            ax1.set_xlabel('Feature', fontsize=12)
            ax1.set_ylabel('Stability Score (higher = more stable)', fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Add average rank line
            ax2 = ax1.twinx()
            ax2.plot(
                range(len(feature_names)),
                avg_ranks,
                'r-',
                linewidth=2,
                marker='o',
                label='Avg. Rank'
            )
            
            ax2.set_ylabel('Average Rank (lower = more important)', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Determine feature types and colors for tick labels
            tick_colors = []
            for name in feature_names:
                if 'pattern_' in name:
                    tick_colors.append('#8B0000')  # Dark red for pattern features
                else:
                    model_name = name.split('_')[0]
                    tick_colors.append(MODEL_COLORS.get(model_name, '#333333'))
            
            # Set x-ticks with colored feature names
            plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
            
            # Color x-tick labels based on feature type
            for i, tick in enumerate(ax1.get_xticklabels()):
                tick.set_color(tick_colors[i])
            
            plt.title('Feature Importance Stability Across Time Windows', fontsize=16)
            
            # Add legend
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper right')
            
            plt.tight_layout()
            plt.savefig('data/images/feature_stability.png', dpi=300)
            plt.show()
            
            return {
                'window_importance': window_importance,
                'stability_metrics': stability_metrics,
                'window_count': len(window_importance)
            }
        else:
            print("Insufficient data for stability analysis.")
            return {'window_importance': window_importance}
        
    except Exception as e:
        print(f"Error tracking feature stability: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_feature_correlation():
    """
    Analyze correlations between meta-features and prediction outcomes.
    
    Returns:
        dict: Feature correlation analysis
    """
    ensure_image_directory()
    
    try:
        # Get prediction logs for actual outcomes
        if not os.path.exists(LOG_FILE):
            print("No prediction log found.")
            return None
            
        log_df = pd.read_csv(LOG_FILE)
        
        # Check necessary columns
        if 'model_contributions' not in log_df.columns:
            print("Log file doesn't have model contribution data.")
            return None
            
        # Get meta-feature data from logs
        meta_features = []
        outcomes = []
        is_correct = []
        
        for _, row in log_df.iterrows():
            actual = row['Actual']
            predicted = row['Predicted']
            correct = 1 if predicted == actual else 0
            
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
                
            # Extract pattern type if available
            pattern_feature = [0, 0, 0, 0]  # [no_pattern, streak, alternating, tie]
            if 'pattern_type' in row:
                pattern_type = row['pattern_type']
                if pattern_type == 'streak':
                    pattern_feature = [0, 1, 0, 0]
                elif pattern_type == 'alternating':
                    pattern_feature = [0, 0, 1, 0]
                elif pattern_type == 'tie':
                    pattern_feature = [0, 0, 0, 1]
                    
            # Get model contributions and create meta-feature vector
            feature_vector = []
            for model in sorted(contribs.keys()):
                # Contributions likely represent confidence in each outcome
                # We'll use the raw contribution as a proxy for outcome probabilities
                contrib = contribs[model]
                feature_vector.append(contrib)
                
            # Add pattern features
            feature_vector.extend(pattern_feature)
            
            # Store data
            meta_features.append(feature_vector)
            outcomes.append(actual)
            is_correct.append(correct)
        
        if len(meta_features) < 10:
            print("Insufficient data for correlation analysis.")
            return None
            
        # Convert to numpy array
        meta_features = np.array(meta_features)
        outcomes = np.array(outcomes)
        is_correct = np.array(is_correct)
        
        # Calculate correlation between features and outcomes
        correlations = []
        
        for i in range(meta_features.shape[1]):
            feature_values = meta_features[:, i]
            
            # Calculate point-biserial correlation with correctness
            # (special case of Pearson correlation when one variable is binary)
            corr_correct = np.corrcoef(feature_values, is_correct)[0, 1]
            
            # For each outcome class, calculate correlation
            corr_classes = []
            for outcome in range(3):  # 0, 1, 2 for Banker, Player, Tie
                # Create binary indicator for this outcome
                outcome_indicator = (outcomes == outcome).astype(int)
                
                # Calculate correlation
                corr = np.corrcoef(feature_values, outcome_indicator)[0, 1]
                corr_classes.append(corr)
                
            correlations.append({
                'feature_idx': i,
                'corr_correct': corr_correct,
                'corr_banker': corr_classes[0],
                'corr_player': corr_classes[1],
                'corr_tie': corr_classes[2]
            })
        
        # Generate feature names
        # We don't know exact mapping without model registry data
        # So we'll use generic names based on position
        # Typically: model contribution features, then pattern features
        feature_names = []
        for i in range(meta_features.shape[1] - 4):  # Assuming last 4 are pattern features
            feature_names.append(f"Model_{i+1}")
            
        # Add pattern feature names
        pattern_names = ['no_pattern', 'streak', 'alternating', 'tie']
        for name in pattern_names:
            feature_names.append(f"pattern_{name}")
            
        # Adjust feature names length to match meta-features width
        if len(feature_names) > meta_features.shape[1]:
            feature_names = feature_names[:meta_features.shape[1]]
        elif len(feature_names) < meta_features.shape[1]:
            for i in range(len(feature_names), meta_features.shape[1]):
                feature_names.append(f"Feature_{i+1}")
        
        # Update correlations with feature names
        for i, corr in enumerate(correlations):
            corr['feature_name'] = feature_names[i]
            
        # Visualize correlations
        plt.figure(figsize=(14, 10))
        
        # Create a correlation matrix for heatmap
        corr_matrix = np.zeros((len(feature_names), 4))
        
        for i, corr in enumerate(correlations):
            corr_matrix[i, 0] = corr['corr_banker']
            corr_matrix[i, 1] = corr['corr_player']
            corr_matrix[i, 2] = corr['corr_tie']
            corr_matrix[i, 3] = corr['corr_correct']
            
        # Create heatmap
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            xticklabels=['Banker', 'Player', 'Tie', 'Correct'],
            yticklabels=feature_names
        )
        
        plt.title('Feature Correlation with Outcomes', fontsize=16)
        plt.tight_layout()
        plt.savefig('data/images/feature_correlation.png', dpi=300)
        plt.show()
        
        # Create bar chart of absolute correlation with correctness
        plt.figure(figsize=(14, 8))
        
        # Sort by absolute correlation with correctness
        sorted_by_corr = sorted(
            correlations, 
            key=lambda x: abs(x['corr_correct']),
            reverse=True
        )
        
        feature_names = [corr['feature_name'] for corr in sorted_by_corr]
        corr_values = [corr['corr_correct'] for corr in sorted_by_corr]
        
        # Create bar chart
        bars = plt.bar(
            range(len(feature_names)), 
            corr_values,
            color=['green' if v > 0 else 'red' for v in corr_values]
        )
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02 if bar.get_height() >= 0 else bar.get_height() - 0.08,
                f"{corr_values[i]:.2f}",
                ha='center',
                va='bottom' if bar.get_height() >= 0 else 'top',
                fontsize=9
            )
            
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Correlation with Correctness', fontsize=12)
        plt.title('Feature Correlation with Prediction Correctness', fontsize=16)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/images/feature_correlation_correctness.png', dpi=300)
        plt.show()
        
        return {
            'correlations': correlations,
            'feature_names': feature_names,
            'sample_count': len(meta_features)
        }
        
    except Exception as e:
        print(f"Error analyzing feature correlation: {e}")
        import traceback
        traceback.print_exc()
        return None

def feature_ablation_study():
    """
    Perform feature ablation study to measure impact of removing features.
    
    Returns:
        dict: Feature ablation analysis
    """
    # This would require the ability to modify and retrain the meta-learner
    # Implementation left as a placeholder for future development
    print("Feature ablation study is not implemented in this version.")
    return None

def analyze_pattern_feature_contribution(days=None):
    """
    Analyze how pattern-specific features contribute to predictions.
    
    Args:
        days: Number of days to analyze, or None for all data
        
    Returns:
        dict: Pattern feature contribution analysis
    """
    ensure_image_directory()
    
    try:
        # Load prediction log
        if not os.path.exists(LOG_FILE):
            print("No prediction log found.")
            return None
            
        log_df = pd.read_csv(LOG_FILE)
        
        # Check necessary columns
        if 'pattern_type' not in log_df.columns:
            print("Log file doesn't have pattern type information.")
            return None
            
        # Apply time filter if specified
        if days and 'Timestamp' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            log_df = log_df[log_df['Timestamp'] >= cutoff_date]
            
        # Calculate accuracy by pattern type
        pattern_accuracy = {}
        pattern_counts = {}
        
        # Get overall accuracy
        overall_accuracy = log_df['Correct'].mean() * 100
        
        for pattern in log_df['pattern_type'].unique():
            if pd.isna(pattern):
                continue
                
            pattern_df = log_df[log_df['pattern_type'] == pattern]
            
            # Skip patterns with too few examples
            if len(pattern_df) < 5:
                continue
                
            # Calculate accuracy for this pattern
            accuracy = pattern_df['Correct'].mean() * 100
            
            pattern_accuracy[pattern] = accuracy
            pattern_counts[pattern] = len(pattern_df)
        
        # Visualize pattern accuracy
        plt.figure(figsize=(12, 8))
        
        # Sort patterns by accuracy
        sorted_patterns = sorted(
            pattern_accuracy.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        patterns = [p[0] for p in sorted_patterns]
        accuracies = [p[1] for p in sorted_patterns]
        counts = [pattern_counts[p] for p in patterns]
        
        # Create bar chart
        bars = plt.bar(
            range(len(patterns)), 
            accuracies,
            color=[PATTERN_COLORS.get(p, '#333333') for p in patterns]
        )
        
        # Add sample count annotations
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"n={counts[i]}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            
        # Add overall accuracy reference line
        plt.axhline(y=overall_accuracy, color='red', linestyle='--', 
                   label=f'Overall: {overall_accuracy:.1f}%')
        
        plt.xlabel('Pattern Type', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Prediction Accuracy by Pattern Type', fontsize=16)
        plt.xticks(range(len(patterns)), patterns)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('data/images/pattern_accuracy.png', dpi=300)
        plt.show()
        
        # Additional analysis: pattern transitions
        if 'Prev_1' in log_df.columns and 'Target' in log_df.columns:
            # For each pattern, calculate transition to next pattern
            pattern_transitions = defaultdict(Counter)
            
            # Determine pattern sequence
            pattern_sequence = []
            
            for i in range(len(log_df) - 1):
                if 'pattern_type' in log_df.columns and not pd.isna(log_df.iloc[i]['pattern_type']):
                    current_pattern = log_df.iloc[i]['pattern_type']
                    next_pattern = log_df.iloc[i+1]['pattern_type'] if not pd.isna(log_df.iloc[i+1]['pattern_type']) else 'unknown'
                    
                    pattern_sequence.append((current_pattern, next_pattern))
                    pattern_transitions[current_pattern][next_pattern] += 1
            
            # Calculate transition probabilities
            transition_probs = {}
            for pattern, transitions in pattern_transitions.items():
                total = sum(transitions.values())
                if total > 0:
                    transition_probs[pattern] = {next_p: count/total for next_p, count in transitions.items()}
            
            # Visualize pattern transitions
            if transition_probs:
                # Get unique patterns
                unique_patterns = sorted(set(p for p in log_df['pattern_type'].unique() if not pd.isna(p)))
                
                # Create transition matrix
                transition_matrix = np.zeros((len(unique_patterns), len(unique_patterns)))
                
                for i, pattern1 in enumerate(unique_patterns):
                    for j, pattern2 in enumerate(unique_patterns):
                        if pattern1 in transition_probs and pattern2 in transition_probs[pattern1]:
                            transition_matrix[i, j] = transition_probs[pattern1][pattern2]
                
                # Visualize as heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    transition_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='YlGnBu',
                    xticklabels=unique_patterns,
                    yticklabels=unique_patterns
                )
                
                plt.title('Pattern Transition Probabilities', fontsize=16)
                plt.xlabel('Next Pattern', fontsize=14)
                plt.ylabel('Current Pattern', fontsize=14)
                
                plt.tight_layout()
                plt.savefig('data/images/pattern_transitions.png', dpi=300)
                plt.show()
        
        return {
            'pattern_accuracy': pattern_accuracy,
            'pattern_counts': pattern_counts,
            'overall_accuracy': overall_accuracy,
            'sample_count': len(log_df)
        }
        
    except Exception as e:
        print(f"Error analyzing pattern feature contribution: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_feature_report():
    """
    Generate a comprehensive report on meta-feature importance and behavior.
    
    Returns:
        dict: Comprehensive feature analysis report
    """
    print("=== Meta-Feature Analysis Report ===")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run all main analyses
    importance_result = analyze_feature_importance()
    correlation_result = analyze_feature_correlation()
    pattern_result = analyze_pattern_feature_contribution()
    key_patterns = detect_key_pattern_features()
    
    # Combine results into comprehensive report
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if importance_result:
        # Print importance summary
        print("\n=== Feature Importance Summary ===")
        print(f"{'Feature':<30} {'Importance':<10}")
        print("-" * 40)
        
        for feature, importance in importance_result['feature_importance'][:10]:
            print(f"{feature:<30} {importance:.4f}")
            
        # Group by type
        print("\nImportance by Feature Type:")
        
        model_importance = importance_result['model_importance']
        pattern_importance = importance_result['pattern_importance']
        
        total_importance = sum(model_importance.values()) + pattern_importance
        
        for model, importance in sorted(model_importance.items(), key=lambda x: x[1], reverse=True):
            percentage = (importance / total_importance) * 100
            print(f"{model:<20} {importance:.4f} ({percentage:.1f}%)")
            
        pattern_percentage = (pattern_importance / total_importance) * 100
        print(f"{'Pattern Features':<20} {pattern_importance:.4f} ({pattern_percentage:.1f}%)")
        
        report['importance'] = importance_result
    
    if correlation_result:
        # Print correlation summary
        print("\n=== Feature-Outcome Correlation Summary ===")
        
        # Sort by absolute correlation with correctness
        sorted_by_corr = sorted(
            correlation_result['correlations'], 
            key=lambda x: abs(x['corr_correct']),
            reverse=True
        )
        
        print(f"{'Feature':<30} {'Corr w/Correct':<15} {'Banker':<10} {'Player':<10} {'Tie':<10}")
        print("-" * 75)
        
        for corr in sorted_by_corr[:10]:
            print(f"{corr['feature_name']:<30} {corr['corr_correct']:>+.2f}{'':^7} {corr['corr_banker']:>+.2f}{'':^2} {corr['corr_player']:>+.2f}{'':^2} {corr['corr_tie']:>+.2f}")
            
        report['correlation'] = correlation_result
    
    if pattern_result:
        # Print pattern effectiveness
        print("\n=== Pattern Type Effectiveness ===")
        print(f"Overall accuracy: {pattern_result['overall_accuracy']:.2f}%")
        print(f"{'Pattern Type':<20} {'Accuracy':<10} {'Count':<10} {'vs Overall':<10}")
        print("-" * 50)
        
        sorted_patterns = sorted(
            [(p, acc) for p, acc in pattern_result['pattern_accuracy'].items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for pattern, accuracy in sorted_patterns:
            count = pattern_result['pattern_counts'][pattern]
            diff = accuracy - pattern_result['overall_accuracy']
            diff_str = f"{'+' if diff > 0 else ''}{diff:.2f}%"
            print(f"{pattern:<20} {accuracy:.2f}%{'':^3} {count:<10} {diff_str:<10}")
            
        report['pattern'] = pattern_result
    
    if key_patterns:
        # Print key pattern-outcome relationships
        print("\n=== Key Pattern-Outcome Relationships ===")
        
        for rel in key_patterns['key_patterns']:
            outcome = rel['outcome']
            support = rel['supporting_pattern']
            oppose = rel['opposing_pattern']
            
            print(f"{outcome} outcome:")
            print(f"  Supported by: {support['pattern']} (coef: {support['coefficient']:.2f})")
            print(f"  Opposed by: {oppose['pattern']} (coef: {oppose['coefficient']:.2f})")
            
        report['key_patterns'] = key_patterns
    
    print("\n=== End of Report ===")
    
    return report