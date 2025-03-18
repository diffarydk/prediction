# analytics/__init__.py - Update to include new modules
"""
Enhanced analytics package for Baccarat Prediction System.
Contains advanced analysis and visualization tools.
"""

from .analytics import (
    analyze_prediction_history,
    plot_accuracy_over_time,
    plot_outcome_distribution,
    analyze_confidence_vs_accuracy,
    analyze_patterns,
    analyze_streak_behavior,
    plot_temporal_pattern,
    analyze_markov_transitions,
    plot_prediction_distribution,
    generate_full_report,
    run_analytics_menu,
    analyze_model_performance,
    analyze_pattern_effectiveness
)

# Import from model_analysis module
from .model_analysis import (
    analyze_stacking_weights,
    model_contribution_analysis,
    compare_base_models,
    track_model_drift,
    evaluate_ensemble_performance,
    analyze_ensemble_weights,
    profile_memory_usage
)

# Import from feature_importance module
from .feature_importance import (
    analyze_feature_importance,
    analyze_meta_feature_contribution,
    detect_key_pattern_features,
    track_feature_stability,
    analyze_feature_correlation,
    analyze_pattern_feature_contribution,
    generate_feature_report
)

__all__ = [
    # From analytics.py
    'analyze_prediction_history',
    'plot_accuracy_over_time',
    'plot_outcome_distribution',
    'analyze_confidence_vs_accuracy',
    'analyze_patterns',
    'analyze_streak_behavior',
    'plot_temporal_pattern',
    'analyze_markov_transitions',
    'plot_prediction_distribution',
    'generate_full_report',
    'run_analytics_menu',
    'analyze_model_performance',
    'analyze_pattern_effectiveness',
    
    # From model_analysis.py
    'analyze_stacking_weights',
    'model_contribution_analysis',
    'compare_base_models',
    'track_model_drift',
    'evaluate_ensemble_performance',
    'analyze_ensemble_weights',
    'profile_memory_usage',
    
    # From feature_importance.py
    'analyze_feature_importance',
    'pattern_feature_impact',
    'temporal_feature_analysis',
    'feature_correlation_matrix',
    'analyze_meta_feature_contribution',
    'detect_key_pattern_features',
    'track_feature_stability',
    'analyze_feature_correlation',
    'analyze_pattern_feature_contribution',
    'generate_feature_report',
    
    # From visualization.py
    'plot_prediction_history',
    'create_model_performance_dashboard',
    'generate_feature_importance_chart',
    'visualize_stacking_ensemble',
    'plot_betting_performance',
    'create_interactive_analysis',
    'plot_model_comparison',
    'plot_feature_importance_heatmap',
    'visualize_confidence_distribution',
    'create_interactive_dashboard',
    'export_visualizations_to_html',
    'plot_temporal_pattern',
    'plot_markov_transitions',
    'generate_report_with_visualizations'
]

# Set module version
__version__ = '2.0.0'