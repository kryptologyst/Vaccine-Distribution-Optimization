#!/usr/bin/env python3
"""Main training script for vaccine distribution optimization.

This script demonstrates the complete pipeline from data generation
to model training and evaluation.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from vaccine_distribution.data import VaccineDataGenerator, VaccineDataProcessor
from vaccine_distribution.models import (
    BaselineRegressor,
    GradientBoostingRegressor,
    NeuralNetworkRegressor,
    VaccineAllocationOptimizer,
)
from vaccine_distribution.evaluation import VaccineEvaluator
from vaccine_distribution.visualization import VaccineVisualizer, MapVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducible results.
    
    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    
    # Set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Set PyTorch seeds for reproducibility")
    except ImportError:
        logger.info("PyTorch not available, skipping PyTorch seed setting")
    
    # Set scikit-learn seeds
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
        logger.info("Set scikit-learn random state")
    except ImportError:
        logger.info("scikit-learn not available, skipping sklearn seed setting")


def generate_and_process_data(
    n_regions: int = 1000,
    seed: int = 42,
    save_data: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate and process vaccine distribution data.
    
    Args:
        n_regions: Number of regions to generate.
        seed: Random seed for reproducibility.
        save_data: Whether to save processed data.
        
    Returns:
        Tuple of (dataframe, features, targets).
    """
    logger.info(f"Generating data for {n_regions} regions")
    
    # Generate synthetic data
    generator = VaccineDataGenerator(seed=seed)
    regions = generator.generate_regions(n_regions=n_regions)
    
    # Process data
    processor = VaccineDataProcessor()
    df = processor.regions_to_dataframe(regions)
    
    # Prepare features and targets
    X, y = processor.prepare_features(df, normalize=True, add_interactions=True)
    
    # Save data if requested
    if save_data:
        data_dir = Path("data/processed")
        data_dir.mkdir(parents=True, exist_ok=True)
        processor.save_data(df, data_dir / "vaccine_data.csv")
        logger.info(f"Saved processed data to {data_dir}")
    
    logger.info(f"Generated dataset with {len(df)} regions and {X.shape[1]} features")
    return df, X, y


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> List[Any]:
    """Train multiple models for vaccine allocation prediction.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        
    Returns:
        List of trained models.
    """
    logger.info("Training multiple models")
    
    models = []
    
    # Baseline models
    baseline_models = [
        BaselineRegressor("linear"),
        BaselineRegressor("ridge"),
        BaselineRegressor("random_forest"),
    ]
    
    for model in baseline_models:
        logger.info(f"Training {model.name}")
        model.fit(X_train, y_train)
        models.append(model)
    
    # Gradient boosting models
    boosting_models = [
        GradientBoostingRegressor("xgboost"),
        GradientBoostingRegressor("lightgbm"),
    ]
    
    for model in boosting_models:
        logger.info(f"Training {model.name}")
        model.fit(X_train, y_train)
        models.append(model)
    
    # Neural network model
    logger.info("Training neural network")
    nn_model = NeuralNetworkRegressor(
        hidden_sizes=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=100,
    )
    nn_model.fit(X_train, y_train)
    models.append(nn_model)
    
    logger.info(f"Trained {len(models)} models successfully")
    return models


def evaluate_models(
    models: List[Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """Evaluate all trained models and create visualizations.
    
    Args:
        models: List of trained models.
        X_test: Test features.
        y_test: Test targets.
        X_train: Training features.
        y_train: Training targets.
    """
    logger.info("Evaluating models")
    
    # Create evaluator
    evaluator = VaccineEvaluator()
    
    # Evaluate all models
    evaluator.evaluate_multiple_models(
        models, X_test, y_test, X_train, y_train
    )
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard()
    print("\n" + "="*60)
    print("MODEL LEADERBOARD")
    print("="*60)
    print(leaderboard.to_string(index=False))
    
    # Get best model
    best_model = evaluator.get_best_model()
    if best_model:
        print(f"\nBest Model: {best_model.model_name}")
        print(f"RMSE: {best_model.rmse:.2f}")
        print(f"MAE: {best_model.mae:.2f}")
        print(f"R²: {best_model.r2:.3f}")
    
    # Save results
    results_dir = Path("assets/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(results_dir / "evaluation_results.json")
    leaderboard.to_csv(results_dir / "leaderboard.csv", index=False)
    
    logger.info("Model evaluation completed")


def create_visualizations(
    models: List[Any],
    df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Create comprehensive visualizations.
    
    Args:
        models: List of trained models.
        df: DataFrame with region data.
        X_test: Test features.
        y_test: Test targets.
    """
    logger.info("Creating visualizations")
    
    assets_dir = Path("assets/visualizations")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = VaccineVisualizer()
    map_viz = MapVisualizer()
    
    # Data distribution analysis
    visualizer.plot_distribution_analysis(df, assets_dir / "data_distribution.png")
    visualizer.plot_correlation_heatmap(df, assets_dir / "correlation_heatmap.png")
    
    # Model comparison
    evaluator = VaccineEvaluator()
    evaluator.evaluate_multiple_models(models, X_test, y_test)
    results = [
        {
            'model_name': m.model_name,
            'rmse': evaluator.results[i].rmse,
            'mae': evaluator.results[i].mae,
            'r2': evaluator.results[i].r2,
        }
        for i, m in enumerate(models)
    ]
    
    visualizer.plot_model_comparison(results, "rmse", assets_dir / "model_comparison.png")
    
    # Best model predictions
    best_model = evaluator.get_best_model()
    if best_model:
        best_model_obj = models[evaluator.results.index(best_model)]
        y_pred = best_model_obj.predict(X_test)
        
        visualizer.plot_prediction_comparison(
            y_test, y_pred, best_model.model_name, 
            assets_dir / "best_model_predictions.png"
        )
        
        # Feature importance if available
        if hasattr(best_model_obj, 'feature_importance_') and best_model_obj.feature_importance_ is not None:
            feature_names = [
                'population', 'infection_rate', 'elderly_ratio', 
                'logistics_score', 'cold_chain_capacity',
                'risk_exposure', 'elderly_pop', 'effective_capacity'
            ]
            visualizer.plot_feature_importance(
                feature_names, best_model_obj.feature_importance_,
                best_model.model_name, assets_dir / "feature_importance.png"
            )
    
    # Interactive maps
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Add dummy allocation column for mapping
        df['vaccine_allocation'] = df['vaccine_demand']  # Use demand as proxy
        
        allocation_map = map_viz.create_vaccine_allocation_map(
            df, save_path=assets_dir / "allocation_map.html"
        )
        
        priority_map = map_viz.create_priority_map(
            df, save_path=assets_dir / "priority_map.html"
        )
        
        # Interactive dashboard
        map_viz.create_interactive_dashboard(
            df, save_path=assets_dir / "dashboard.html"
        )
    
    logger.info(f"Visualizations saved to {assets_dir}")


def demonstrate_optimization(df: pd.DataFrame) -> None:
    """Demonstrate vaccine allocation optimization.
    
    Args:
        df: DataFrame with region data.
    """
    logger.info("Demonstrating vaccine allocation optimization")
    
    # Convert DataFrame to region dictionaries
    regions = []
    for _, row in df.iterrows():
        regions.append({
            'region_id': row['region_id'],
            'population': row['population'],
            'infection_rate': row['infection_rate'],
            'elderly_ratio': row['elderly_ratio'],
            'logistics_score': row['logistics_score'],
            'cold_chain_capacity': row['cold_chain_capacity'],
            'vaccine_demand': row['vaccine_demand'],
            'priority_score': row['priority_score'],
        })
    
    # Create optimizer
    optimizer = VaccineAllocationOptimizer()
    
    # Optimize allocation
    total_vaccines = int(df['vaccine_demand'].sum() * 0.8)  # 80% of total demand
    optimal_allocations = optimizer.optimize_allocation(
        regions, total_vaccines,
        constraints={'max_per_region': df['cold_chain_capacity'].max()}
    )
    
    # Display results
    print("\n" + "="*60)
    print("VACCINE ALLOCATION OPTIMIZATION")
    print("="*60)
    print(f"Total vaccines available: {total_vaccines:,}")
    print(f"Total demand: {df['vaccine_demand'].sum():,.0f}")
    print(f"Coverage: {total_vaccines / df['vaccine_demand'].sum() * 100:.1f}%")
    
    # Show top 10 allocations
    sorted_allocations = sorted(
        optimal_allocations.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    print("\nTop 10 Allocations:")
    for region_id, allocation in sorted_allocations:
        region_data = df[df['region_id'] == region_id].iloc[0]
        print(f"{region_id}: {allocation:,.0f} vaccines "
              f"(Priority: {region_data['priority_score']:.3f}, "
              f"Demand: {region_data['vaccine_demand']:,.0f})")
    
    # Save optimization results
    results_dir = Path("assets/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    optimization_df = pd.DataFrame([
        {'region_id': rid, 'optimal_allocation': alloc}
        for rid, alloc in optimal_allocations.items()
    ])
    optimization_df.to_csv(results_dir / "optimal_allocations.csv", index=False)
    
    logger.info("Optimization demonstration completed")


def main():
    """Main function to run the complete vaccine distribution pipeline."""
    logger.info("Starting Vaccine Distribution Optimization Pipeline")
    
    # Set deterministic seeds
    set_deterministic_seeds(seed=42)
    
    # Generate and process data
    df, X, y = generate_and_process_data(n_regions=1000, seed=42)
    
    # Split data
    processor = VaccineDataProcessor()
    X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2)
    
    # Train models
    models = train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate models
    evaluate_models(models, X_test, y_test, X_train, y_train)
    
    # Create visualizations
    create_visualizations(models, df, X_test, y_test)
    
    # Demonstrate optimization
    demonstrate_optimization(df)
    
    logger.info("Vaccine Distribution Optimization Pipeline completed successfully!")
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Check the following directories for outputs:")
    print("- assets/visualizations/: Charts and maps")
    print("- assets/results/: Model results and leaderboard")
    print("- data/processed/: Processed dataset")


if __name__ == "__main__":
    main()
