"""Evaluation metrics and model comparison for vaccine distribution optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    
    model_name: str
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    smape: float
    mase: float
    feature_importance: Optional[np.ndarray] = None


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics for vaccine allocation models."""
    
    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        pass
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_baseline: Optional[np.ndarray] = None,
        model_name: str = "model",
    ) -> ModelMetrics:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            y_baseline: Baseline predictions for MASE calculation.
            model_name: Name of the model.
            
        Returns:
            ModelMetrics object with all calculated metrics.
        """
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Percentage error metrics
        mape = self._calculate_mape(y_true, y_pred)
        smape = self._calculate_smape(y_true, y_pred)
        
        # MASE (Mean Absolute Scaled Error)
        if y_baseline is not None:
            mase = self._calculate_mase(y_true, y_pred, y_baseline)
        else:
            mase = np.nan
        
        metrics = ModelMetrics(
            model_name=model_name,
            mse=mse,
            rmse=rmse,
            mae=mae,
            mape=mape,
            r2=r2,
            smape=smape,
            mase=mase,
        )
        
        logger.info(f"Calculated metrics for {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            MAPE value.
        """
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.nan
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            SMAPE value.
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if not np.any(mask):
            return np.nan
        
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def _calculate_mase(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_baseline: np.ndarray
    ) -> float:
        """Calculate Mean Absolute Scaled Error.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            y_baseline: Baseline predictions.
            
        Returns:
            MASE value.
        """
        mae_model = mean_absolute_error(y_true, y_pred)
        mae_baseline = mean_absolute_error(y_true, y_baseline)
        
        if mae_baseline == 0:
            return np.nan
        
        return mae_model / mae_baseline
    
    def calculate_spatial_metrics(
        self,
        df: pd.DataFrame,
        y_true_col: str,
        y_pred_col: str,
        region_col: str = "region_id",
    ) -> Dict[str, float]:
        """Calculate spatial-specific evaluation metrics.
        
        Args:
            df: DataFrame with predictions and region information.
            y_true_col: Column name for true values.
            y_pred_col: Column name for predicted values.
            region_col: Column name for region identifier.
            
        Returns:
            Dictionary of spatial metrics.
        """
        metrics = {}
        
        # Regional RMSE
        regional_rmse = []
        for region in df[region_col].unique():
            region_data = df[df[region_col] == region]
            if len(region_data) > 0:
                rmse = np.sqrt(mean_squared_error(
                    region_data[y_true_col], 
                    region_data[y_pred_col]
                ))
                regional_rmse.append(rmse)
        
        metrics['mean_regional_rmse'] = np.mean(regional_rmse)
        metrics['std_regional_rmse'] = np.std(regional_rmse)
        
        # Spatial correlation (if coordinates available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            spatial_corr = self._calculate_spatial_correlation(df, y_true_col, y_pred_col)
            metrics['spatial_correlation'] = spatial_corr
        
        logger.info(f"Calculated spatial metrics: {metrics}")
        return metrics
    
    def _calculate_spatial_correlation(
        self,
        df: pd.DataFrame,
        y_true_col: str,
        y_pred_col: str,
    ) -> float:
        """Calculate spatial correlation of residuals.
        
        Args:
            df: DataFrame with coordinates and predictions.
            y_true_col: Column name for true values.
            y_pred_col: Column name for predicted values.
            
        Returns:
            Spatial correlation coefficient.
        """
        residuals = df[y_true_col] - df[y_pred_col]
        
        # Simple distance-based correlation
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import pearsonr
        
        coords = df[['latitude', 'longitude']].values
        distances = squareform(pdist(coords))
        
        # Calculate correlation between residuals and distances
        # (simplified spatial correlation measure)
        if len(residuals) > 1:
            correlation, _ = pearsonr(residuals, np.mean(distances, axis=1))
            return correlation
        else:
            return 0.0


class VaccineEvaluator:
    """Comprehensive evaluator for vaccine distribution models."""
    
    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.metrics_calculator = MetricsCalculator()
        self.results = []
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
    ) -> ModelMetrics:
        """Evaluate a single model.
        
        Args:
            model: Trained model object.
            X_test: Test features.
            y_test: Test targets.
            X_train: Training features (for baseline).
            y_train: Training targets (for baseline).
            model_name: Name of the model.
            
        Returns:
            ModelMetrics object.
        """
        if model_name is None:
            model_name = getattr(model, 'name', 'unknown_model')
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate baseline predictions (simple mean)
        if X_train is not None and y_train is not None:
            y_baseline = np.full_like(y_test, np.mean(y_train))
        else:
            y_baseline = None
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            y_test, y_pred, y_baseline, model_name
        )
        
        # Add feature importance if available
        if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
            metrics.feature_importance = model.feature_importance_
        
        self.results.append(metrics)
        return metrics
    
    def evaluate_multiple_models(
        self,
        models: List[Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> List[ModelMetrics]:
        """Evaluate multiple models and return results.
        
        Args:
            models: List of trained model objects.
            X_test: Test features.
            y_test: Test targets.
            X_train: Training features (for baseline).
            y_train: Training targets (for baseline).
            
        Returns:
            List of ModelMetrics objects.
        """
        results = []
        
        for model in models:
            metrics = self.evaluate_model(
                model, X_test, y_test, X_train, y_train
            )
            results.append(metrics)
        
        logger.info(f"Evaluated {len(models)} models")
        return results
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard from evaluation results.
        
        Returns:
            DataFrame with model rankings and metrics.
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        data = []
        for metrics in self.results:
            data.append({
                'Model': metrics.model_name,
                'RMSE': metrics.rmse,
                'MAE': metrics.mae,
                'MAPE': metrics.mape,
                'R²': metrics.r2,
                'SMAPE': metrics.smape,
                'MASE': metrics.mase,
            })
        
        df = pd.DataFrame(data)
        
        # Sort by RMSE (lower is better)
        df = df.sort_values('RMSE').reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        columns = ['Rank', 'Model', 'RMSE', 'MAE', 'MAPE', 'R²', 'SMAPE', 'MASE']
        df = df[columns]
        
        logger.info("Created model leaderboard")
        return df
    
    def get_best_model(self) -> Optional[ModelMetrics]:
        """Get the best performing model based on RMSE.
        
        Returns:
            Best ModelMetrics object or None if no results.
        """
        if not self.results:
            return None
        
        best_model = min(self.results, key=lambda x: x.rmse)
        logger.info(f"Best model: {best_model.model_name} (RMSE: {best_model.rmse:.2f})")
        return best_model
    
    def compare_models(
        self,
        model1_metrics: ModelMetrics,
        model2_metrics: ModelMetrics,
    ) -> Dict[str, Any]:
        """Compare two models and return improvement statistics.
        
        Args:
            model1_metrics: First model metrics.
            model2_metrics: Second model metrics.
            
        Returns:
            Dictionary with comparison results.
        """
        comparison = {
            'model1': model1_metrics.model_name,
            'model2': model2_metrics.model_name,
            'rmse_improvement': (model1_metrics.rmse - model2_metrics.rmse) / model1_metrics.rmse * 100,
            'mae_improvement': (model1_metrics.mae - model2_metrics.mae) / model1_metrics.mae * 100,
            'r2_improvement': model2_metrics.r2 - model1_metrics.r2,
        }
        
        logger.info(f"Model comparison: {comparison}")
        return comparison
    
    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report.
        
        Returns:
            Formatted evaluation report string.
        """
        if not self.results:
            return "No evaluation results available."
        
        report = []
        report.append("VACCINE DISTRIBUTION MODEL EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append(f"Total models evaluated: {len(self.results)}")
        
        rmse_values = [m.rmse for m in self.results]
        report.append(f"RMSE range: {min(rmse_values):.2f} - {max(rmse_values):.2f}")
        
        r2_values = [m.r2 for m in self.results]
        report.append(f"R² range: {min(r2_values):.3f} - {max(r2_values):.3f}")
        report.append("")
        
        # Leaderboard
        leaderboard = self.create_leaderboard()
        report.append("MODEL LEADERBOARD:")
        report.append(leaderboard.to_string(index=False))
        report.append("")
        
        # Best model details
        best_model = self.get_best_model()
        if best_model:
            report.append("BEST MODEL DETAILS:")
            report.append(f"Model: {best_model.model_name}")
            report.append(f"RMSE: {best_model.rmse:.2f}")
            report.append(f"MAE: {best_model.mae:.2f}")
            report.append(f"R²: {best_model.r2:.3f}")
            report.append(f"MAPE: {best_model.mape:.2f}%")
        
        return "\n".join(report)
    
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save evaluation results to file.
        
        Args:
            filepath: Path to save the results.
        """
        from pathlib import Path
        import json
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        results_data = []
        for metrics in self.results:
            result_dict = {
                'model_name': metrics.model_name,
                'mse': float(metrics.mse),
                'rmse': float(metrics.rmse),
                'mae': float(metrics.mae),
                'mape': float(metrics.mape),
                'r2': float(metrics.r2),
                'smape': float(metrics.smape),
                'mase': float(metrics.mase),
            }
            if metrics.feature_importance is not None:
                result_dict['feature_importance'] = metrics.feature_importance.tolist()
            results_data.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved evaluation results to {filepath}")
