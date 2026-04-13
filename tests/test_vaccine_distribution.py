"""Tests for vaccine distribution optimization package."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vaccine_distribution.data import VaccineDataGenerator, VaccineDataProcessor
from vaccine_distribution.models import (
    BaselineRegressor,
    GradientBoostingRegressor,
    NeuralNetworkRegressor,
    VaccineAllocationOptimizer,
)
from vaccine_distribution.evaluation import VaccineEvaluator, MetricsCalculator


class TestVaccineDataGenerator:
    """Test cases for VaccineDataGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = VaccineDataGenerator(seed=42)
        assert generator.seed == 42
    
    def test_generate_regions(self):
        """Test region generation."""
        generator = VaccineDataGenerator(seed=42)
        regions = generator.generate_regions(n_regions=10)
        
        assert len(regions) == 10
        assert all(hasattr(region, 'region_id') for region in regions)
        assert all(hasattr(region, 'population') for region in regions)
        assert all(hasattr(region, 'vaccine_demand') for region in regions)
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        generator1 = VaccineDataGenerator(seed=42)
        generator2 = VaccineDataGenerator(seed=42)
        
        regions1 = generator1.generate_regions(n_regions=5)
        regions2 = generator2.generate_regions(n_regions=5)
        
        assert len(regions1) == len(regions2)
        for r1, r2 in zip(regions1, regions2):
            assert r1.population == r2.population
            assert r1.infection_rate == r2.infection_rate


class TestVaccineDataProcessor:
    """Test cases for VaccineDataProcessor."""
    
    def test_init(self):
        """Test processor initialization."""
        processor = VaccineDataProcessor()
        assert processor.feature_columns is not None
        assert processor.target_column == 'vaccine_demand'
    
    def test_regions_to_dataframe(self):
        """Test conversion of regions to DataFrame."""
        generator = VaccineDataGenerator(seed=42)
        regions = generator.generate_regions(n_regions=5)
        
        processor = VaccineDataProcessor()
        df = processor.regions_to_dataframe(regions)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'region_id' in df.columns
        assert 'population' in df.columns
        assert 'vaccine_demand' in df.columns
    
    def test_prepare_features(self):
        """Test feature preparation."""
        generator = VaccineDataGenerator(seed=42)
        regions = generator.generate_regions(n_regions=10)
        
        processor = VaccineDataProcessor()
        df = processor.regions_to_dataframe(regions)
        X, y = processor.prepare_features(df, normalize=True, add_interactions=True)
        
        assert X.shape[0] == 10
        assert X.shape[1] == 8  # 5 base + 3 interaction features
        assert y.shape[0] == 10
        assert len(y.shape) == 1
    
    def test_split_data(self):
        """Test data splitting."""
        generator = VaccineDataGenerator(seed=42)
        regions = generator.generate_regions(n_regions=20)
        
        processor = VaccineDataProcessor()
        df = processor.regions_to_dataframe(regions)
        X, y = processor.prepare_features(df)
        
        X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2)
        
        assert X_train.shape[0] == 16  # 80% of 20
        assert X_test.shape[0] == 4    # 20% of 20
        assert y_train.shape[0] == 16
        assert y_test.shape[0] == 4


class TestBaselineRegressor:
    """Test cases for BaselineRegressor."""
    
    def test_init(self):
        """Test model initialization."""
        model = BaselineRegressor("linear")
        assert model.name == "baseline_linear"
        assert model.model_type == "linear"
        assert not model.is_fitted
    
    def test_fit_predict(self):
        """Test model fitting and prediction."""
        # Generate test data
        generator = VaccineDataGenerator(seed=42)
        regions = generator.generate_regions(n_regions=20)
        processor = VaccineDataProcessor()
        df = processor.regions_to_dataframe(regions)
        X, y = processor.prepare_features(df)
        
        # Test linear regression
        model = BaselineRegressor("linear")
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)


class TestGradientBoostingRegressor:
    """Test cases for GradientBoostingRegressor."""
    
    def test_init(self):
        """Test model initialization."""
        model = GradientBoostingRegressor("xgboost")
        assert model.name == "gradient_boosting_xgboost"
        assert model.model_type == "xgboost"
    
    def test_fit_predict(self):
        """Test model fitting and prediction."""
        # Generate test data
        generator = VaccineDataGenerator(seed=42)
        regions = generator.generate_regions(n_regions=20)
        processor = VaccineDataProcessor()
        df = processor.regions_to_dataframe(regions)
        X, y = processor.prepare_features(df)
        
        # Test XGBoost
        model = GradientBoostingRegressor("xgboost")
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)


class TestNeuralNetworkRegressor:
    """Test cases for NeuralNetworkRegressor."""
    
    def test_init(self):
        """Test model initialization."""
        model = NeuralNetworkRegressor()
        assert model.name == "neural_network"
        assert not model.is_fitted
    
    def test_fit_predict(self):
        """Test model fitting and prediction."""
        # Generate test data
        generator = VaccineDataGenerator(seed=42)
        regions = generator.generate_regions(n_regions=20)
        processor = VaccineDataProcessor()
        df = processor.regions_to_dataframe(regions)
        X, y = processor.prepare_features(df)
        
        # Test neural network
        model = NeuralNetworkRegressor(epochs=10)  # Reduced epochs for testing
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)


class TestVaccineAllocationOptimizer:
    """Test cases for VaccineAllocationOptimizer."""
    
    def test_init(self):
        """Test optimizer initialization."""
        optimizer = VaccineAllocationOptimizer()
        assert optimizer.name == "allocation_optimizer"
        assert not optimizer.is_fitted
    
    def test_optimize_allocation(self):
        """Test allocation optimization."""
        # Create test regions
        regions = [
            {
                'region_id': 'region_001',
                'population': 100000,
                'infection_rate': 0.02,
                'elderly_ratio': 0.15,
                'logistics_score': 0.8,
                'cold_chain_capacity': 5000,
                'vaccine_demand': 3000,
                'priority_score': 0.7
            },
            {
                'region_id': 'region_002',
                'population': 50000,
                'infection_rate': 0.01,
                'elderly_ratio': 0.10,
                'logistics_score': 0.6,
                'cold_chain_capacity': 3000,
                'vaccine_demand': 1500,
                'priority_score': 0.5
            }
        ]
        
        optimizer = VaccineAllocationOptimizer()
        total_vaccines = 3000
        allocations = optimizer.optimize_allocation(regions, total_vaccines)
        
        assert isinstance(allocations, dict)
        assert len(allocations) == 2
        assert 'region_001' in allocations
        assert 'region_002' in allocations
        
        # Check that total allocation doesn't exceed available vaccines
        total_allocated = sum(allocations.values())
        assert total_allocated <= total_vaccines


class TestMetricsCalculator:
    """Test cases for MetricsCalculator."""
    
    def test_init(self):
        """Test calculator initialization."""
        calculator = MetricsCalculator()
        assert calculator is not None
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        calculator = MetricsCalculator()
        
        # Create test data
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        y_baseline = np.array([150, 150, 150, 150, 150])
        
        metrics = calculator.calculate_metrics(
            y_true, y_pred, y_baseline, "test_model"
        )
        
        assert metrics.model_name == "test_model"
        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert metrics.r2 is not None
        assert metrics.mape > 0
        assert metrics.smape > 0
        assert metrics.mase is not None


class TestVaccineEvaluator:
    """Test cases for VaccineEvaluator."""
    
    def test_init(self):
        """Test evaluator initialization."""
        evaluator = VaccineEvaluator()
        assert evaluator.results == []
    
    def test_evaluate_model(self):
        """Test single model evaluation."""
        # Generate test data
        generator = VaccineDataGenerator(seed=42)
        regions = generator.generate_regions(n_regions=20)
        processor = VaccineDataProcessor()
        df = processor.regions_to_dataframe(regions)
        X, y = processor.prepare_features(df)
        X_train, X_test, y_train, y_test = processor.split_data(X, y)
        
        # Train and evaluate model
        model = BaselineRegressor("linear")
        model.fit(X_train, y_train)
        
        evaluator = VaccineEvaluator()
        metrics = evaluator.evaluate_model(model, X_test, y_test, X_train, y_train)
        
        assert metrics.model_name == "baseline_linear"
        assert metrics.rmse > 0
        assert len(evaluator.results) == 1
    
    def test_create_leaderboard(self):
        """Test leaderboard creation."""
        evaluator = VaccineEvaluator()
        
        # Add some dummy results
        from vaccine_distribution.evaluation import ModelMetrics
        evaluator.results = [
            ModelMetrics("model1", 100, 10, 5, 2, 0.8, 1.5, 0.9),
            ModelMetrics("model2", 80, 8.9, 4, 1.8, 0.85, 1.3, 0.95),
        ]
        
        leaderboard = evaluator.create_leaderboard()
        
        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == 2
        assert 'Rank' in leaderboard.columns
        assert 'Model' in leaderboard.columns
        assert 'RMSE' in leaderboard.columns


if __name__ == "__main__":
    pytest.main([__file__])
