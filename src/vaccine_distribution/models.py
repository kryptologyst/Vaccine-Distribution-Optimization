"""Machine learning models for vaccine distribution optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import logging
import joblib
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class BaseVaccineModel(ABC):
    """Abstract base class for vaccine distribution models."""
    
    def __init__(self, name: str) -> None:
        """Initialize the model.
        
        Args:
            name: Model name for identification.
        """
        self.name = name
        self.is_fitted = False
        self.feature_importance_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to training data.
        
        Args:
            X: Feature matrix.
            y: Target vector.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted values.
        """
        pass
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved {self.name} model to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load the model from disk.
        
        Args:
            filepath: Path to load the model from.
        """
        filepath = Path(filepath)
        loaded_model = joblib.load(filepath)
        self.__dict__.update(loaded_model.__dict__)
        logger.info(f"Loaded {self.name} model from {filepath}")


class BaselineRegressor(BaseVaccineModel):
    """Baseline regression models for vaccine allocation prediction."""
    
    def __init__(self, model_type: str = "linear") -> None:
        """Initialize baseline regressor.
        
        Args:
            model_type: Type of baseline model ('linear', 'ridge', 'random_forest').
        """
        super().__init__(f"baseline_{model_type}")
        self.model_type = model_type
        self.model = self._create_model()
    
    def _create_model(self):
        """Create the appropriate baseline model."""
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the baseline model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        
        logger.info(f"Fitted {self.name} model")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted vaccine allocations.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        return predictions


class GradientBoostingRegressor(BaseVaccineModel):
    """Gradient boosting models for vaccine allocation prediction."""
    
    def __init__(self, model_type: str = "xgboost") -> None:
        """Initialize gradient boosting regressor.
        
        Args:
            model_type: Type of boosting model ('xgboost', 'lightgbm').
        """
        super().__init__(f"gradient_boosting_{model_type}")
        self.model_type = model_type
        self.model = self._create_model()
    
    def _create_model(self):
        """Create the appropriate boosting model."""
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the gradient boosting model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        
        logger.info(f"Fitted {self.name} model")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted vaccine allocations.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        return predictions


class VaccineNeuralNetwork(nn.Module):
    """Neural network for vaccine allocation prediction."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32], dropout_rate: float = 0.2) -> None:
        """Initialize the neural network.
        
        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        return self.network(x)


class NeuralNetworkRegressor(BaseVaccineModel):
    """Neural network regressor for vaccine allocation prediction."""
    
    def __init__(
        self,
        hidden_sizes: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: Optional[str] = None,
    ) -> None:
        """Initialize neural network regressor.
        
        Args:
            hidden_sizes: List of hidden layer sizes.
            dropout_rate: Dropout rate for regularization.
            learning_rate: Learning rate for optimization.
            batch_size: Batch size for training.
            epochs: Number of training epochs.
            device: Device to use ('cuda', 'mps', 'cpu').
        """
        super().__init__("neural_network")
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        logger.info(f"Initialized neural network on device: {self.device}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the neural network model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Move to device
        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        
        # Create model
        self.model = VaccineNeuralNetwork(
            input_size=X.shape[1],
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_fitted = True
        logger.info(f"Fitted {self.name} model")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted vaccine allocations.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        return predictions


class VaccineAllocationOptimizer:
    """Optimization-based vaccine allocation using linear programming."""
    
    def __init__(self) -> None:
        """Initialize the optimizer."""
        self.name = "allocation_optimizer"
        self.is_fitted = False
        self.optimal_allocations = None
    
    def optimize_allocation(
        self,
        regions: List[Dict[str, Any]],
        total_vaccines: int,
        constraints: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Optimize vaccine allocation across regions.
        
        Args:
            regions: List of region data dictionaries.
            total_vaccines: Total vaccines available.
            constraints: Optional constraints (e.g., max_per_region).
            
        Returns:
            Dictionary mapping region_id to optimal allocation.
        """
        try:
            import cvxpy as cp
        except ImportError:
            logger.warning("CVXPY not available, using simple proportional allocation")
            return self._proportional_allocation(regions, total_vaccines)
        
        n_regions = len(regions)
        
        # Decision variables: allocation per region
        allocations = cp.Variable(n_regions, nonneg=True)
        
        # Extract priority scores and demands
        priorities = np.array([r.get('priority_score', 0.5) for r in regions])
        demands = np.array([r.get('vaccine_demand', 0) for r in regions])
        
        # Objective: maximize weighted allocation (priority-weighted)
        objective = cp.Maximize(cp.sum(cp.multiply(allocations, priorities)))
        
        # Constraints
        constraints_list = [
            cp.sum(allocations) <= total_vaccines,  # Total vaccine constraint
            allocations >= 0,  # Non-negative allocations
        ]
        
        # Demand constraints (don't exceed demand)
        for i in range(n_regions):
            constraints_list.append(allocations[i] <= demands[i])
        
        # Additional constraints if provided
        if constraints:
            if 'max_per_region' in constraints:
                max_per_region = constraints['max_per_region']
                for i in range(n_regions):
                    constraints_list.append(allocations[i] <= max_per_region)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints_list)
        problem.solve(verbose=False)
        
        if problem.status == cp.OPTIMAL:
            optimal_allocations = allocations.value
            self.optimal_allocations = optimal_allocations
            self.is_fitted = True
            
            # Create result dictionary
            result = {}
            for i, region in enumerate(regions):
                result[region['region_id']] = float(optimal_allocations[i])
            
            logger.info(f"Optimized vaccine allocation for {n_regions} regions")
            return result
        else:
            logger.warning(f"Optimization failed with status: {problem.status}")
            return self._proportional_allocation(regions, total_vaccines)
    
    def _proportional_allocation(
        self, 
        regions: List[Dict[str, Any]], 
        total_vaccines: int
    ) -> Dict[str, float]:
        """Fallback proportional allocation based on priority scores.
        
        Args:
            regions: List of region data dictionaries.
            total_vaccines: Total vaccines available.
            
        Returns:
            Dictionary mapping region_id to allocation.
        """
        priorities = np.array([r.get('priority_score', 0.5) for r in regions])
        demands = np.array([r.get('vaccine_demand', 0) for r in regions])
        
        # Weight by priority and cap by demand
        weights = priorities * np.minimum(1.0, demands / np.mean(demands))
        weights = weights / np.sum(weights)
        
        allocations = weights * total_vaccines
        
        result = {}
        for i, region in enumerate(regions):
            result[region['region_id']] = float(allocations[i])
        
        logger.info("Used proportional allocation as fallback")
        return result
