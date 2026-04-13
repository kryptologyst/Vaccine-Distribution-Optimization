"""Data generation and processing for vaccine distribution optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class VaccineRegionData:
    """Data structure for a vaccine distribution region."""
    
    region_id: str
    population: int
    infection_rate: float
    elderly_ratio: float
    logistics_score: float
    cold_chain_capacity: int
    vaccine_demand: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    priority_score: Optional[float] = None


class VaccineDataGenerator:
    """Generate synthetic vaccine distribution data for modeling and testing."""
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize the data generator with deterministic seeding.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self._set_seeds()
    
    def _set_seeds(self) -> None:
        """Set all random seeds for deterministic behavior."""
        np.random.seed(self.seed)
        logger.info(f"Set random seed to {self.seed} for deterministic data generation")
    
    def generate_regions(
        self, 
        n_regions: int = 1000,
        population_mean: float = 100000,
        population_std: float = 20000,
        infection_rate_mean: float = 0.02,
        infection_rate_std: float = 0.01,
        elderly_ratio_mean: float = 0.15,
        elderly_ratio_std: float = 0.05,
        cold_chain_capacity_mean: float = 5000,
        cold_chain_capacity_std: float = 1000,
        include_geographic: bool = True,
    ) -> List[VaccineRegionData]:
        """Generate synthetic vaccine distribution regions.
        
        Args:
            n_regions: Number of regions to generate.
            population_mean: Mean population per region.
            population_std: Standard deviation of population.
            infection_rate_mean: Mean infection rate (cases per capita).
            infection_rate_std: Standard deviation of infection rate.
            elderly_ratio_mean: Mean ratio of elderly population.
            elderly_ratio_std: Standard deviation of elderly ratio.
            cold_chain_capacity_mean: Mean cold chain capacity (vaccines/day).
            cold_chain_capacity_std: Standard deviation of cold chain capacity.
            include_geographic: Whether to include geographic coordinates.
            
        Returns:
            List of VaccineRegionData objects.
        """
        regions = []
        
        for i in range(n_regions):
            # Generate demographic and health data
            population = max(1000, int(np.random.normal(population_mean, population_std)))
            infection_rate = max(0.001, np.random.normal(infection_rate_mean, infection_rate_std))
            elderly_ratio = max(0.05, min(0.4, np.random.normal(elderly_ratio_mean, elderly_ratio_std)))
            logistics_score = np.random.uniform(0, 1)
            cold_chain_capacity = max(100, int(np.random.normal(cold_chain_capacity_mean, cold_chain_capacity_std)))
            
            # Calculate vaccine demand based on multiple factors
            vaccine_demand = self._calculate_vaccine_demand(
                population, infection_rate, elderly_ratio, logistics_score, cold_chain_capacity
            )
            
            # Generate geographic coordinates if requested
            latitude = longitude = None
            if include_geographic:
                latitude = np.random.uniform(-60, 60)  # Global distribution
                longitude = np.random.uniform(-180, 180)
            
            # Calculate priority score based on risk factors
            priority_score = self._calculate_priority_score(
                infection_rate, elderly_ratio, logistics_score
            )
            
            region = VaccineRegionData(
                region_id=f"region_{i:04d}",
                population=population,
                infection_rate=infection_rate,
                elderly_ratio=elderly_ratio,
                logistics_score=logistics_score,
                cold_chain_capacity=cold_chain_capacity,
                vaccine_demand=vaccine_demand,
                latitude=latitude,
                longitude=longitude,
                priority_score=priority_score,
            )
            regions.append(region)
        
        logger.info(f"Generated {n_regions} vaccine distribution regions")
        return regions
    
    def _calculate_vaccine_demand(
        self,
        population: int,
        infection_rate: float,
        elderly_ratio: float,
        logistics_score: float,
        cold_chain_capacity: int,
    ) -> float:
        """Calculate vaccine demand based on multiple factors.
        
        Args:
            population: Total population in region.
            infection_rate: Current infection rate.
            elderly_ratio: Ratio of elderly population.
            logistics_score: Logistics capability score (0-1).
            cold_chain_capacity: Cold chain storage capacity.
            
        Returns:
            Calculated vaccine demand per day.
        """
        # Base demand from infected population (40% target coverage)
        infected_demand = population * infection_rate * 0.4
        
        # Priority demand for elderly (30% coverage)
        elderly_demand = elderly_ratio * population * 0.3
        
        # Logistics-adjusted delivery capacity
        logistics_demand = cold_chain_capacity * logistics_score * 0.6
        
        # Combine factors with some randomness
        total_demand = (
            infected_demand + 
            elderly_demand + 
            logistics_demand + 
            np.random.normal(0, 500)
        )
        
        return max(0, total_demand)
    
    def _calculate_priority_score(
        self,
        infection_rate: float,
        elderly_ratio: float,
        logistics_score: float,
    ) -> float:
        """Calculate priority score for vaccine allocation.
        
        Args:
            infection_rate: Current infection rate.
            elderly_ratio: Ratio of elderly population.
            logistics_score: Logistics capability score.
            
        Returns:
            Priority score (0-1, higher = more priority).
        """
        # Higher infection rate = higher priority
        infection_priority = min(1.0, infection_rate * 50)
        
        # Higher elderly ratio = higher priority
        elderly_priority = min(1.0, elderly_ratio * 5)
        
        # Lower logistics score = higher priority (need more support)
        logistics_priority = 1.0 - logistics_score
        
        # Weighted combination
        priority = (
            0.5 * infection_priority +
            0.3 * elderly_priority +
            0.2 * logistics_priority
        )
        
        return min(1.0, max(0.0, priority))


class VaccineDataProcessor:
    """Process and prepare vaccine distribution data for modeling."""
    
    def __init__(self) -> None:
        """Initialize the data processor."""
        self.feature_columns = [
            'population', 'infection_rate', 'elderly_ratio', 
            'logistics_score', 'cold_chain_capacity'
        ]
        self.target_column = 'vaccine_demand'
    
    def regions_to_dataframe(self, regions: List[VaccineRegionData]) -> pd.DataFrame:
        """Convert regions to pandas DataFrame.
        
        Args:
            regions: List of VaccineRegionData objects.
            
        Returns:
            DataFrame with region data.
        """
        data = []
        for region in regions:
            data.append({
                'region_id': region.region_id,
                'population': region.population,
                'infection_rate': region.infection_rate,
                'elderly_ratio': region.elderly_ratio,
                'logistics_score': region.logistics_score,
                'cold_chain_capacity': region.cold_chain_capacity,
                'vaccine_demand': region.vaccine_demand,
                'latitude': region.latitude,
                'longitude': region.longitude,
                'priority_score': region.priority_score,
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Converted {len(regions)} regions to DataFrame with shape {df.shape}")
        return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame,
        normalize: bool = True,
        add_interactions: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for modeling.
        
        Args:
            df: DataFrame with region data.
            normalize: Whether to normalize features.
            add_interactions: Whether to add interaction features.
            
        Returns:
            Tuple of (features, targets) arrays.
        """
        # Extract base features
        X = df[self.feature_columns].values
        
        # Add interaction features
        if add_interactions:
            X = self._add_interaction_features(X)
        
        # Normalize features
        if normalize:
            X = self._normalize_features(X)
        
        # Extract targets
        y = df[self.target_column].values
        
        logger.info(f"Prepared features with shape {X.shape} and targets with shape {y.shape}")
        return X, y
    
    def _add_interaction_features(self, X: np.ndarray) -> np.ndarray:
        """Add interaction features to improve model performance.
        
        Args:
            X: Base feature matrix.
            
        Returns:
            Feature matrix with interaction features.
        """
        # population * infection_rate (risk exposure)
        risk_exposure = X[:, 0:1] * X[:, 1:2]
        
        # elderly_ratio * population (elderly population)
        elderly_pop = X[:, 2:3] * X[:, 0:1]
        
        # logistics_score * cold_chain_capacity (effective capacity)
        effective_capacity = X[:, 3:4] * X[:, 4:5]
        
        # Combine original and interaction features
        X_interactions = np.concatenate([X, risk_exposure, elderly_pop, effective_capacity], axis=1)
        
        logger.info(f"Added interaction features: {X.shape} -> {X_interactions.shape}")
        return X_interactions
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Normalized feature matrix.
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
        
        logger.info("Normalized features using z-score normalization")
        return X_normalized
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Split data: train {X_train.shape}, test {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def save_data(self, df: pd.DataFrame, filepath: Union[str, Path]) -> None:
        """Save processed data to file.
        
        Args:
            df: DataFrame to save.
            filepath: Path to save the data.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.csv':
            df.to_csv(filepath, index=False)
        elif filepath.suffix == '.parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load processed data from file.
        
        Args:
            filepath: Path to load the data from.
            
        Returns:
            Loaded DataFrame.
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded data from {filepath} with shape {df.shape}")
        return df
