# Vaccine Distribution Optimization

A comprehensive toolkit for optimizing vaccine distribution in public health scenarios, including demand prediction, logistics optimization, and resource allocation using machine learning and operations research techniques.

## Overview

This project addresses the critical challenge of efficient vaccine distribution during pandemics or routine immunization campaigns. It combines machine learning models for demand prediction with optimization algorithms for resource allocation, providing a complete pipeline from data generation to interactive visualization.

### Key Features

- **Multiple ML Models**: Linear regression, gradient boosting (XGBoost, LightGBM), and neural networks
- **Optimization Engine**: Linear programming-based vaccine allocation optimization
- **Interactive Demo**: Streamlit-based web application for exploration and analysis
- **Geographic Visualization**: Interactive maps showing allocation patterns and priority scores
- **Comprehensive Evaluation**: Multiple metrics including RMSE, MAE, R², MAPE, and spatial analysis
- **Synthetic Data Generation**: Realistic vaccine distribution scenarios for testing and research

## Project Structure

```
vaccine-distribution-optimization/
├── src/vaccine_distribution/          # Core package
│   ├── __init__.py
│   ├── data.py                      # Data generation and processing
│   ├── models.py                    # ML models and optimization
│   ├── evaluation.py                # Metrics and evaluation
│   └── visualization.py             # Charts and maps
├── scripts/                         # Training and utility scripts
│   └── train_models.py              # Main training pipeline
├── demo/                           # Interactive demo
│   └── app.py                      # Streamlit application
├── configs/                        # Configuration files
├── data/                          # Data directories
│   ├── raw/                       # Raw data
│   ├── processed/                # Processed data
│   └── external/                 # External data sources
├── assets/                        # Generated outputs
│   ├── visualizations/           # Charts and plots
│   └── results/                  # Model results
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
├── DISCLAIMER.md                # Important usage disclaimer
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Vaccine-Distribution-Optimization.git
   cd Vaccine-Distribution-Optimization
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install in development mode:
   ```bash
   pip install -e .
   ```

### Optional Dependencies

For development and testing:
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Run the Training Pipeline

Execute the complete machine learning pipeline:

```bash
python scripts/train_models.py
```

This will:
- Generate synthetic vaccine distribution data
- Train multiple ML models (linear, gradient boosting, neural networks)
- Evaluate models and create a leaderboard
- Generate visualizations and maps
- Demonstrate optimization algorithms

### 2. Launch the Interactive Demo

Start the Streamlit web application:

```bash
streamlit run demo/app.py
```

The demo provides:
- Interactive data generation and visualization
- Model training and comparison
- Geographic analysis with maps
- Vaccine allocation optimization
- Downloadable results

### 3. Use the Python API

```python
from vaccine_distribution.data import VaccineDataGenerator, VaccineDataProcessor
from vaccine_distribution.models import GradientBoostingRegressor
from vaccine_distribution.evaluation import VaccineEvaluator

# Generate data
generator = VaccineDataGenerator(seed=42)
regions = generator.generate_regions(n_regions=1000)

# Process data
processor = VaccineDataProcessor()
df = processor.regions_to_dataframe(regions)
X, y = processor.prepare_features(df)

# Train model
model = GradientBoostingRegressor("xgboost")
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

## Data Schema

### Input Features

- **population**: Total population in the region
- **infection_rate**: Current infection rate (cases per capita)
- **elderly_ratio**: Ratio of elderly population (65+ years)
- **logistics_score**: Logistics capability score (0-1, higher = better)
- **cold_chain_capacity**: Cold chain storage capacity (vaccines/day)

### Target Variable

- **vaccine_demand**: Optimal daily vaccine allocation needed

### Generated Features

The pipeline automatically creates interaction features:
- **risk_exposure**: population × infection_rate
- **elderly_population**: elderly_ratio × population
- **effective_capacity**: logistics_score × cold_chain_capacity

## Models

### Baseline Models
- **Linear Regression**: Simple linear baseline
- **Ridge Regression**: Regularized linear model
- **Random Forest**: Ensemble tree-based model

### Advanced Models
- **XGBoost**: Gradient boosting with XGBoost
- **LightGBM**: Gradient boosting with LightGBM
- **Neural Network**: Multi-layer perceptron with PyTorch

### Optimization
- **Linear Programming**: CVXPY-based allocation optimization
- **Constraint Handling**: Demand limits, capacity constraints
- **Priority Weighting**: Risk-based allocation prioritization

## Evaluation Metrics

### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error

### Spatial Metrics
- **Regional RMSE**: RMSE by geographic region
- **Spatial Correlation**: Correlation of residuals across space
- **Coverage Analysis**: Allocation coverage by priority zones

## Visualization

### Static Charts
- Prediction vs actual scatter plots
- Residual analysis
- Feature importance rankings
- Model comparison bar charts
- Data distribution histograms
- Correlation heatmaps

### Interactive Maps
- Vaccine allocation maps (Folium)
- Priority score visualization
- Geographic distribution analysis
- Interactive dashboards (Plotly)

## Configuration

### Data Generation Parameters

```python
generator = VaccineDataGenerator(seed=42)
regions = generator.generate_regions(
    n_regions=1000,
    population_mean=100000,
    population_std=20000,
    infection_rate_mean=0.02,
    infection_rate_std=0.01,
    elderly_ratio_mean=0.15,
    elderly_ratio_std=0.05,
    cold_chain_capacity_mean=5000,
    cold_chain_capacity_std=1000,
    include_geographic=True
)
```

### Model Parameters

```python
# Neural Network
nn_model = NeuralNetworkRegressor(
    hidden_sizes=[64, 32],
    dropout_rate=0.2,
    learning_rate=0.001,
    epochs=100,
    device="auto"  # CUDA, MPS, or CPU
)

# Gradient Boosting
gb_model = GradientBoostingRegressor("xgboost")
```

## Usage Examples

### Basic Training and Evaluation

```python
from vaccine_distribution import *

# Generate data
generator = VaccineDataGenerator(seed=42)
regions = generator.generate_regions(n_regions=1000)
processor = VaccineDataProcessor()
df = processor.regions_to_dataframe(regions)
X, y = processor.prepare_features(df)

# Split data
X_train, X_test, y_train, y_test = processor.split_data(X, y)

# Train models
models = [
    BaselineRegressor("linear"),
    GradientBoostingRegressor("xgboost"),
    NeuralNetworkRegressor()
]

for model in models:
    model.fit(X_train, y_train)

# Evaluate
evaluator = VaccineEvaluator()
evaluator.evaluate_multiple_models(models, X_test, y_test)
leaderboard = evaluator.create_leaderboard()
print(leaderboard)
```

### Optimization Example

```python
from vaccine_distribution.models import VaccineAllocationOptimizer

# Prepare region data
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
    # ... more regions
]

# Optimize allocation
optimizer = VaccineAllocationOptimizer()
total_vaccines = 100000
optimal_allocations = optimizer.optimize_allocation(regions, total_vaccines)

print("Optimal allocations:", optimal_allocations)
```

### Visualization Example

```python
from vaccine_distribution.visualization import VaccineVisualizer, MapVisualizer

# Create visualizations
visualizer = VaccineVisualizer()
map_viz = MapVisualizer()

# Prediction comparison
visualizer.plot_prediction_comparison(y_test, y_pred, "XGBoost Model")

# Feature importance
visualizer.plot_feature_importance(feature_names, importance_values)

# Interactive map
allocation_map = map_viz.create_vaccine_allocation_map(df)
allocation_map.save("allocation_map.html")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

**IMPORTANT**: This project is for research and educational purposes only. It uses synthetic data and simplified models that have not been validated for operational use. Do not use for real-world vaccine distribution planning without proper validation and domain expertise.

See [DISCLAIMER.md](DISCLAIMER.md) for complete details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{vaccine_distribution_optimization,
  title={Vaccine Distribution Optimization},
  author={Kryptologyst},
  url={https://github.com/kryptologyst/Vaccine-Distribution-Optimization},
  year={2026}
}
```

## Acknowledgments

- Public health researchers and practitioners
- Open-source machine learning community
- Operations research optimization libraries
- Geographic data visualization tools

---

**Author**: [kryptologyst](https://github.com/kryptologyst)  
**GitHub**: [https://github.com/kryptologyst](https://github.com/kryptologyst)
# Vaccine-Distribution-Optimization
