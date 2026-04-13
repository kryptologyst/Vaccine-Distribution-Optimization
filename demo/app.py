"""Interactive Streamlit demo for vaccine distribution optimization."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vaccine_distribution.data import VaccineDataGenerator, VaccineDataProcessor
from vaccine_distribution.models import (
    BaselineRegressor,
    GradientBoostingRegressor,
    NeuralNetworkRegressor,
    VaccineAllocationOptimizer,
)
from vaccine_distribution.evaluation import VaccineEvaluator
from vaccine_distribution.visualization import VaccineVisualizer, MapVisualizer

# Page configuration
st.set_page_config(
    page_title="Vaccine Distribution Optimization",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💉 Vaccine Distribution Optimization</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <strong>⚠️ Research Demo Disclaimer:</strong> This is a research and educational demonstration tool. 
    It uses synthetic data and simplified models. Do not use for operational vaccine distribution planning 
    without proper validation and domain expertise. For issues or questions, visit: 
    <a href="https://github.com/kryptologyst" target="_blank">https://github.com/kryptologyst</a>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Data generation parameters
st.sidebar.header("Data Parameters")
n_regions = st.sidebar.slider("Number of Regions", 100, 2000, 1000)
population_mean = st.sidebar.slider("Mean Population", 50000, 200000, 100000)
infection_rate_mean = st.sidebar.slider("Mean Infection Rate", 0.005, 0.05, 0.02)
elderly_ratio_mean = st.sidebar.slider("Mean Elderly Ratio", 0.05, 0.25, 0.15)

# Model parameters
st.sidebar.header("Model Parameters")
include_nn = st.sidebar.checkbox("Include Neural Network", value=True)
nn_epochs = st.sidebar.slider("Neural Network Epochs", 50, 200, 100)
include_geographic = st.sidebar.checkbox("Include Geographic Data", value=True)

# Optimization parameters
st.sidebar.header("Optimization Parameters")
total_vaccines = st.sidebar.number_input("Total Vaccines Available", 100000, 10000000, 500000, step=10000)
coverage_target = st.sidebar.slider("Coverage Target (%)", 50, 100, 80)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "🤖 Model Training", 
    "📈 Results & Evaluation", 
    "🗺️ Geographic Analysis", 
    "⚖️ Optimization"
])

with tab1:
    st.header("Data Overview")
    
    if st.button("Generate New Dataset", type="primary"):
        with st.spinner("Generating synthetic vaccine distribution data..."):
            # Generate data
            generator = VaccineDataGenerator(seed=42)
            regions = generator.generate_regions(
                n_regions=n_regions,
                population_mean=population_mean,
                infection_rate_mean=infection_rate_mean,
                elderly_ratio_mean=elderly_ratio_mean,
                include_geographic=include_geographic
            )
            
            # Process data
            processor = VaccineDataProcessor()
            df = processor.regions_to_dataframe(regions)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.data_generated = True
        
        st.success(f"Generated dataset with {len(df)} regions!")
    
    if st.session_state.data_generated:
        df = st.session_state.df
        
        # Dataset summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Regions", len(df))
        with col2:
            st.metric("Total Population", f"{df['population'].sum():,}")
        with col3:
            st.metric("Total Vaccine Demand", f"{df['vaccine_demand'].sum():,.0f}")
        with col4:
            st.metric("Avg Priority Score", f"{df['priority_score'].mean():.3f}")
        
        # Data distribution plots
        st.subheader("Data Distribution")
        
        # Create distribution plots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Population', 'Infection Rate', 'Elderly Ratio',
                          'Logistics Score', 'Cold Chain Capacity', 'Vaccine Demand')
        )
        
        # Population
        fig.add_trace(go.Histogram(x=df['population'], name='Population'), row=1, col=1)
        
        # Infection Rate
        fig.add_trace(go.Histogram(x=df['infection_rate'], name='Infection Rate'), row=1, col=2)
        
        # Elderly Ratio
        fig.add_trace(go.Histogram(x=df['elderly_ratio'], name='Elderly Ratio'), row=1, col=3)
        
        # Logistics Score
        fig.add_trace(go.Histogram(x=df['logistics_score'], name='Logistics Score'), row=2, col=1)
        
        # Cold Chain Capacity
        fig.add_trace(go.Histogram(x=df['cold_chain_capacity'], name='Cold Chain Capacity'), row=2, col=2)
        
        # Vaccine Demand
        fig.add_trace(go.Histogram(x=df['vaccine_demand'], name='Vaccine Demand'), row=2, col=3)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data table
        st.subheader("Sample Data")
        st.dataframe(df.head(20), use_container_width=True)

with tab2:
    st.header("Model Training")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
    else:
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                df = st.session_state.df
                
                # Prepare features
                processor = VaccineDataProcessor()
                X, y = processor.prepare_features(df, normalize=True, add_interactions=True)
                X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2)
                
                # Train models
                models = []
                
                # Baseline models
                baseline_models = [
                    BaselineRegressor("linear"),
                    BaselineRegressor("ridge"),
                    BaselineRegressor("random_forest"),
                ]
                
                for model in baseline_models:
                    model.fit(X_train, y_train)
                    models.append(model)
                
                # Gradient boosting models
                boosting_models = [
                    GradientBoostingRegressor("xgboost"),
                    GradientBoostingRegressor("lightgbm"),
                ]
                
                for model in boosting_models:
                    model.fit(X_train, y_train)
                    models.append(model)
                
                # Neural network (if enabled)
                if include_nn:
                    nn_model = NeuralNetworkRegressor(
                        hidden_sizes=[64, 32],
                        dropout_rate=0.2,
                        learning_rate=0.001,
                        epochs=nn_epochs,
                    )
                    nn_model.fit(X_train, y_train)
                    models.append(nn_model)
                
                # Store in session state
                st.session_state.models = models
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.models_trained = True
            
            st.success(f"Trained {len(models)} models successfully!")
        
        if st.session_state.models_trained:
            models = st.session_state.models
            
            # Model comparison
            st.subheader("Model Performance Comparison")
            
            # Evaluate models
            evaluator = VaccineEvaluator()
            evaluator.evaluate_multiple_models(
                models, 
                st.session_state.X_test, 
                st.session_state.y_test,
                st.session_state.X_train,
                st.session_state.y_train
            )
            
            # Create comparison chart
            results = []
            for i, metrics in enumerate(evaluator.results):
                results.append({
                    'Model': metrics.model_name,
                    'RMSE': metrics.rmse,
                    'MAE': metrics.mae,
                    'R²': metrics.r2,
                    'MAPE': metrics.mape
                })
            
            results_df = pd.DataFrame(results)
            
            # RMSE comparison
            fig_rmse = px.bar(
                results_df, 
                x='Model', 
                y='RMSE',
                title='Model RMSE Comparison (Lower is Better)',
                color='RMSE',
                color_continuous_scale='RdYlGn_r'
            )
            fig_rmse.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_rmse, use_container_width=True)
            
            # R² comparison
            fig_r2 = px.bar(
                results_df, 
                x='Model', 
                y='R²',
                title='Model R² Comparison (Higher is Better)',
                color='R²',
                color_continuous_scale='RdYlGn'
            )
            fig_r2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_r2, use_container_width=True)
            
            # Detailed results table
            st.subheader("Detailed Results")
            st.dataframe(results_df, use_container_width=True)

with tab3:
    st.header("Results & Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training tab.")
    else:
        models = st.session_state.models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Get best model
        evaluator = VaccineEvaluator()
        evaluator.evaluate_multiple_models(models, X_test, y_test)
        best_model_metrics = evaluator.get_best_model()
        
        if best_model_metrics:
            best_model_idx = evaluator.results.index(best_model_metrics)
            best_model = models[best_model_idx]
            
            st.subheader(f"Best Model: {best_model_metrics.model_name}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("RMSE", f"{best_model_metrics.rmse:.2f}")
            with col2:
                st.metric("MAE", f"{best_model_metrics.mae:.2f}")
            with col3:
                st.metric("R²", f"{best_model_metrics.r2:.3f}")
            with col4:
                st.metric("MAPE", f"{best_model_metrics.mape:.2f}%")
            
            # Predictions vs Actual
            y_pred = best_model.predict(X_test)
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'
            ))
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            fig_pred.update_layout(
                title='Predictions vs Actual Values',
                xaxis_title='Actual Vaccine Allocation',
                yaxis_title='Predicted Vaccine Allocation',
                height=500
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Residuals plot
            residuals = y_test - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                hovertemplate='Predicted: %{x}<br>Residual: %{y}<extra></extra>'
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig_res.update_layout(
                title='Residuals Plot',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                height=400
            )
            
            st.plotly_chart(fig_res, use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(best_model, 'feature_importance_') and best_model.feature_importance_ is not None:
                st.subheader("Feature Importance")
                
                feature_names = [
                    'Population', 'Infection Rate', 'Elderly Ratio',
                    'Logistics Score', 'Cold Chain Capacity',
                    'Risk Exposure', 'Elderly Population', 'Effective Capacity'
                ]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': best_model.feature_importance_
                }).sort_values('Importance', ascending=True)
                
                fig_imp = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_imp, use_container_width=True)

with tab4:
    st.header("Geographic Analysis")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
    elif not include_geographic:
        st.warning("Geographic data is disabled. Enable it in the sidebar to see maps.")
    else:
        df = st.session_state.df
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.subheader("Interactive Maps")
            
            # Map type selection
            map_type = st.selectbox(
                "Select Map Type",
                ["Vaccine Allocation", "Priority Score", "Infection Rate", "Population Density"]
            )
            
            # Create map based on selection
            map_viz = MapVisualizer()
            
            if map_type == "Vaccine Allocation":
                df['vaccine_allocation'] = df['vaccine_demand']  # Use demand as proxy
                m = map_viz.create_vaccine_allocation_map(df)
            elif map_type == "Priority Score":
                m = map_viz.create_priority_map(df)
            else:
                # Create custom map for other metrics
                center_lat = df['latitude'].mean()
                center_lon = df['longitude'].mean()
                m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
                
                # Determine color column
                if map_type == "Infection Rate":
                    color_col = 'infection_rate'
                    title = "Infection Rate Map"
                else:  # Population Density
                    color_col = 'population'
                    title = "Population Density Map"
                
                # Add markers
                for _, row in df.iterrows():
                    value = row[color_col]
                    max_val = df[color_col].max()
                    min_val = df[color_col].min()
                    
                    normalized_val = (value - min_val) / (max_val - min_val)
                    color = f'rgb({int(255 * normalized_val)}, {int(255 * (1 - normalized_val))}, 0)'
                    
                    popup_text = f"""
                    <b>{row['region_id']}</b><br>
                    {map_type}: {value:,.0f}<br>
                    Population: {row['population']:,}<br>
                    Priority Score: {row['priority_score']:.3f}
                    """
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=8,
                        popup=folium.Popup(popup_text, max_width=300),
                        color='black',
                        weight=1,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500)
            
            # Geographic statistics
            st.subheader("Geographic Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latitude Range", f"{df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
            with col2:
                st.metric("Longitude Range", f"{df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
            with col3:
                st.metric("Geographic Coverage", f"{df['latitude'].max() - df['latitude'].min():.1f}° × {df['longitude'].max() - df['longitude'].min():.1f}°")
            
            # Scatter plot: Geographic vs Demand
            fig_geo = px.scatter(
                df,
                x='longitude',
                y='latitude',
                size='vaccine_demand',
                color='priority_score',
                hover_data=['region_id', 'population', 'infection_rate'],
                title='Geographic Distribution of Vaccine Demand',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_geo, use_container_width=True)

with tab5:
    st.header("Vaccine Allocation Optimization")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
    else:
        df = st.session_state.df
        
        st.subheader("Optimization Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Vaccine Demand", f"{df['vaccine_demand'].sum():,.0f}")
            st.metric("Average Demand per Region", f"{df['vaccine_demand'].mean():,.0f}")
        
        with col2:
            st.metric("Available Vaccines", f"{total_vaccines:,}")
            coverage_percent = (total_vaccines / df['vaccine_demand'].sum()) * 100
            st.metric("Coverage Percentage", f"{coverage_percent:.1f}%")
        
        if st.button("Run Optimization", type="primary"):
            with st.spinner("Optimizing vaccine allocation..."):
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
                optimal_allocations = optimizer.optimize_allocation(
                    regions, 
                    total_vaccines,
                    constraints={'max_per_region': df['cold_chain_capacity'].max()}
                )
                
                # Store results
                st.session_state.optimal_allocations = optimal_allocations
        
        if 'optimal_allocations' in st.session_state:
            optimal_allocations = st.session_state.optimal_allocations
            
            # Create results DataFrame
            results_df = pd.DataFrame([
                {'region_id': rid, 'optimal_allocation': alloc}
                for rid, alloc in optimal_allocations.items()
            ])
            
            # Merge with original data
            results_df = results_df.merge(df, on='region_id')
            
            st.subheader("Optimization Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Allocated", f"{results_df['optimal_allocation'].sum():,.0f}")
            with col2:
                st.metric("Regions Covered", f"{len(results_df[results_df['optimal_allocation'] > 0])}")
            with col3:
                avg_allocation = results_df['optimal_allocation'].mean()
                st.metric("Avg Allocation", f"{avg_allocation:,.0f}")
            with col4:
                max_allocation = results_df['optimal_allocation'].max()
                st.metric("Max Allocation", f"{max_allocation:,.0f}")
            
            # Top allocations
            st.subheader("Top 20 Allocations")
            top_allocations = results_df.nlargest(20, 'optimal_allocation')[
                ['region_id', 'optimal_allocation', 'priority_score', 'vaccine_demand', 'population']
            ]
            st.dataframe(top_allocations, use_container_width=True)
            
            # Allocation distribution
            fig_alloc = px.histogram(
                results_df,
                x='optimal_allocation',
                nbins=30,
                title='Distribution of Optimal Allocations',
                labels={'optimal_allocation': 'Optimal Allocation', 'count': 'Number of Regions'}
            )
            st.plotly_chart(fig_alloc, use_container_width=True)
            
            # Priority vs Allocation
            fig_priority = px.scatter(
                results_df,
                x='priority_score',
                y='optimal_allocation',
                size='population',
                color='infection_rate',
                hover_data=['region_id', 'vaccine_demand'],
                title='Priority Score vs Optimal Allocation',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_priority, use_container_width=True)
            
            # Geographic optimization map (if coordinates available)
            if 'latitude' in df.columns and 'longitude' in df.columns:
                st.subheader("Geographic Optimization Results")
                
                # Create optimization map
                map_viz = MapVisualizer()
                opt_map = map_viz.create_vaccine_allocation_map(
                    results_df, 
                    allocation_col='optimal_allocation'
                )
                st_folium(opt_map, width=700, height=500)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Optimization Results",
                data=csv,
                file_name="vaccine_allocation_results.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Vaccine Distribution Optimization Demo | Author: <a href="https://github.com/kryptologyst" target="_blank">kryptologyst</a></p>
    <p>This is a research and educational tool. Not for operational use.</p>
</div>
""", unsafe_allow_html=True)
