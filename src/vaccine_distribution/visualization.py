"""Visualization components for vaccine distribution optimization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VaccineVisualizer:
    """Create visualizations for vaccine distribution analysis."""
    
    def __init__(self, style: str = "seaborn-v0_8") -> None:
        """Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use.
        """
        self.style = style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_prediction_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot actual vs predicted vaccine allocations.
        
        Args:
            y_true: True vaccine allocations.
            y_pred: Predicted vaccine allocations.
            model_name: Name of the model.
            save_path: Optional path to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Vaccine Allocation')
        ax1.set_ylabel('Predicted Vaccine Allocation')
        ax1.set_title(f'{model_name}: Actual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Calculate R² for display
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Vaccine Allocation')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name}: Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prediction comparison plot to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot feature importance for tree-based models.
        
        Args:
            feature_names: List of feature names.
            importance_values: Feature importance values.
            model_name: Name of the model.
            save_path: Optional path to save the plot.
        """
        # Sort features by importance
        sorted_indices = np.argsort(importance_values)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_importance = importance_values[sorted_indices]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(sorted_names)), sorted_importance)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'{model_name}: Feature Importance')
        plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
        
        # Color bars by importance
        colors = plt.cm.viridis(sorted_importance / sorted_importance.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        results: List[Dict[str, Any]],
        metric: str = "rmse",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot comparison of multiple models.
        
        Args:
            results: List of model results dictionaries.
            metric: Metric to compare (rmse, mae, r2, etc.).
            save_path: Optional path to save the plot.
        """
        model_names = [r['model_name'] for r in results]
        metric_values = [r[metric] for r in results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values)
        plt.xlabel('Models')
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison: {metric.upper()}')
        plt.xticks(rotation=45, ha='right')
        
        # Color bars based on performance
        if metric in ['rmse', 'mae', 'mape']:
            # Lower is better
            colors = plt.cm.RdYlGn_r(np.array(metric_values) / max(metric_values))
        else:
            # Higher is better (e.g., r2)
            colors = plt.cm.RdYlGn(np.array(metric_values) / max(metric_values))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {save_path}")
        
        plt.show()
    
    def plot_distribution_analysis(
        self,
        df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot distribution analysis of vaccine data.
        
        Args:
            df: DataFrame with vaccine distribution data.
            save_path: Optional path to save the plot.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Population distribution
        axes[0].hist(df['population'], bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_title('Population Distribution')
        axes[0].set_xlabel('Population')
        axes[0].set_ylabel('Frequency')
        
        # Infection rate distribution
        axes[1].hist(df['infection_rate'], bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_title('Infection Rate Distribution')
        axes[1].set_xlabel('Infection Rate')
        axes[1].set_ylabel('Frequency')
        
        # Elderly ratio distribution
        axes[2].hist(df['elderly_ratio'], bins=30, alpha=0.7, edgecolor='black')
        axes[2].set_title('Elderly Ratio Distribution')
        axes[2].set_xlabel('Elderly Ratio')
        axes[2].set_ylabel('Frequency')
        
        # Logistics score distribution
        axes[3].hist(df['logistics_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[3].set_title('Logistics Score Distribution')
        axes[3].set_xlabel('Logistics Score')
        axes[3].set_ylabel('Frequency')
        
        # Cold chain capacity distribution
        axes[4].hist(df['cold_chain_capacity'], bins=30, alpha=0.7, edgecolor='black')
        axes[4].set_title('Cold Chain Capacity Distribution')
        axes[4].set_xlabel('Cold Chain Capacity')
        axes[4].set_ylabel('Frequency')
        
        # Vaccine demand distribution
        axes[5].hist(df['vaccine_demand'], bins=30, alpha=0.7, edgecolor='black')
        axes[5].set_title('Vaccine Demand Distribution')
        axes[5].set_xlabel('Vaccine Demand')
        axes[5].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved distribution analysis plot to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot correlation heatmap of vaccine data features.
        
        Args:
            df: DataFrame with vaccine distribution data.
            save_path: Optional path to save the plot.
        """
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation heatmap to {save_path}")
        
        plt.show()


class MapVisualizer:
    """Create interactive maps for vaccine distribution visualization."""
    
    def __init__(self) -> None:
        """Initialize the map visualizer."""
        pass
    
    def create_vaccine_allocation_map(
        self,
        df: pd.DataFrame,
        allocation_col: str = "vaccine_allocation",
        region_col: str = "region_id",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        title: str = "Vaccine Allocation Map",
        save_path: Optional[Union[str, Path]] = None,
    ) -> folium.Map:
        """Create an interactive map showing vaccine allocations.
        
        Args:
            df: DataFrame with region data and allocations.
            allocation_col: Column name for vaccine allocations.
            region_col: Column name for region identifiers.
            lat_col: Column name for latitude.
            lon_col: Column name for longitude.
            title: Title for the map.
            save_path: Optional path to save the map.
            
        Returns:
            Folium map object.
        """
        # Create base map
        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        
        # Add markers for each region
        for _, row in df.iterrows():
            # Determine marker color based on allocation
            allocation = row[allocation_col]
            max_allocation = df[allocation_col].max()
            min_allocation = df[allocation_col].min()
            
            # Normalize allocation for color mapping
            normalized_allocation = (allocation - min_allocation) / (max_allocation - min_allocation)
            
            # Color from green (low) to red (high)
            color = f'rgb({int(255 * normalized_allocation)}, {int(255 * (1 - normalized_allocation))}, 0)'
            
            # Create popup text
            popup_text = f"""
            <b>{row[region_col]}</b><br>
            Vaccine Allocation: {allocation:,.0f}<br>
            Population: {row.get('population', 'N/A'):,}<br>
            Infection Rate: {row.get('infection_rate', 'N/A'):.3f}<br>
            Priority Score: {row.get('priority_score', 'N/A'):.3f}
            """
            
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=8,
                popup=folium.Popup(popup_text, max_width=300),
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 80px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Vaccine Allocation</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Low Allocation</p>
        <p><i class="fa fa-circle" style="color:red"></i> High Allocation</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(str(save_path))
            logger.info(f"Saved vaccine allocation map to {save_path}")
        
        return m
    
    def create_priority_map(
        self,
        df: pd.DataFrame,
        priority_col: str = "priority_score",
        region_col: str = "region_id",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        title: str = "Priority Score Map",
        save_path: Optional[Union[str, Path]] = None,
    ) -> folium.Map:
        """Create an interactive map showing priority scores.
        
        Args:
            df: DataFrame with region data and priority scores.
            priority_col: Column name for priority scores.
            region_col: Column name for region identifiers.
            lat_col: Column name for latitude.
            lon_col: Column name for longitude.
            title: Title for the map.
            save_path: Optional path to save the map.
            
        Returns:
            Folium map object.
        """
        # Create base map
        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        
        # Add markers for each region
        for _, row in df.iterrows():
            priority = row[priority_col]
            
            # Color from blue (low priority) to red (high priority)
            color_intensity = int(255 * priority)
            color = f'rgb({color_intensity}, 0, {255 - color_intensity})'
            
            # Create popup text
            popup_text = f"""
            <b>{row[region_col]}</b><br>
            Priority Score: {priority:.3f}<br>
            Population: {row.get('population', 'N/A'):,}<br>
            Infection Rate: {row.get('infection_rate', 'N/A'):.3f}<br>
            Elderly Ratio: {row.get('elderly_ratio', 'N/A'):.3f}
            """
            
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=8,
                popup=folium.Popup(popup_text, max_width=300),
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 80px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Priority Score</b></p>
        <p><i class="fa fa-circle" style="color:blue"></i> Low Priority</p>
        <p><i class="fa fa-circle" style="color:red"></i> High Priority</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(str(save_path))
            logger.info(f"Saved priority map to {save_path}")
        
        return m
    
    def create_interactive_dashboard(
        self,
        df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Create an interactive dashboard with multiple visualizations.
        
        Args:
            df: DataFrame with vaccine distribution data.
            save_path: Optional path to save the dashboard.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Population vs Vaccine Demand', 'Infection Rate vs Priority',
                          'Logistics Score Distribution', 'Cold Chain Capacity vs Allocation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Population vs Vaccine Demand
        fig.add_trace(
            go.Scatter(
                x=df['population'],
                y=df['vaccine_demand'],
                mode='markers',
                name='Regions',
                text=df['region_id'],
                hovertemplate='<b>%{text}</b><br>Population: %{x:,}<br>Demand: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Infection Rate vs Priority
        fig.add_trace(
            go.Scatter(
                x=df['infection_rate'],
                y=df['priority_score'],
                mode='markers',
                name='Priority',
                text=df['region_id'],
                hovertemplate='<b>%{text}</b><br>Infection Rate: %{x:.3f}<br>Priority: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Logistics Score Distribution
        fig.add_trace(
            go.Histogram(
                x=df['logistics_score'],
                name='Logistics Score',
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # Cold Chain Capacity vs Allocation
        fig.add_trace(
            go.Scatter(
                x=df['cold_chain_capacity'],
                y=df['vaccine_demand'],
                mode='markers',
                name='Capacity vs Demand',
                text=df['region_id'],
                hovertemplate='<b>%{text}</b><br>Capacity: %{x:,.0f}<br>Demand: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Vaccine Distribution Dashboard",
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Population", row=1, col=1)
        fig.update_yaxes(title_text="Vaccine Demand", row=1, col=1)
        fig.update_xaxes(title_text="Infection Rate", row=1, col=2)
        fig.update_yaxes(title_text="Priority Score", row=1, col=2)
        fig.update_xaxes(title_text="Logistics Score", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Cold Chain Capacity", row=2, col=2)
        fig.update_yaxes(title_text="Vaccine Demand", row=2, col=2)
        
        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Saved interactive dashboard to {save_path}")
        
        fig.show()
