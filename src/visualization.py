import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

class Visualization:
    def __init__(self):
        # Set default color palette
        self.colors = {
            'primary': '#00ff88',
            'secondary': '#4ecdc4',
            'accent': '#ff6b6b',
            'warning': '#ffd93d',
            'success': '#6bcf7f',
            'error': '#ff5757'
        }
    
    def plot_var_distribution(self, returns, confidence_level, var_value):
        """Plot returns distribution with VaR threshold"""
        try:
            fig = go.Figure()
            
            # Histogram of returns
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns Distribution',
                marker_color=self.colors['primary'],
                opacity=0.7
            ))
            
            # VaR threshold line
            var_threshold = -var_value / 100000  # Convert back to return space
            fig.add_vline(
                x=var_threshold,
                line_dash="dash",
                line_color=self.colors['error'],
                annotation_text=f"VaR ({confidence_level*100:.0f}%)",
                annotation_position="top"
            )
            
            fig.update_layout(
                title="Returns Distribution with VaR Threshold",
                xaxis_title="Daily Returns",
                yaxis_title="Frequency",
                template="plotly_dark",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating VaR distribution plot: {str(e)}")
            return go.Figure()
    
    def plot_var_violations(self, returns, var_estimates, violation_dates):
        """Plot VaR violations over time"""
        try:
            fig = go.Figure()
            
            # Portfolio returns
            fig.add_trace(go.Scatter(
                x=returns.index,
                y=returns.values,
                mode='lines',
                name='Portfolio Returns',
                line=dict(color=self.colors['primary'], width=1)
            ))
            
            # VaR estimates
            fig.add_trace(go.Scatter(
                x=returns.index[-len(var_estimates):],
                y=[-v/100000 for v in var_estimates],  # Convert to return space
                mode='lines',
                name='VaR Estimates',
                line=dict(color=self.colors['secondary'], width=2)
            ))
            
            # Violation points
            if violation_dates:
                violation_returns = [returns.loc[date] for date in violation_dates if date in returns.index]
                fig.add_trace(go.Scatter(
                    x=violation_dates,
                    y=violation_returns,
                    mode='markers',
                    name='VaR Violations',
                    marker=dict(color=self.colors['error'], size=8, symbol='x')
                ))
            
            fig.update_layout(
                title="VaR Backtesting: Violations Over Time",
                xaxis_title="Date",
                yaxis_title="Returns",
                template="plotly_dark",
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating VaR violations plot: {str(e)}")
            return go.Figure()
    
    def plot_rolling_metrics(self, rolling_data, metric_name):
        """Plot rolling metrics"""
        try:
            fig = go.Figure()
            
            if isinstance(rolling_data, dict):
                # Multiple series
                for key, series in rolling_data.items():
                    fig.add_trace(go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode='lines',
                        name=key,
                        line=dict(width=2)
                    ))
            else:
                # Single series
                fig.add_trace(go.Scatter(
                    x=rolling_data.index,
                    y=rolling_data.values,
                    mode='lines',
                    name=metric_name,
                    line=dict(color=self.colors['primary'], width=2)
                ))
            
            fig.update_layout(
                title=f"Rolling {metric_name}",
                xaxis_title="Date",
                yaxis_title=metric_name,
                template="plotly_dark",
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating rolling metrics plot: {str(e)}")
            return go.Figure()
    
    def plot_correlation_heatmap(self, correlation_matrix):
        """Plot correlation heatmap"""
        try:
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                template="plotly_dark",
                color_continuous_scale="RdBu_r",
                title="Asset Correlation Matrix"
            )
            
            fig.update_layout(
                width=600,
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def plot_drawdown(self, drawdown_series):
        """Plot drawdown chart"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values * 100,
                mode='lines',
                fill='tonexty',
                name='Drawdown',
                line=dict(color=self.colors['error'], width=2),
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            
            fig.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating drawdown plot: {str(e)}")
            return go.Figure()
    
    def plot_stress_test_results(self, stress_results):
        """Plot stress test results"""
        try:
            scenarios = list(stress_results.keys())
            var_values = [stress_results[scenario]['stressed_var'] for scenario in scenarios]
            
            fig = px.bar(
                x=scenarios,
                y=var_values,
                title="Stress Test Results: VaR Across Scenarios",
                template="plotly_dark",
                color=var_values,
                color_continuous_scale="Reds"
            )
            
            fig.update_layout(
                xaxis_title="Scenario",
                yaxis_title="Stressed VaR ($)",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating stress test plot: {str(e)}")
            return go.Figure()
    
    def plot_options_payoff(self, S_range, K, option_type, T, r, sigma):
        """Plot option payoff diagram"""
        try:
            from src.options_var import OptionsVaR
            options_var = OptionsVaR()
            
            option_prices = []
            intrinsic_values = []
            
            for S in S_range:
                # Option price
                price = options_var.black_scholes_price(S, K, T, r, sigma, option_type)
                option_prices.append(price)
                
                # Intrinsic value
                if option_type.lower() == 'call':
                    intrinsic = max(S - K, 0)
                else:
                    intrinsic = max(K - S, 0)
                intrinsic_values.append(intrinsic)
            
            fig = go.Figure()
            
            # Option price
            fig.add_trace(go.Scatter(
                x=S_range,
                y=option_prices,
                mode='lines',
                name='Option Price',
                line=dict(color=self.colors['primary'], width=3)
            ))
            
            # Intrinsic value
            fig.add_trace(go.Scatter(
                x=S_range,
                y=intrinsic_values,
                mode='lines',
                name='Intrinsic Value',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
            
            # Strike price line
            fig.add_vline(
                x=K,
                line_dash="dot",
                line_color=self.colors['accent'],
                annotation_text=f"Strike: ${K}",
                annotation_position="top"
            )
            
            fig.update_layout(
                title=f"{option_type.title()} Option Payoff Diagram",
                xaxis_title="Stock Price ($)",
                yaxis_title="Option Value ($)",
                template="plotly_dark",
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating options payoff plot: {str(e)}")
            return go.Figure()
    
    def plot_var_comparison(self, var_results):
        """Plot VaR model comparison"""
        try:
            methods = list(var_results.keys())
            var_values = list(var_results.values())
            
            fig = px.bar(
                x=methods,
                y=var_values,
                title="VaR Model Comparison",
                template="plotly_dark",
                color=var_values,
                color_continuous_scale="Blues"
            )
            
            fig.update_layout(
                xaxis_title="VaR Method",
                yaxis_title="VaR ($)",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating VaR comparison plot: {str(e)}")
            return go.Figure()
    
    def plot_sensitivity_analysis(self, sensitivity_data):
        """Plot sensitivity analysis results"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sensitivity_data['shock'],
                y=sensitivity_data['var'],
                mode='lines+markers',
                name='VaR Sensitivity',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="VaR Sensitivity to Market Shocks",
                xaxis_title="Market Shock (%)",
                yaxis_title="VaR ($)",
                template="plotly_dark",
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating sensitivity analysis plot: {str(e)}")
            return go.Figure()
    
    def create_risk_dashboard(self, portfolio_data):
        """Create comprehensive risk dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Portfolio Performance',
                    'VaR Over Time',
                    'Drawdown',
                    'Risk Metrics'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "bar"}]]
            )
            
            # Portfolio performance
            if 'cumulative_returns' in portfolio_data:
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_data['cumulative_returns'].index,
                        y=portfolio_data['cumulative_returns'].values,
                        name='Cumulative Returns',
                        line=dict(color=self.colors['primary'])
                    ),
                    row=1, col=1
                )
            
            # VaR over time
            if 'rolling_var' in portfolio_data:
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_data['rolling_var'].index,
                        y=portfolio_data['rolling_var'].values,
                        name='Rolling VaR',
                        line=dict(color=self.colors['secondary'])
                    ),
                    row=1, col=2
                )
            
            # Drawdown
            if 'drawdown' in portfolio_data:
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_data['drawdown'].index,
                        y=portfolio_data['drawdown'].values * 100,
                        name='Drawdown (%)',
                        fill='tonexty',
                        line=dict(color=self.colors['error'])
                    ),
                    row=2, col=1
                )
            
            # Risk metrics
            if 'risk_metrics' in portfolio_data:
                metrics = list(portfolio_data['risk_metrics'].keys())
                values = list(portfolio_data['risk_metrics'].values())
                
                fig.add_trace(
                    go.Bar(
                        x=metrics,
                        y=values,
                        name='Risk Metrics',
                        marker_color=self.colors['accent']
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Risk Analytics Dashboard",
                template="plotly_dark",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating risk dashboard: {str(e)}")
            return go.Figure()