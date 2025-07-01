import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import streamlit as st
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

class VaREngines:
    def __init__(self):
        pass
    
    def calculate_parametric_var(self, returns, confidence_level, time_horizon=1):
        """Calculate VaR using parametric (Delta-Normal) method"""
        try:
            if len(returns) == 0:
                return 0
            
            # Calculate mean and standard deviation
            mu = returns.mean()
            sigma = returns.std()
            
            # Calculate VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mu * time_horizon + sigma * np.sqrt(time_horizon) * z_score)
            
            # Convert to dollar amount (assuming $1 portfolio)
            portfolio_value = 100000  # $100,000 default portfolio
            var_dollar = var * portfolio_value
            
            return var_dollar
            
        except Exception as e:
            st.error(f"Error calculating parametric VaR: {str(e)}")
            return 0
    
    def calculate_historical_var(self, returns, confidence_level, time_horizon=1):
        """Calculate VaR using historical simulation"""
        try:
            if len(returns) == 0:
                return 0
            
            # Scale returns for time horizon
            scaled_returns = returns * np.sqrt(time_horizon)
            
            # Calculate VaR as percentile
            var_percentile = (1 - confidence_level) * 100
            var = -np.percentile(scaled_returns, var_percentile)
            
            # Convert to dollar amount
            portfolio_value = 100000
            var_dollar = var * portfolio_value
            
            return var_dollar
            
        except Exception as e:
            st.error(f"Error calculating historical VaR: {str(e)}")
            return 0
    
    def calculate_monte_carlo_var(self, returns, confidence_level, time_horizon=1, num_simulations=10000):
        """Calculate VaR using Monte Carlo simulation"""
        try:
            if len(returns) == 0:
                return 0
            
            # Calculate parameters
            mu = returns.mean()
            sigma = returns.std()
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            random_returns = np.random.normal(
                mu * time_horizon, 
                sigma * np.sqrt(time_horizon), 
                num_simulations
            )
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var = -np.percentile(random_returns, var_percentile)
            
            # Convert to dollar amount
            portfolio_value = 100000
            var_dollar = var * portfolio_value
            
            return var_dollar
            
        except Exception as e:
            st.error(f"Error calculating Monte Carlo VaR: {str(e)}")
            return 0
    
    def calculate_garch_var(self, returns, confidence_level, time_horizon=1, p=1, q=1):
        """Calculate VaR using GARCH model"""
        try:
            if len(returns) < 100:  # Need sufficient data for GARCH
                return self.calculate_parametric_var(returns, confidence_level, time_horizon)
            
            # Fit GARCH model
            returns_pct = returns * 100  # Convert to percentage
            
            # Use arch library for GARCH fitting
            model = arch_model(returns_pct, vol='Garch', p=p, q=q, dist='normal')
            fitted_model = model.fit(disp='off')
            
            # Forecast volatility
            forecast = fitted_model.forecast(horizon=time_horizon)
            forecasted_vol = forecast.variance.iloc[-1, 0] / 100  # Convert back from percentage
            
            # Calculate conditional VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(returns.mean() * time_horizon + np.sqrt(forecasted_vol * time_horizon) * z_score)
            
            # Convert to dollar amount
            portfolio_value = 100000
            var_dollar = var * portfolio_value
            
            return var_dollar
            
        except Exception as e:
            st.warning(f"GARCH model failed, using parametric VaR: {str(e)}")
            return self.calculate_parametric_var(returns, confidence_level, time_horizon)
    
    def calculate_evt_var(self, returns, confidence_level, threshold_percentile=95):
        """Calculate VaR using Extreme Value Theory"""
        try:
            if len(returns) == 0:
                return 0
            
            # Define threshold for extreme values
            threshold = np.percentile(returns, threshold_percentile)
            
            # Extract exceedances
            exceedances = returns[returns > threshold] - threshold
            
            if len(exceedances) < 10:  # Need sufficient extreme values
                return self.calculate_historical_var(returns, confidence_level)
            
            # Fit Generalized Pareto Distribution (GPD)
            shape, loc, scale = stats.genpareto.fit(exceedances)
            
            # Calculate VaR using EVT
            n = len(returns)
            nu = len(exceedances)
            prob = (1 - confidence_level) * n / nu
            
            if shape != 0:
                var = threshold + (scale / shape) * (prob**(-shape) - 1)
            else:
                var = threshold + scale * np.log(prob)
            
            # Convert to dollar amount
            portfolio_value = 100000
            var_dollar = abs(var) * portfolio_value
            
            return var_dollar
            
        except Exception as e:
            st.warning(f"EVT model failed, using historical VaR: {str(e)}")
            return self.calculate_historical_var(returns, confidence_level)
    
    def calculate_expected_shortfall(self, returns, confidence_level):
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if len(returns) == 0:
                return 0
            
            # Calculate VaR first
            var_percentile = (1 - confidence_level) * 100
            var_threshold = np.percentile(returns, var_percentile)
            
            # Calculate Expected Shortfall
            tail_returns = returns[returns <= var_threshold]
            
            if len(tail_returns) == 0:
                return 0
            
            expected_shortfall = -tail_returns.mean()
            
            # Convert to dollar amount
            portfolio_value = 100000
            es_dollar = expected_shortfall * portfolio_value
            
            return es_dollar
            
        except Exception as e:
            st.error(f"Error calculating Expected Shortfall: {str(e)}")
            return 0
    
    def calculate_cornish_fisher_var(self, returns, confidence_level):
        """Calculate VaR with Cornish-Fisher adjustment for skewness and kurtosis"""
        try:
            if len(returns) == 0:
                return 0
            
            # Calculate moments
            mu = returns.mean()
            sigma = returns.std()
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Standard normal quantile
            z = stats.norm.ppf(1 - confidence_level)
            
            # Cornish-Fisher adjustment
            cf_adjustment = (
                z + 
                (z**2 - 1) * skewness / 6 + 
                (z**3 - 3*z) * kurtosis / 24 - 
                (2*z**3 - 5*z) * (skewness**2) / 36
            )
            
            # Calculate adjusted VaR
            var = -(mu + sigma * cf_adjustment)
            
            # Convert to dollar amount
            portfolio_value = 100000
            var_dollar = var * portfolio_value
            
            return var_dollar
            
        except Exception as e:
            st.error(f"Error calculating Cornish-Fisher VaR: {str(e)}")
            return self.calculate_parametric_var(returns, confidence_level)
    
    def calculate_marginal_var(self, returns_matrix, weights, confidence_level):
        """Calculate marginal VaR for portfolio components"""
        try:
            if returns_matrix.empty:
                return {}
            
            # Calculate portfolio returns
            portfolio_returns = returns_matrix.dot(weights)
            
            # Calculate portfolio VaR
            portfolio_var = self.calculate_parametric_var(portfolio_returns, confidence_level)
            
            # Calculate marginal VaR for each asset
            marginal_vars = {}
            epsilon = 0.001  # Small change for numerical derivative
            
            for i, asset in enumerate(returns_matrix.columns):
                # Create perturbed weights
                perturbed_weights = weights.copy()
                perturbed_weights[i] += epsilon
                
                # Renormalize weights
                perturbed_weights = perturbed_weights / perturbed_weights.sum()
                
                # Calculate VaR with perturbed weights
                perturbed_returns = returns_matrix.dot(perturbed_weights)
                perturbed_var = self.calculate_parametric_var(perturbed_returns, confidence_level)
                
                # Calculate marginal VaR
                marginal_var = (perturbed_var - portfolio_var) / epsilon
                marginal_vars[asset] = marginal_var
            
            return marginal_vars
            
        except Exception as e:
            st.error(f"Error calculating marginal VaR: {str(e)}")
            return {}
    
    def calculate_component_var(self, returns_matrix, weights, confidence_level):
        """Calculate component VaR for portfolio"""
        try:
            if returns_matrix.empty:
                return {}
            
            # Calculate marginal VaR
            marginal_vars = self.calculate_marginal_var(returns_matrix, weights, confidence_level)
            
            # Calculate component VaR
            component_vars = {}
            for i, asset in enumerate(returns_matrix.columns):
                component_var = marginal_vars[asset] * weights[i]
                component_vars[asset] = component_var
            
            return component_vars
            
        except Exception as e:
            st.error(f"Error calculating component VaR: {str(e)}")
            return {}