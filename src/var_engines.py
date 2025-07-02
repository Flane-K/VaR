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
    
    def calculate_parametric_var(self, returns, confidence_level, time_horizon=1, cornish_fisher: bool = False):
        """
        Calculate VaR using parametric (Delta-Normal) method.
        Optionally apply Cornish-Fisher adjustment.
        """
        try:
            if len(returns) == 0:
                return 0
            
            if cornish_fisher:
                # If Cornish-Fisher adjustment is requested, use that method
                return self.calculate_cornish_fisher_var(returns, confidence_level)
            
            # Original Parametric VaR calculation
            mu = returns.mean()
            sigma = returns.std()
            
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mu * time_horizon + sigma * np.sqrt(time_horizon) * z_score)
            
            # Convert to dollar amount (assuming $100,000 portfolio)
            portfolio_value = 100000
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
                st.warning("Insufficient data for GARCH model, using parametric VaR")
                return self.calculate_parametric_var(returns, confidence_level, time_horizon, cornish_fisher=False)
            
            # Fit GARCH model
            returns_pct = returns * 100  # Convert to percentage
            
            # Use arch library for GARCH fitting
            model = arch_model(returns_pct, vol='Garch', p=p, q=q, dist='normal')
            fitted_model = model.fit(disp='off')
            
            # Forecast volatility
            forecast = fitted_model.forecast(horizon=time_horizon)
            # Get the forecasted variance and convert to standard deviation
            forecasted_variance = forecast.variance.iloc[-1, 0]
            forecasted_vol = np.sqrt(forecasted_variance) / 100  # Convert back from percentage
            
            # Calculate conditional VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            # Use the mean of the returns from historical data, and the forecasted volatility
            var = -(returns.mean() * time_horizon + forecasted_vol * np.sqrt(time_horizon) * z_score)
            
            # Convert to dollar amount
            portfolio_value = 100000
            var_dollar = var * portfolio_value
            
            return var_dollar
            
        except Exception as e:
            st.warning(f"GARCH model failed, using parametric VaR: {str(e)}")
            return self.calculate_parametric_var(returns, confidence_level, time_horizon, cornish_fisher=False)
    
    def calculate_evt_var(self, returns, confidence_level, threshold_percentile=95):
        """Calculate VaR using Extreme Value Theory"""
        try:
            if len(returns) == 0:
                return 0
            
            # Define threshold for extreme values (for losses, this would be a low percentile)
            losses = -returns # Convert returns to losses
            
            # Threshold for extreme losses (e.g., 95th percentile of losses)
            threshold = np.percentile(losses, threshold_percentile)
            
            # Extract exceedances (values above the threshold)
            exceedances = losses[losses > threshold] - threshold
            
            if len(exceedances) < 10:  # Need sufficient extreme values for GPD fitting
                st.warning("Not enough extreme values for EVT, falling back to historical VaR.")
                return self.calculate_historical_var(returns, confidence_level)
            
            # Fit Generalized Pareto Distribution (GPD)
            shape, loc, scale = stats.genpareto.fit(exceedances, loc=0) # Fix loc to 0 for exceedances
            
            # Calculate VaR using EVT formula
            n = len(returns) # Total number of observations
            nu = len(exceedances) # Number of exceedances
            
            # Probability of exceeding the VaR for the given confidence level
            prob_exceed = (1 - confidence_level) 
            
            # Calculate VaR using the GPD parameters and the formula
            if shape != 0:
                var = threshold + (scale / shape) * ((n / nu * prob_exceed)**(-shape) - 1)
            else: # Exponential distribution case (shape = 0)
                var = threshold + scale * np.log(n / nu * prob_exceed)
            
            # Convert to dollar amount
            portfolio_value = 100000
            var_dollar = var * portfolio_value
            
            return var_dollar
            
        except Exception as e:
            st.warning(f"EVT model failed, using historical VaR: {str(e)}")
            return self.calculate_historical_var(returns, confidence_level)
    
    def calculate_expected_shortfall(self, returns, confidence_level):
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if len(returns) == 0:
                return 0
            
            # Use negative returns (losses) for percentile calculation for ES.
            losses = -returns
            
            var_percentile_for_es = confidence_level * 100
            var_threshold = np.percentile(losses, var_percentile_for_es)
            
            # Calculate Expected Shortfall
            # ES is the average of losses that are worse than the VaR threshold
            tail_losses = losses[losses >= var_threshold]
            
            if len(tail_losses) == 0:
                return 0
            
            expected_shortfall = tail_losses.mean()
            
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
            
            # Standard normal quantile for the desired confidence level
            alpha = 1 - confidence_level
            z_alpha = stats.norm.ppf(alpha) # This will be a negative value for common alpha (e.g., 0.05)
            
            # Cornish-Fisher adjusted quantile
            cf_adjusted_quantile = (
                z_alpha +
                (skewness / 6) * (z_alpha**2 - 1) +
                (kurtosis / 24) * (z_alpha**3 - 3 * z_alpha) -
                ((skewness**2) / 36) * (2 * z_alpha**3 - 5 * z_alpha)
            )
            
            # Calculate adjusted VaR (as a return percentage)
            var = -(mu + sigma * cf_adjusted_quantile)
            
            # Convert to dollar amount
            portfolio_value = 100000
            var_dollar = var * portfolio_value
            
            return var_dollar
            
        except Exception as e:
            st.error(f"Error calculating Cornish-Fisher VaR: {str(e)}")
            # Fallback to parametric VaR if Cornish-Fisher fails
            return self.calculate_parametric_var(returns, confidence_level, cornish_fisher=False)
    
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
            
            # Ensure weights are a numpy array for direct manipulation
            weights_arr = np.array(weights)

            for i, asset in enumerate(returns_matrix.columns):
                # Create perturbed weights
                perturbed_weights_arr = weights_arr.copy()
                perturbed_weights_arr[i] += epsilon
                
                # Renormalize weights
                perturbed_weights_arr = perturbed_weights_arr / perturbed_weights_arr.sum()
                
                # Calculate VaR with perturbed weights
                perturbed_returns = returns_matrix.dot(perturbed_weights_arr)
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
            # Ensure weights are iterable in the same order as columns
            if isinstance(weights, dict):
                weights_list = [weights.get(col, 0) for col in returns_matrix.columns]
            else: # Assume it's a list or array
                weights_list = weights
            
            for i, asset in enumerate(returns_matrix.columns):
                if asset in marginal_vars:
                    component_var = marginal_vars[asset] * weights_list[i]
                    component_vars[asset] = component_var
                else:
                    component_vars[asset] = 0
            
            return component_vars
            
        except Exception as e:
            st.error(f"Error calculating component VaR: {str(e)}")
            return {}