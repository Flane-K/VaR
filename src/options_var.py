import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import streamlit as st

class OptionsVaR:
    def __init__(self):
        pass
    
    def black_scholes_price(self, S, K, T, r, sigma, option_type='call'):
        """Calculate Black-Scholes option price"""
        try:
            if T <= 0:
                if option_type.lower() == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            
            return price
            
        except Exception as e:
            st.error(f"Error calculating Black-Scholes price: {str(e)}")
            return 0
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        try:
            if T <= 0:
                return {
                    'delta': 0,
                    'gamma': 0,
                    'theta': 0,
                    'vega': 0,
                    'rho': 0
                }
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type.lower() == 'call':
                delta = stats.norm.cdf(d1)
            else:
                delta = stats.norm.cdf(d1) - 1
            
            # Gamma
            gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            if option_type.lower() == 'call':
                theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
            else:
                theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365
            
            # Vega
            vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
            
            # Rho
            if option_type.lower() == 'call':
                rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            st.error(f"Error calculating Greeks: {str(e)}")
            return {}
    
    def calculate_options_var(self, S, K, T, r, sigma, option_type, method, confidence_level):
        """Calculate VaR for options using different methods"""
        try:
            if method == "Delta-Normal":
                return self._delta_normal_var(S, K, T, r, sigma, option_type, confidence_level)
            elif method == "Delta-Gamma":
                return self._delta_gamma_var(S, K, T, r, sigma, option_type, confidence_level)
            elif method == "Full Revaluation Monte Carlo":
                return self._monte_carlo_var(S, K, T, r, sigma, option_type, confidence_level)
            else:
                st.error(f"Unknown method: {method}")
                return {}
                
        except Exception as e:
            st.error(f"Error calculating options VaR: {str(e)}")
            return {}
    
    def _delta_normal_var(self, S, K, T, r, sigma, option_type, confidence_level):
        """Calculate VaR using Delta-Normal method"""
        try:
            # Calculate current option price and Greeks
            current_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            
            # Assume stock returns are normally distributed
            stock_volatility = sigma  # Daily volatility
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # Calculate VaR using delta approximation
            var = abs(greeks['delta'] * S * stock_volatility * z_score)
            
            return {
                'var': var,
                'current_price': current_price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'method': 'Delta-Normal'
            }
            
        except Exception as e:
            st.error(f"Error in Delta-Normal VaR: {str(e)}")
            return {}
    
    def _delta_gamma_var(self, S, K, T, r, sigma, option_type, confidence_level):
        """Calculate VaR using Delta-Gamma method"""
        try:
            # Calculate current option price and Greeks
            current_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            
            # Stock price change distribution
            stock_volatility = sigma
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # Calculate stock price change
            stock_change = S * stock_volatility * z_score
            
            # Delta-Gamma approximation
            delta_pnl = greeks['delta'] * stock_change
            gamma_pnl = 0.5 * greeks['gamma'] * (stock_change ** 2)
            
            total_pnl = delta_pnl + gamma_pnl
            var = abs(total_pnl)
            
            return {
                'var': var,
                'current_price': current_price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'delta_pnl': delta_pnl,
                'gamma_pnl': gamma_pnl,
                'method': 'Delta-Gamma'
            }
            
        except Exception as e:
            st.error(f"Error in Delta-Gamma VaR: {str(e)}")
            return {}
    
    def _monte_carlo_var(self, S, K, T, r, sigma, option_type, confidence_level, num_simulations=10000):
        """Calculate VaR using Monte Carlo full revaluation"""
        try:
            # Current option price
            current_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            
            # Generate random stock price scenarios
            np.random.seed(42)  # For reproducibility
            
            # Simulate stock prices after 1 day
            dt = 1/365  # 1 day
            random_shocks = np.random.normal(0, 1, num_simulations)
            
            # Geometric Brownian Motion
            stock_prices = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks)
            
            # Calculate option prices for each scenario
            option_prices = []
            for stock_price in stock_prices:
                new_T = max(T - dt, 0)  # Time decay
                option_price = self.black_scholes_price(stock_price, K, new_T, r, sigma, option_type)
                option_prices.append(option_price)
            
            option_prices = np.array(option_prices)
            
            # Calculate P&L
            pnl = option_prices - current_price
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var = -np.percentile(pnl, var_percentile)
            
            return {
                'var': var,
                'current_price': current_price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'simulated_pnl': pnl,
                'method': 'Monte Carlo'
            }
            
        except Exception as e:
            st.error(f"Error in Monte Carlo VaR: {str(e)}")
            return {}
    
    def calculate_portfolio_options_var(self, portfolio, confidence_level, method="Delta-Gamma"):
        """Calculate VaR for a portfolio of options"""
        try:
            total_var = 0
            portfolio_details = []
            
            for option in portfolio:
                # Extract option parameters
                S = option['spot_price']
                K = option['strike_price']
                T = option['time_to_expiry']
                r = option['risk_free_rate']
                sigma = option['volatility']
                option_type = option['option_type']
                quantity = option['quantity']
                
                # Calculate individual option VaR
                option_var = self.calculate_options_var(S, K, T, r, sigma, option_type, method, confidence_level)
                
                # Scale by quantity
                scaled_var = option_var['var'] * abs(quantity)
                total_var += scaled_var
                
                portfolio_details.append({
                    'option': f"{option_type} {K}",
                    'quantity': quantity,
                    'var': scaled_var,
                    'details': option_var
                })
            
            return {
                'total_var': total_var,
                'portfolio_details': portfolio_details,
                'method': method
            }
            
        except Exception as e:
            st.error(f"Error calculating portfolio options VaR: {str(e)}")
            return {}
    
    def calculate_implied_volatility(self, S, K, T, r, market_price, option_type='call'):
        """Calculate implied volatility using Brent's method"""
        try:
            def objective(sigma):
                theoretical_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
                return (theoretical_price - market_price) ** 2
            
            # Search for implied volatility
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            
            if result.success:
                return result.x
            else:
                return 0.2  # Default volatility if optimization fails
                
        except Exception as e:
            st.error(f"Error calculating implied volatility: {str(e)}")
            return 0.2
    
    def calculate_option_sensitivities(self, S, K, T, r, sigma, option_type='call'):
        """Calculate comprehensive option sensitivities"""
        try:
            base_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            
            # Price sensitivities
            sensitivities = {}
            
            # Spot price sensitivity (Delta)
            spot_up = self.black_scholes_price(S * 1.01, K, T, r, sigma, option_type)
            spot_down = self.black_scholes_price(S * 0.99, K, T, r, sigma, option_type)
            sensitivities['delta_numerical'] = (spot_up - spot_down) / (0.02 * S)
            
            # Volatility sensitivity (Vega)
            vol_up = self.black_scholes_price(S, K, T, r, sigma * 1.01, option_type)
            vol_down = self.black_scholes_price(S, K, T, r, sigma * 0.99, option_type)
            sensitivities['vega_numerical'] = (vol_up - vol_down) / (0.02 * sigma)
            
            # Time sensitivity (Theta)
            if T > 1/365:  # More than 1 day
                time_decay = self.black_scholes_price(S, K, T - 1/365, r, sigma, option_type)
                sensitivities['theta_numerical'] = time_decay - base_price
            else:
                sensitivities['theta_numerical'] = 0
            
            # Interest rate sensitivity (Rho)
            rate_up = self.black_scholes_price(S, K, T, r + 0.01, sigma, option_type)
            rate_down = self.black_scholes_price(S, K, T, r - 0.01, sigma, option_type)
            sensitivities['rho_numerical'] = (rate_up - rate_down) / 0.02
            
            return sensitivities
            
        except Exception as e:
            st.error(f"Error calculating option sensitivities: {str(e)}")
            return {}
    
    def create_options_risk_report(self, S, K, T, r, sigma, option_type, confidence_level):
        """Create comprehensive options risk report"""
        try:
            report = {}
            
            # Basic option information
            report['option_details'] = {
                'spot_price': S,
                'strike_price': K,
                'time_to_expiry': T,
                'risk_free_rate': r,
                'volatility': sigma,
                'option_type': option_type
            }
            
            # Option price and Greeks
            current_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            
            report['pricing'] = {
                'current_price': current_price,
                'greeks': greeks
            }
            
            # VaR calculations
            var_methods = ["Delta-Normal", "Delta-Gamma", "Full Revaluation Monte Carlo"]
            report['var_results'] = {}
            
            for method in var_methods:
                var_result = self.calculate_options_var(S, K, T, r, sigma, option_type, method, confidence_level)
                report['var_results'][method] = var_result
            
            # Sensitivity analysis
            report['sensitivities'] = self.calculate_option_sensitivities(S, K, T, r, sigma, option_type)
            
            return report
            
        except Exception as e:
            st.error(f"Error creating options risk report: {str(e)}")
            return {}