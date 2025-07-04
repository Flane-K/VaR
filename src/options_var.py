import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, Optional, Dict, Any
import yfinance as yf
from datetime import datetime, timedelta

class OptionsVaR:
    """
    A comprehensive class for calculating Value at Risk (VaR) for options portfolios
    using various methodologies including Delta-Gamma parametric approach.
    """
    
    def __init__(self):
        self.cache = {}
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """Calculate Black-Scholes option price."""
        try:
            if T <= 0:
                if option_type.lower() == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)

            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            return max(price, 0)
        except:
            return 0
    
    def black_scholes_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate Black-Scholes Greeks (Delta, Gamma, Theta, Vega) for an option.
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        
        Returns:
        Dictionary with delta, gamma, theta, vega
        """
        try:
            if T <= 0:
                # Option has expired
                if option_type.lower() == 'call':
                    delta = 1.0 if S > K else 0.0
                else:  # put
                    delta = -1.0 if S < K else 0.0
                return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta calculation
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:  # put
                delta = norm.cdf(d1) - 1
            
            # Gamma calculation (same for calls and puts)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta calculation
            if option_type.lower() == 'call':
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:  # put
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega calculation (same for calls and puts)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def get_stock_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch stock data with caching."""
        cache_key = f"{symbol}_{period}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            self.cache[cache_key] = data
            return data
        except Exception as e:
            raise ValueError(f"Error fetching data for {symbol}: {str(e)}")
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate daily returns from price data."""
        if 'Close' not in data.columns:
            raise ValueError("Price data must contain 'Close' column")
        
        returns = data['Close'].pct_change().dropna()
        if returns.empty:
            raise ValueError("Unable to calculate returns from price data")
        
        return returns
    
    def calculate_options_var_comprehensive(self, option_returns: pd.Series, confidence_level: float, method: str = 'historical') -> float:
        """Calculate VaR for options using various methods"""
        try:
            if option_returns.empty:
                return 0

            if method == 'historical':
                var_percentile = (1 - confidence_level) * 100
                var = np.percentile(option_returns, var_percentile)
            elif method == 'parametric':
                mean = option_returns.mean()
                std = option_returns.std()
                var = mean - norm.ppf(confidence_level) * std
            elif method == 'monte_carlo':
                # Simple Monte Carlo simulation
                simulated_returns = np.random.normal(option_returns.mean(), 
                                                   option_returns.std(), 10000)
                var_percentile = (1 - confidence_level) * 100
                var = np.percentile(simulated_returns, var_percentile)
            else:
                var = np.percentile(option_returns, (1 - confidence_level) * 100)

            return abs(var)
        except Exception as e:
            return 0
    
    def calculate_delta_gamma_var(self, S: float, K: float, T: float, r: float, sigma: float, 
                                 option_type: str, confidence_level: float, underlying_returns: pd.Series, 
                                 portfolio_value: float = 100000) -> float:
        """Calculate Delta-Gamma VaR for options using Taylor expansion"""
        try:
            # Calculate Greeks
            greeks = self.black_scholes_greeks(S, K, T, r, sigma, option_type)
            delta = greeks['delta']
            gamma = greeks['gamma']
            
            # Get underlying return statistics
            if len(underlying_returns) == 0:
                return 0
                
            underlying_vol = underlying_returns.std()
            
            # Calculate VaR percentile
            z_score = norm.ppf(confidence_level)
            
            # Underlying price change for VaR calculation
            delta_S = S * underlying_vol * z_score
            
            # Delta-Gamma approximation for option price change
            # ΔP ≈ Δ × ΔS + 0.5 × Γ × (ΔS)²
            option_price_change = delta * delta_S + 0.5 * gamma * (delta_S ** 2)
            
            # Convert to portfolio VaR
            var_result = abs(option_price_change * portfolio_value / S)
            
            return var_result
            
        except Exception as e:
            return 0
    
    def calculate_portfolio_var(self, symbol: str, spot_price: float, strike_price: float, 
                               time_to_expiry: float, risk_free_rate: float, volatility: float,
                               option_type: str, quantity: int, confidence_level: float,
                               var_model: str, lookback_days: int = 252) -> Dict[str, Any]:
        """
        Main method to calculate VaR for options portfolio using specified method.
        Returns a dictionary with all relevant metrics.
        """
        try:
            # Calculate current option price and portfolio value
            current_option_price = self.black_scholes_price(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            portfolio_value = current_option_price * abs(quantity)
            
            # Get historical data
            data = self.get_stock_data(symbol, period="2y")
            returns = self.calculate_returns(data)
            
            # Ensure we have enough data
            if len(returns) < max(lookback_days, 30):
                lookback_days = min(len(returns), 30)
            
            recent_returns = returns.tail(lookback_days)
            
            if len(recent_returns) == 0:
                return self._create_empty_result()
            
            # Calculate option returns for VaR calculation
            option_returns = []
            for ret in recent_returns:
                new_S = spot_price * (1 + ret)
                new_option_price = self.black_scholes_price(new_S, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)
                option_pnl = (new_option_price - current_option_price) * quantity
                option_returns.append(option_pnl)
            
            option_returns = pd.Series(option_returns)
            
            # Calculate VaR based on selected method
            if var_model == "Historical Simulation" or var_model == "Historic Simulation":
                var_dollar = self._historical_simulation_var(
                    recent_returns, spot_price, strike_price, time_to_expiry,
                    risk_free_rate, volatility, option_type, confidence_level,
                    current_option_price, quantity
                )
            
            elif var_model == "Parametric (Delta-Normal)":
                var_dollar = self._delta_normal_var(
                    recent_returns, spot_price, strike_price, time_to_expiry,
                    risk_free_rate, volatility, option_type, confidence_level,
                    current_option_price, quantity
                )
            
            elif var_model == "Parametric (Delta-Gamma)" or var_model == "Delta-Gamma Parametric":
                var_dollar = self._delta_gamma_var(
                    recent_returns, spot_price, strike_price, time_to_expiry,
                    risk_free_rate, volatility, option_type, confidence_level, quantity
                )
            
            elif var_model == "Monte Carlo":
                var_dollar = self._monte_carlo_var(
                    recent_returns, spot_price, strike_price, time_to_expiry,
                    risk_free_rate, volatility, option_type, confidence_level,
                    current_option_price, quantity
                )
            
            else:
                # Default to historical simulation
                var_dollar = self._historical_simulation_var(
                    recent_returns, spot_price, strike_price, time_to_expiry,
                    risk_free_rate, volatility, option_type, confidence_level,
                    current_option_price, quantity
                )
            
            # Calculate Expected Shortfall
            expected_shortfall = self._calculate_expected_shortfall(
                recent_returns, spot_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, option_type, confidence_level,
                current_option_price, quantity, var_model
            )
            
            # Calculate Greeks for additional information
            greeks = self.black_scholes_greeks(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            
            return {
                'var_dollar': var_dollar,
                'var_percentage': (var_dollar / portfolio_value * 100) if portfolio_value > 0 else 0,
                'expected_shortfall': expected_shortfall,
                'portfolio_value': portfolio_value,
                'option_price': current_option_price,
                'delta': greeks['delta'] * quantity,
                'gamma': greeks['gamma'] * quantity,
                'theta': greeks['theta'] * quantity,
                'vega': greeks['vega'] * quantity,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'var_dollar': 0.0,
                'var_percentage': 0.0,
                'expected_shortfall': 0.0,
                'portfolio_value': 0.0,
                'option_price': 0.0,
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result dictionary."""
        return {
            'var_dollar': 0.0,
            'var_percentage': 0.0,
            'expected_shortfall': 0.0,
            'portfolio_value': 0.0,
            'option_price': 0.0,
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'success': True,
            'error': 'Insufficient data'
        }
    
    def _historical_simulation_var(self, returns: pd.Series, S: float, K: float, T: float,
                                 r: float, sigma: float, option_type: str, 
                                 confidence_level: float, current_price: float, quantity: int) -> float:
        """Calculate VaR using historical simulation method."""
        portfolio_pnl = []
        current_portfolio_value = current_price * abs(quantity)
        
        for ret in returns:
            # Calculate new underlying price
            new_S = S * (1 + ret)
            
            # Calculate new option price
            new_option_price = self.black_scholes_price(new_S, K, T, r, sigma, option_type)
            
            # Calculate portfolio P&L
            option_pnl = (new_option_price - current_price) * quantity
            portfolio_pnl.append(option_pnl)
        
        if not portfolio_pnl:
            return 0.0
        
        # Calculate VaR as the percentile of losses
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(portfolio_pnl, var_percentile)
        
        # Return absolute value of loss (positive number)
        return abs(min(var_value, 0))
    
    def _delta_normal_var(self, returns: pd.Series, S: float, K: float, T: float,
                         r: float, sigma: float, option_type: str, 
                         confidence_level: float, current_price: float, quantity: int) -> float:
        """Calculate VaR using delta-normal parametric method."""
        # Calculate delta
        greeks = self.black_scholes_greeks(S, K, T, r, sigma, option_type)
        delta = greeks['delta']
        
        if abs(delta) < 1e-10:  # Delta is essentially zero
            return 0.0
        
        # Calculate underlying return statistics
        returns_std = returns.std()
        if np.isnan(returns_std) or returns_std == 0:
            return 0.0
        
        # Calculate VaR using delta approximation
        z_score = norm.ppf(confidence_level)  # Use confidence_level directly for upper tail
        
        # Portfolio delta exposure
        portfolio_delta = delta * quantity
        
        # Underlying VaR (dollar amount)
        underlying_var = z_score * returns_std * S
        
        # Option portfolio VaR using delta approximation
        option_var = abs(portfolio_delta * underlying_var)
        
        return option_var
    
    def _delta_gamma_var(self, returns: pd.Series, S: float, K: float, T: float,
                        r: float, sigma: float, option_type: str, 
                        confidence_level: float, quantity: int) -> float:
        """Calculate VaR using delta-gamma parametric method."""
        # Calculate Greeks
        greeks = self.black_scholes_greeks(S, K, T, r, sigma, option_type)
        delta = greeks['delta']
        gamma = greeks['gamma']
        
        # Calculate underlying return statistics
        returns_std = returns.std()
        if np.isnan(returns_std) or returns_std == 0:
            return 0.0
        
        # Portfolio Greeks
        portfolio_delta = delta * quantity
        portfolio_gamma = gamma * quantity
        
        # Z-score for confidence level
        z_score = norm.ppf(confidence_level)
        
        # Underlying price change at confidence level
        underlying_change = z_score * returns_std * S
        
        # Delta-Gamma approximation for portfolio P&L
        # ΔP ≈ Delta × ΔS + 0.5 × Gamma × (ΔS)²
        delta_component = portfolio_delta * underlying_change
        gamma_component = 0.5 * portfolio_gamma * (underlying_change ** 2)
        
        # Total P&L change
        total_pnl_change = delta_component + gamma_component
        
        # For VaR, we want the loss amount (positive number)
        # If the position is long, negative P&L is a loss
        # If the position is short, positive P&L is a loss
        if quantity > 0:  # Long position
            var_value = abs(min(total_pnl_change, 0))
        else:  # Short position
            var_value = abs(max(total_pnl_change, 0))
        
        return var_value
    
    def _monte_carlo_var(self, returns: pd.Series, S: float, K: float, T: float,
                        r: float, sigma: float, option_type: str, 
                        confidence_level: float, current_price: float, quantity: int,
                        num_simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        returns_mean = returns.mean()
        returns_std = returns.std()
        
        # Handle edge cases
        if np.isnan(returns_std) or returns_std == 0:
            return 0.0
        
        if np.isnan(returns_mean):
            returns_mean = 0.0
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(returns_mean, returns_std, num_simulations)
        
        portfolio_pnl = []
        
        for sim_return in simulated_returns:
            # Calculate new underlying price
            new_S = S * (1 + sim_return)
            
            # Calculate new option price
            new_option_price = self.black_scholes_price(new_S, K, T, r, sigma, option_type)
            
            # Calculate portfolio P&L
            option_pnl = (new_option_price - current_price) * quantity
            portfolio_pnl.append(option_pnl)
        
        if not portfolio_pnl:
            return 0.0
        
        # Calculate VaR as the percentile of losses
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(portfolio_pnl, var_percentile)
        
        # Return absolute value of loss (positive number)
        return abs(min(var_value, 0))
    
    def _calculate_expected_shortfall(self, returns: pd.Series, S: float, K: float, T: float,
                                    r: float, sigma: float, option_type: str,
                                    confidence_level: float, current_price: float, 
                                    quantity: int, method: str) -> float:
        """Calculate Expected Shortfall (Conditional VaR) for options."""
        try:
            portfolio_pnl = []
            
            if method == "Monte Carlo":
                # Use Monte Carlo simulation for ES calculation
                returns_mean = returns.mean()
                returns_std = returns.std()
                
                if np.isnan(returns_std) or returns_std == 0:
                    return 0.0
                
                if np.isnan(returns_mean):
                    returns_mean = 0.0
                
                np.random.seed(42)
                simulated_returns = np.random.normal(returns_mean, returns_std, 10000)
                
                for sim_return in simulated_returns:
                    new_S = S * (1 + sim_return)
                    new_option_price = self.black_scholes_price(new_S, K, T, r, sigma, option_type)
                    option_pnl = (new_option_price - current_price) * quantity
                    portfolio_pnl.append(option_pnl)
            
            else:
                # Use historical simulation for ES calculation
                for ret in returns:
                    new_S = S * (1 + ret)
                    new_option_price = self.black_scholes_price(new_S, K, T, r, sigma, option_type)
                    option_pnl = (new_option_price - current_price) * quantity
                    portfolio_pnl.append(option_pnl)
            
            if not portfolio_pnl:
                return 0.0
            
            # Find the VaR threshold
            var_percentile = (1 - confidence_level) * 100
            var_threshold = np.percentile(portfolio_pnl, var_percentile)
            
            # Calculate expected shortfall as the mean of losses beyond VaR
            tail_losses = [pnl for pnl in portfolio_pnl if pnl <= var_threshold]
            
            if not tail_losses:
                return 0.0
            
            expected_shortfall = abs(np.mean(tail_losses))
            return expected_shortfall
            
        except Exception as e:
            print(f"Error in Expected Shortfall calculation: {str(e)}")
            return 0.0
    
    def perform_stress_test(self, symbol: str, spot_price: float, strike_price: float,
                           time_to_expiry: float, risk_free_rate: float, volatility: float,
                           option_type: str, quantity: int, stress_scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Perform stress testing on the options portfolio.
        
        Parameters:
        stress_scenarios: Dictionary with scenario names and corresponding underlying price changes (as percentages)
        
        Returns:
        Dictionary with scenario names and corresponding P&L values
        """
        current_option_price = self.black_scholes_price(
            spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
        )
        
        stress_results = {}
        
        for scenario_name, price_change_pct in stress_scenarios.items():
            try:
                # Calculate new underlying price
                new_spot_price = spot_price * (1 + price_change_pct / 100)
                
                # Calculate new option price
                new_option_price = self.black_scholes_price(
                    new_spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
                )
                
                # Calculate P&L
                pnl = (new_option_price - current_option_price) * quantity
                stress_results[scenario_name] = pnl
                
            except Exception as e:
                stress_results[scenario_name] = 0.0
        
        return stress_results
    
    def backtest_var(self, symbol: str, spot_price: float, strike_price: float,
                    time_to_expiry: float, risk_free_rate: float, volatility: float,
                    option_type: str, quantity: int, confidence_level: float,
                    var_model: str, backtest_days: int = 252) -> Dict[str, Any]:
        """
        Perform VaR backtesting to validate the model accuracy.
        
        Returns:
        Dictionary with backtesting results including violation rate and statistics
        """
        try:
            # Get sufficient historical data for backtesting
            required_period = max(backtest_days + 252, 504)  # Ensure enough data
            data = self.get_stock_data(symbol, period="5y")  # Get more data for backtesting
            returns = self.calculate_returns(data)
            
            if len(returns) < required_period:
                return {
                    'success': False,
                    'error': f'Insufficient data for backtesting. Need {required_period} days, got {len(returns)}'
                }
            
            violations = 0
            total_tests = 0
            var_estimates = []
            actual_losses = []
            
            # Perform rolling VaR calculation and backtesting
            for i in range(252, min(len(returns), backtest_days + 252)):
                try:
                    # Use historical data up to day i-1 for VaR calculation
                    historical_returns = returns.iloc[i-252:i]
                    
                    # Get the actual return for day i
                    actual_return = returns.iloc[i]
                    
                    # Calculate VaR using historical data
                    current_price = self.black_scholes_price(
                        spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
                    )
                    
                    if var_model == "Historical Simulation":
                        var_estimate = self._historical_simulation_var(
                            historical_returns, spot_price, strike_price, time_to_expiry,
                            risk_free_rate, volatility, option_type, confidence_level,
                            current_price, quantity
                        )
                    elif var_model == "Parametric (Delta-Normal)":
                        var_estimate = self._delta_normal_var(
                            historical_returns, spot_price, strike_price, time_to_expiry,
                            risk_free_rate, volatility, option_type, confidence_level,
                            current_price, quantity
                        )
                    elif var_model == "Parametric (Delta-Gamma)":
                        var_estimate = self._delta_gamma_var(
                            historical_returns, spot_price, strike_price, time_to_expiry,
                            risk_free_rate, volatility, option_type, confidence_level, quantity
                        )
                    elif var_model == "Monte Carlo":
                        var_estimate = self._monte_carlo_var(
                            historical_returns, spot_price, strike_price, time_to_expiry,
                            risk_free_rate, volatility, option_type, confidence_level,
                            current_price, quantity
                        )
                    
                    # Calculate actual P&L for day i
                    new_spot = spot_price * (1 + actual_return)
                    new_option_price = self.black_scholes_price(
                        new_spot, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
                    )
                    actual_pnl = (new_option_price - current_price) * quantity
                    actual_loss = abs(min(actual_pnl, 0))  # Loss is positive
                    
                    # Check for VaR violation
                    if actual_loss > var_estimate:
                        violations += 1
                    
                    var_estimates.append(var_estimate)
                    actual_losses.append(actual_loss)
                    total_tests += 1
                    
                except Exception as e:
                    continue
            
            if total_tests == 0:
                return {
                    'success': False,
                    'error': 'No valid backtesting data points'
                }
            
            violation_rate = violations / total_tests
            expected_violation_rate = 1 - confidence_level
            
            # Calculate additional statistics
            avg_var = np.mean(var_estimates) if var_estimates else 0
            avg_loss = np.mean(actual_losses) if actual_losses else 0
            max_loss = max(actual_losses) if actual_losses else 0
            
            return {
                'success': True,
                'violation_rate': violation_rate,
                'expected_violation_rate': expected_violation_rate,
                'total_violations': violations,
                'total_tests': total_tests,
                'avg_var_estimate': avg_var,
                'avg_actual_loss': avg_loss,
                'max_actual_loss': max_loss,
                'model_accuracy': abs(violation_rate - expected_violation_rate) < 0.05,  # Within 5% tolerance
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
