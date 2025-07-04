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
    
    def calculate_options_var(self, spot_price: float, strike_price: float, time_to_expiry: float,
                             risk_free_rate: float, volatility: float, option_type: str,
                             method: str, confidence_level: float, quantity: int = 1) -> Dict[str, Any]:
        """
        Calculate VaR for options using specified method.
        
        Parameters:
        spot_price: Current underlying price
        strike_price: Option strike price
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free rate
        volatility: Implied volatility
        option_type: 'call' or 'put'
        method: VaR calculation method
        confidence_level: Confidence level (e.g., 0.95)
        quantity: Number of option contracts
        
        Returns:
        Dictionary with VaR and other metrics
        """
        try:
            # Calculate current option price
            current_price = self.black_scholes_price(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            
            # Calculate Greeks
            greeks = self.black_scholes_greeks(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            
            # Portfolio value
            portfolio_value = current_price * abs(quantity)
            
            if method == "Delta-Normal":
                var_result = self._delta_normal_var(
                    spot_price, strike_price, time_to_expiry, risk_free_rate,
                    volatility, option_type, confidence_level, quantity
                )
            elif method == "Delta-Gamma" or method == "Parametric (Delta-Gamma)":
                var_result = self._delta_gamma_var(
                    spot_price, strike_price, time_to_expiry, risk_free_rate,
                    volatility, option_type, confidence_level, quantity
                )
            elif method == "Monte Carlo":
                var_result = self._monte_carlo_var(
                    spot_price, strike_price, time_to_expiry, risk_free_rate,
                    volatility, option_type, confidence_level, quantity
                )
            elif method == "Historical Simulation":
                var_result = self._historical_simulation_var(
                    spot_price, strike_price, time_to_expiry, risk_free_rate,
                    volatility, option_type, confidence_level, quantity
                )
            else:
                # Default to Delta-Normal
                var_result = self._delta_normal_var(
                    spot_price, strike_price, time_to_expiry, risk_free_rate,
                    volatility, option_type, confidence_level, quantity
                )
            
            # Calculate Expected Shortfall
            expected_shortfall = var_result * 1.2  # Approximation for options
            
            return {
                'var': var_result,
                'expected_shortfall': expected_shortfall,
                'portfolio_value': portfolio_value,
                'option_price': current_price,
                'delta': greeks['delta'] * quantity,
                'gamma': greeks['gamma'] * quantity,
                'theta': greeks['theta'] * quantity,
                'vega': greeks['vega'] * quantity,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'var': 0.0,
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
    
    def calculate_options_var_historical(self, spot_price: float, strike_price: float,
                                       time_to_expiry: float, risk_free_rate: float,
                                       volatility: float, option_type: str,
                                       historical_returns: pd.Series, confidence_level: float,
                                       quantity: int = 1) -> Dict[str, Any]:
        """Calculate options VaR using historical simulation with actual market data."""
        try:
            current_price = self.black_scholes_price(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            
            portfolio_pnl = []
            
            for ret in historical_returns:
                # Calculate new underlying price
                new_spot = spot_price * (1 + ret)
                
                # Calculate new option price
                new_price = self.black_scholes_price(
                    new_spot, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
                )
                
                # Calculate P&L
                pnl = (new_price - current_price) * quantity
                portfolio_pnl.append(pnl)
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(portfolio_pnl, var_percentile)
            
            return {
                'var': abs(min(var_value, 0)),
                'portfolio_pnl': portfolio_pnl,
                'success': True
            }
            
        except Exception as e:
            return {
                'var': 0.0,
                'portfolio_pnl': [],
                'success': False,
                'error': str(e)
            }
    
    def _delta_normal_var(self, spot_price: float, strike_price: float, time_to_expiry: float,
                         risk_free_rate: float, volatility: float, option_type: str,
                         confidence_level: float, quantity: int) -> float:
        """Calculate VaR using Delta-Normal method."""
        try:
            # Calculate delta
            greeks = self.black_scholes_greeks(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            delta = greeks['delta']
            
            # Portfolio delta
            portfolio_delta = delta * quantity
            
            # Calculate VaR using delta approximation
            z_score = norm.ppf(confidence_level)
            daily_vol = volatility / np.sqrt(252)
            underlying_var = z_score * daily_vol * spot_price
            
            option_var = abs(portfolio_delta * underlying_var)
            
            return option_var
            
        except Exception as e:
            return 0.0
    
    def _delta_gamma_var(self, spot_price: float, strike_price: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float, option_type: str,
                        confidence_level: float, quantity: int) -> float:
        """Calculate VaR using Delta-Gamma parametric method."""
        try:
            # Calculate Greeks
            greeks = self.black_scholes_greeks(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            delta = greeks['delta']
            gamma = greeks['gamma']
            
            # Portfolio Greeks
            portfolio_delta = delta * quantity
            portfolio_gamma = gamma * quantity
            
            # Calculate underlying price change at confidence level
            z_score = norm.ppf(confidence_level)
            daily_vol = volatility / np.sqrt(252)
            underlying_change = z_score * daily_vol * spot_price
            
            # Delta-Gamma approximation for portfolio P&L
            # ΔP ≈ Delta × ΔS + 0.5 × Gamma × (ΔS)²
            delta_component = portfolio_delta * underlying_change
            gamma_component = 0.5 * portfolio_gamma * (underlying_change ** 2)
            
            total_change = delta_component + gamma_component
            
            # For VaR, we want the loss amount (positive number)
            if quantity > 0:  # Long position
                var_value = abs(min(total_change, 0))
            else:  # Short position
                var_value = abs(max(total_change, 0))
            
            return var_value
            
        except Exception as e:
            return 0.0
    
    def _monte_carlo_var(self, spot_price: float, strike_price: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float, option_type: str,
                        confidence_level: float, quantity: int, num_simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        try:
            current_price = self.black_scholes_price(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            
            # Generate random price movements
            np.random.seed(42)
            daily_vol = volatility / np.sqrt(252)
            random_returns = np.random.normal(0, daily_vol, num_simulations)
            
            portfolio_pnl = []
            
            for ret in random_returns:
                # Calculate new underlying price
                new_spot = spot_price * (1 + ret)
                
                # Calculate new option price
                new_price = self.black_scholes_price(
                    new_spot, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
                )
                
                # Calculate P&L
                pnl = (new_price - current_price) * quantity
                portfolio_pnl.append(pnl)
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(portfolio_pnl, var_percentile)
            
            return abs(min(var_value, 0))
            
        except Exception as e:
            return 0.0
    
    def _historical_simulation_var(self, spot_price: float, strike_price: float,
                                  time_to_expiry: float, risk_free_rate: float,
                                  volatility: float, option_type: str,
                                  confidence_level: float, quantity: int) -> float:
        """Calculate VaR using historical simulation with synthetic data."""
        try:
            # Generate synthetic historical returns (normal distribution)
            np.random.seed(42)
            daily_vol = volatility / np.sqrt(252)
            historical_returns = np.random.normal(0, daily_vol, 1000)
            
            current_price = self.black_scholes_price(
                spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
            )
            
            portfolio_pnl = []
            
            for ret in historical_returns:
                # Calculate new underlying price
                new_spot = spot_price * (1 + ret)
                
                # Calculate new option price
                new_price = self.black_scholes_price(
                    new_spot, strike_price, time_to_expiry, risk_free_rate, volatility, option_type
                )
                
                # Calculate P&L
                pnl = (new_price - current_price) * quantity
                portfolio_pnl.append(pnl)
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(portfolio_pnl, var_percentile)
            
            return abs(min(var_value, 0))
            
        except Exception as e:
            return 0.0
    
    def perform_stress_test(self, spot_price: float, strike_price: float,
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
                    var_model: str, backtest_days: int = 198) -> Dict[str, Any]:
        """
        Perform VaR backtesting to validate the model accuracy.
        
        Returns:
        Dictionary with backtesting results including violation rate and statistics
        """
        try:
            # Get sufficient historical data for backtesting
            required_period = max(backtest_days + 252, 450)  # Ensure enough data
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
                        var_result = self.calculate_options_var_historical(
                            spot_price, strike_price, time_to_expiry, risk_free_rate,
                            volatility, option_type, historical_returns, confidence_level, quantity
                        )
                        var_estimate = var_result.get('var', 0)
                    else:
                        var_result = self.calculate_options_var(
                            spot_price, strike_price, time_to_expiry, risk_free_rate,
                            volatility, option_type, var_model, confidence_level, quantity
                        )
                        var_estimate = var_result.get('var', 0)
                    
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
