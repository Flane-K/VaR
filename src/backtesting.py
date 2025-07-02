import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

class Backtesting:
    def __init__(self):
        pass
    
    def perform_backtesting(self, returns, confidence_level, window_size, var_method):
        """Perform backtesting of VaR model"""
        try:
            if len(returns) < window_size + 50:
                st.warning("Insufficient data for backtesting")
                return {}
            
            # Initialize arrays
            var_estimates = []
            actual_returns = []
            violations = []
            violations_dates = []
            
            # Rolling window backtesting
            for i in range(window_size, len(returns)):
                # Get historical window
                hist_returns = returns.iloc[i-window_size:i]
                
                # Calculate VaR estimate based on method
                try:
                    if hasattr(var_method, '__call__'):
                        # If var_method is a function, call it directly
                        var_est = var_method(hist_returns, confidence_level, 1)
                        # Convert to return space (assuming var_method returns dollar amount)
                        var_est = var_est / 100000  # Convert back to return percentage
                    else:
                        # Fallback to parametric method
                        var_est = self._calculate_parametric_var_simple(hist_returns, confidence_level)
                except Exception as e:
                    st.warning(f"Error calculating VaR at step {i}: {e}")
                    var_est = self._calculate_parametric_var_simple(hist_returns, confidence_level)
                
                # Get actual return
                actual_return = returns.iloc[i]
                
                # Check for violation (actual loss exceeds VaR)
                violation = actual_return < -var_est
                
                var_estimates.append(var_est)
                actual_returns.append(actual_return)
                violations.append(violation)
                violations_dates.append(returns.index[i])
            
            # Calculate metrics
            total_observations = len(violations)
            total_violations = sum(violations)
            expected_violations = total_observations * (1 - confidence_level)
            violation_rate = total_violations / total_observations if total_observations > 0 else 0
            
            # Kupiec Test (Unconditional Coverage)
            kupiec_lr, kupiec_pvalue = self._kupiec_test(total_violations, total_observations, confidence_level)
            
            # Christoffersen Test (Independence)
            independence_lr, independence_pvalue = self._christoffersen_independence_test(violations)
            
            # Conditional Coverage Test
            cc_lr, cc_pvalue = self._conditional_coverage_test(violations, confidence_level)
            
            return {
                'var_estimates': var_estimates,
                'actual_returns': actual_returns,
                'violations': total_violations,
                'expected_violations': expected_violations,
                'violation_rate': violation_rate,
                'violations_dates': violations_dates,
                'kupiec_lr': kupiec_lr,
                'kupiec_pvalue': kupiec_pvalue,
                'independence_lr': independence_lr,
                'independence_pvalue': independence_pvalue,
                'cc_lr': cc_lr,
                'cc_pvalue': cc_pvalue
            }
            
        except Exception as e:
            st.error(f"Error in backtesting: {str(e)}")
            return {}
    
    def _calculate_parametric_var_simple(self, returns, confidence_level):
        """Simple parametric VaR calculation for backtesting"""
        try:
            mu = returns.mean()
            sigma = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return -(mu + sigma * z_score)
        except Exception as e:
            st.warning(f"Error in parametric VaR calculation: {e}")
            return 0
    
    def _calculate_historical_var_simple(self, returns, confidence_level):
        """Simple historical VaR calculation for backtesting"""
        try:
            var_percentile = (1 - confidence_level) * 100
            return -np.percentile(returns, var_percentile)
        except Exception as e:
            st.warning(f"Error in historical VaR calculation: {e}")
            return 0
    
    def _calculate_monte_carlo_var_simple(self, returns, confidence_level, num_simulations=1000):
        """Simple Monte Carlo VaR calculation for backtesting"""
        try:
            mu = returns.mean()
            sigma = returns.std()
            
            np.random.seed(42)
            simulated_returns = np.random.normal(mu, sigma, num_simulations)
            
            var_percentile = (1 - confidence_level) * 100
            return -np.percentile(simulated_returns, var_percentile)
        except Exception as e:
            st.warning(f"Error in Monte Carlo VaR calculation: {e}")
            return 0
    
    def _kupiec_test(self, violations, observations, confidence_level):
        """Kupiec's Proportion of Failures (POF) test"""
        try:
            if observations == 0:
                return 0, 1.0
                
            expected_rate = 1 - confidence_level
            observed_rate = violations / observations
            
            if violations == 0 or violations == observations:
                return 0, 1.0
            
            # Log-likelihood ratio
            lr = 2 * (
                violations * np.log(observed_rate / expected_rate) +
                (observations - violations) * np.log((1 - observed_rate) / (1 - expected_rate))
            )
            
            # P-value from chi-squared distribution with 1 degree of freedom
            pvalue = 1 - stats.chi2.cdf(lr, 1)
            
            return lr, pvalue
            
        except Exception as e:
            st.warning(f"Error in Kupiec test: {str(e)}")
            return 0, 1.0
    
    def _christoffersen_independence_test(self, violations):
        """Christoffersen's Independence test"""
        try:
            if len(violations) < 2:
                return 0, 1.0
                
            # Transition matrix
            n00 = n01 = n10 = n11 = 0
            
            for i in range(1, len(violations)):
                if not violations[i-1] and not violations[i]:
                    n00 += 1
                elif not violations[i-1] and violations[i]:
                    n01 += 1
                elif violations[i-1] and not violations[i]:
                    n10 += 1
                elif violations[i-1] and violations[i]:
                    n11 += 1
            
            if n01 + n11 == 0 or n00 + n10 == 0:
                return 0, 1.0
            
            pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
            pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
            pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
            
            if pi01 == 0 or pi11 == 0 or pi == 0:
                return 0, 1.0
            
            # Log-likelihood ratio for independence
            lr = 2 * (
                n00 * np.log((1 - pi01) / (1 - pi)) +
                n01 * np.log(pi01 / pi) +
                n10 * np.log((1 - pi11) / (1 - pi)) +
                n11 * np.log(pi11 / pi)
            )
            
            pvalue = 1 - stats.chi2.cdf(lr, 1)
            
            return lr, pvalue
            
        except Exception as e:
            st.warning(f"Error in Independence test: {str(e)}")
            return 0, 1.0
    
    def _conditional_coverage_test(self, violations, confidence_level):
        """Christoffersen's Conditional Coverage test"""
        try:
            # Combine Kupiec and Independence tests
            kupiec_lr, _ = self._kupiec_test(sum(violations), len(violations), confidence_level)
            independence_lr, _ = self._christoffersen_independence_test(violations)
            
            # Combined test statistic
            cc_lr = kupiec_lr + independence_lr
            pvalue = 1 - stats.chi2.cdf(cc_lr, 2)
            
            return cc_lr, pvalue
            
        except Exception as e:
            st.warning(f"Error in Conditional Coverage test: {str(e)}")
            return 0, 1.0
    
    def basel_traffic_light(self, violations, expected_violations):
        """Basel Traffic Light System"""
        try:
            if violations <= expected_violations + 4:
                return "Green"
            elif violations <= expected_violations + 9:
                return "Yellow"
            else:
                return "Red"
                
        except Exception as e:
            st.warning(f"Error in Basel Traffic Light: {str(e)}")
            return "Unknown"
    
    def calculate_hit_ratio(self, violations, window_size=30):
        """Calculate rolling hit ratio"""
        try:
            if len(violations) < window_size:
                return []
            
            hit_ratios = []
            for i in range(window_size, len(violations)):
                window_violations = violations[i-window_size:i]
                hit_ratio = sum(window_violations) / window_size
                hit_ratios.append(hit_ratio)
            
            return hit_ratios
            
        except Exception as e:
            st.warning(f"Error calculating hit ratio: {str(e)}")
            return []
    
    def backtest_comparison(self, returns, confidence_level, window_size):
        """Compare multiple VaR models in backtesting"""
        try:
            models = ["Parametric", "Historical", "Monte Carlo"]
            results = {}
            
            for model in models:
                model_results = self.perform_backtesting(returns, confidence_level, window_size, model)
                results[model] = model_results
            
            return results
            
        except Exception as e:
            st.error(f"Error in backtest comparison: {str(e)}")
            return {}