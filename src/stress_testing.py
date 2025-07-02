import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

class StressTesting:
    def __init__(self):
        # Historical scenarios data
        self.historical_scenarios = {
            "2008 Financial Crisis": {
                "volatility_shock": 150,  # 150% increase
                "correlation_shock": 0.3,  # Correlations increase to 0.3
                "market_shock": -30,  # 30% market drop
                "description": "2008 Financial Crisis scenario"
            },
            "COVID-19 Pandemic": {
                "volatility_shock": 200,  # 200% increase
                "correlation_shock": 0.4,  # Correlations increase to 0.4
                "market_shock": -35,  # 35% market drop
                "description": "COVID-19 Pandemic scenario"
            },
            "Dot-com Crash": {
                "volatility_shock": 100,  # 100% increase
                "correlation_shock": 0.2,  # Correlations increase to 0.2
                "market_shock": -25,  # 25% market drop
                "description": "Dot-com Crash scenario"
            }
        }
    
    def run_stress_test(self, returns, scenario_type, confidence_level, time_horizon=1):
        """Run stress test on portfolio"""
        try:
            if len(returns) == 0:
                return {}
            
            # Get baseline VaR
            baseline_var = self._calculate_baseline_var(returns, confidence_level, time_horizon)
            
            # Apply scenario
            if scenario_type in self.historical_scenarios:
                scenario = self.historical_scenarios[scenario_type]
                stressed_returns = self._apply_historical_stress(returns, scenario)
            else:
                st.warning(f"Unknown scenario type: {scenario_type}")
                return {}
            
            # Calculate stressed VaR
            stressed_var = self._calculate_baseline_var(stressed_returns, confidence_level, time_horizon)
            
            # Calculate metrics
            var_increase = ((stressed_var - baseline_var) / baseline_var) * 100 if baseline_var != 0 else 0
            worst_case = np.percentile(stressed_returns, 1) * 100000  # 1st percentile
            
            return {
                "baseline_var": baseline_var,
                "stressed_var": stressed_var,
                "var_increase": var_increase,
                "worst_case": abs(worst_case),
                "scenario_description": scenario_type,
                "stressed_returns": stressed_returns
            }
            
        except Exception as e:
            st.error(f"Error in stress testing: {str(e)}")
            return {}
    
    def run_custom_stress_test(self, returns, vol_shock_pct, corr_shock, market_shock_pct, confidence_level, time_horizon=1):
        """Run custom stress test on portfolio"""
        try:
            if len(returns) == 0:
                return {}
            
            # Get baseline VaR
            baseline_var = self._calculate_baseline_var(returns, confidence_level, time_horizon)
            
            # Apply custom stress
            stressed_returns = self._apply_custom_stress(returns, vol_shock_pct, corr_shock, market_shock_pct)
            
            # Calculate stressed VaR
            stressed_var = self._calculate_baseline_var(stressed_returns, confidence_level, time_horizon)
            
            # Calculate metrics
            var_increase = ((stressed_var - baseline_var) / baseline_var) * 100 if baseline_var != 0 else 0
            worst_case = np.percentile(stressed_returns, 1) * 100000  # 1st percentile
            
            return {
                "baseline_var": baseline_var,
                "stressed_var": stressed_var,
                "var_increase": var_increase,
                "worst_case": abs(worst_case),
                "scenario_description": "Custom Scenario",
                "stressed_returns": stressed_returns
            }
            
        except Exception as e:
            st.error(f"Error in custom stress testing: {str(e)}")
            return {}
    
    def _calculate_baseline_var(self, returns, confidence_level, time_horizon):
        """Calculate baseline VaR (parametric method)"""
        try:
            mu = returns.mean()
            sigma = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mu * time_horizon + sigma * np.sqrt(time_horizon) * z_score)
            return var * 100000  # Convert to dollar amount
        except Exception as e:
            st.error(f"Error calculating baseline VaR: {str(e)}")
            return 0
    
    def _apply_historical_stress(self, returns, scenario):
        """Apply historical stress scenario"""
        try:
            # Get scenario parameters
            vol_shock = scenario["volatility_shock"] / 100
            corr_shock = scenario["correlation_shock"]
            market_shock = scenario["market_shock"] / 100
            
            # Apply volatility shock
            current_vol = returns.std()
            shocked_vol = current_vol * (1 + vol_shock)
            
            # Apply market shock
            shocked_mean = returns.mean() + market_shock / 252  # Daily shock
            
            # Generate stressed returns
            np.random.seed(42)
            stressed_returns = np.random.normal(
                shocked_mean, 
                shocked_vol, 
                len(returns)
            )
            
            return pd.Series(stressed_returns, index=returns.index)
            
        except Exception as e:
            st.error(f"Error applying historical stress: {str(e)}")
            return returns
    
    def _apply_custom_stress(self, returns, vol_shock_pct, corr_shock, market_shock_pct):
        """Apply custom stress scenario"""
        try:
            # Convert percentages to decimals
            vol_shock = vol_shock_pct / 100
            market_shock = market_shock_pct / 100
            
            # Apply shocks
            current_vol = returns.std()
            shocked_vol = current_vol * (1 + vol_shock)
            shocked_mean = returns.mean() + market_shock / 252
            
            # Generate stressed returns
            np.random.seed(42)
            stressed_returns = np.random.normal(
                shocked_mean,
                shocked_vol,
                len(returns)
            )
            
            return pd.Series(stressed_returns, index=returns.index)
            
        except Exception as e:
            st.error(f"Error applying custom stress: {str(e)}")
            return returns
    
    def scenario_analysis(self, returns, scenarios_list, confidence_level):
        """Analyze multiple scenarios"""
        try:
            results = {}
            
            for scenario in scenarios_list:
                scenario_result = self.run_stress_test(returns, scenario, confidence_level)
                results[scenario] = scenario_result
            
            return results
            
        except Exception as e:
            st.error(f"Error in scenario analysis: {str(e)}")
            return {}
    
    def sensitivity_analysis(self, returns, confidence_level, shock_range=[-50, 50], num_points=11):
        """Perform sensitivity analysis"""
        try:
            shocks = np.linspace(shock_range[0], shock_range[1], num_points)
            sensitivity_results = []
            
            for shock in shocks:
                # Apply market shock
                shocked_returns = self._apply_custom_stress(returns, 0, 0, shock)
                shocked_var = self._calculate_baseline_var(shocked_returns, confidence_level, 1)
                
                sensitivity_results.append({
                    'shock': shock,
                    'var': shocked_var
                })
            
            return pd.DataFrame(sensitivity_results)
            
        except Exception as e:
            st.error(f"Error in sensitivity analysis: {str(e)}")
            return pd.DataFrame()
    
    def correlation_stress_test(self, returns_matrix, weights, confidence_level, correlation_shock=0.5):
        """Stress test correlation matrix"""
        try:
            if returns_matrix.empty:
                return {}
            
            # Calculate baseline portfolio VaR
            portfolio_returns = returns_matrix.dot(weights)
            baseline_var = self._calculate_baseline_var(portfolio_returns, confidence_level, 1)
            
            # Calculate correlation matrix
            corr_matrix = returns_matrix.corr()
            
            # Apply correlation shock (increase all correlations)
            shocked_corr = corr_matrix * (1 - correlation_shock) + correlation_shock
            np.fill_diagonal(shocked_corr.values, 1.0)  # Ensure diagonal is 1
            
            # Generate stressed returns using Cholesky decomposition
            try:
                chol = np.linalg.cholesky(shocked_corr)
                
                # Generate uncorrelated random returns
                np.random.seed(42)
                random_returns = np.random.normal(0, 1, (len(returns_matrix), len(returns_matrix.columns)))
                
                # Apply correlation structure
                correlated_returns = random_returns @ chol.T
                
                # Scale by original volatilities and add means
                for i, col in enumerate(returns_matrix.columns):
                    original_mean = returns_matrix[col].mean()
                    original_std = returns_matrix[col].std()
                    correlated_returns[:, i] = correlated_returns[:, i] * original_std + original_mean
                
                # Create DataFrame
                stressed_returns_matrix = pd.DataFrame(
                    correlated_returns,
                    columns=returns_matrix.columns,
                    index=returns_matrix.index
                )
                
                # Calculate stressed portfolio returns
                stressed_portfolio_returns = stressed_returns_matrix.dot(weights)
                stressed_var = self._calculate_baseline_var(stressed_portfolio_returns, confidence_level, 1)
                
                return {
                    "baseline_var": baseline_var,
                    "stressed_var": stressed_var,
                    "var_increase": ((stressed_var - baseline_var) / baseline_var) * 100,
                    "correlation_shock": correlation_shock,
                    "stressed_returns": stressed_portfolio_returns
                }
                
            except np.linalg.LinAlgError:
                st.warning("Correlation matrix is not positive definite, using original returns")
                return {
                    "baseline_var": baseline_var,
                    "stressed_var": baseline_var,
                    "var_increase": 0,
                    "correlation_shock": correlation_shock,
                    "stressed_returns": portfolio_returns
                }
            
        except Exception as e:
            st.error(f"Error in correlation stress test: {str(e)}")
            return {}
    
    def tail_risk_analysis(self, returns, confidence_levels=[0.95, 0.99, 0.999]):
        """Analyze tail risk at different confidence levels"""
        try:
            tail_risk_results = {}
            
            for cl in confidence_levels:
                var = self._calculate_baseline_var(returns, cl, 1)
                
                # Calculate Expected Shortfall
                var_percentile = (1 - cl) * 100
                var_threshold = np.percentile(returns, var_percentile)
                tail_returns = returns[returns <= var_threshold]
                
                if len(tail_returns) > 0:
                    expected_shortfall = -tail_returns.mean() * 100000
                else:
                    expected_shortfall = var
                
                tail_risk_results[f"{cl*100:.1f}%"] = {
                    "VaR": var,
                    "Expected_Shortfall": expected_shortfall
                }
            
            return tail_risk_results
            
        except Exception as e:
            st.error(f"Error in tail risk analysis: {str(e)}")
            return {}
    
    def generate_stress_report(self, returns, confidence_level):
        """Generate comprehensive stress testing report"""
        try:
            report = {}
            
            # Run all historical scenarios
            for scenario_name in self.historical_scenarios.keys():
                scenario_result = self.run_stress_test(returns, scenario_name, confidence_level)
                report[scenario_name] = scenario_result
            
            # Add sensitivity analysis
            sensitivity = self.sensitivity_analysis(returns, confidence_level)
            report["Sensitivity_Analysis"] = sensitivity
            
            # Add tail risk analysis
            tail_risk = self.tail_risk_analysis(returns)
            report["Tail_Risk_Analysis"] = tail_risk
            
            return report
            
        except Exception as e:
            st.error(f"Error generating stress report: {str(e)}")
            return {}