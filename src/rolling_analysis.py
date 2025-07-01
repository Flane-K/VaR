import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

class RollingAnalysis:
    def __init__(self):
        pass
    
    def calculate_rolling_var(self, returns, confidence_level, window_size):
        """Calculate rolling VaR"""
        try:
            if len(returns) < window_size:
                st.warning(f"Insufficient data for rolling analysis (need at least {window_size} points)")
                return pd.Series()
            
            rolling_var = []
            dates = []
            
            for i in range(window_size, len(returns)):
                window_returns = returns.iloc[i-window_size:i]
                
                # Calculate parametric VaR for the window
                mu = window_returns.mean()
                sigma = window_returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                var = -(mu + sigma * z_score) * 100000  # Convert to dollar amount
                
                rolling_var.append(var)
                dates.append(returns.index[i])
            
            return pd.Series(rolling_var, index=dates)
            
        except Exception as e:
            st.error(f"Error calculating rolling VaR: {str(e)}")
            return pd.Series()
    
    def calculate_rolling_volatility(self, returns, window_size, annualize=True):
        """Calculate rolling volatility"""
        try:
            rolling_vol = returns.rolling(window=window_size).std()
            
            if annualize:
                rolling_vol = rolling_vol * np.sqrt(252)  # Annualize
            
            return rolling_vol.dropna()
            
        except Exception as e:
            st.error(f"Error calculating rolling volatility: {str(e)}")
            return pd.Series()
    
    def calculate_rolling_sharpe(self, returns, window_size, risk_free_rate=0.02):
        """Calculate rolling Sharpe ratio"""
        try:
            rolling_mean = returns.rolling(window=window_size).mean() * 252  # Annualize
            rolling_vol = returns.rolling(window=window_size).std() * np.sqrt(252)  # Annualize
            
            # Calculate Sharpe ratio
            rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_vol
            
            return rolling_sharpe.dropna()
            
        except Exception as e:
            st.error(f"Error calculating rolling Sharpe ratio: {str(e)}")
            return pd.Series()
    
    def calculate_rolling_beta(self, asset_returns, market_returns, window_size):
        """Calculate rolling beta"""
        try:
            if len(asset_returns) != len(market_returns):
                st.error("Asset and market returns must have the same length")
                return pd.Series()
            
            rolling_beta = []
            dates = []
            
            for i in range(window_size, len(asset_returns)):
                asset_window = asset_returns.iloc[i-window_size:i]
                market_window = market_returns.iloc[i-window_size:i]
                
                # Calculate beta using linear regression
                covariance = np.cov(asset_window, market_window)[0, 1]
                market_variance = np.var(market_window)
                
                if market_variance != 0:
                    beta = covariance / market_variance
                else:
                    beta = 0
                
                rolling_beta.append(beta)
                dates.append(asset_returns.index[i])
            
            return pd.Series(rolling_beta, index=dates)
            
        except Exception as e:
            st.error(f"Error calculating rolling beta: {str(e)}")
            return pd.Series()
    
    def calculate_drawdown(self, returns):
        """Calculate drawdown series"""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            return drawdown
            
        except Exception as e:
            st.error(f"Error calculating drawdown: {str(e)}")
            return pd.Series()
    
    def calculate_maximum_drawdown(self, returns):
        """Calculate maximum drawdown"""
        try:
            drawdown = self.calculate_drawdown(returns)
            max_drawdown = drawdown.min()
            
            # Find the dates of maximum drawdown
            max_dd_date = drawdown.idxmin()
            
            # Find the peak before the maximum drawdown
            peak_date = None
            for date in reversed(drawdown.index):
                if date >= max_dd_date:
                    continue
                if drawdown[date] == 0:
                    peak_date = date
                    break
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_date': max_dd_date,
                'peak_date': peak_date,
                'drawdown_series': drawdown
            }
            
        except Exception as e:
            st.error(f"Error calculating maximum drawdown: {str(e)}")
            return {}
    
    def calculate_rolling_correlations(self, returns_matrix, window_size):
        """Calculate rolling correlations between assets"""
        try:
            if returns_matrix.shape[1] < 2:
                st.warning("Need at least 2 assets for correlation analysis")
                return {}
            
            assets = returns_matrix.columns
            rolling_correlations = {}
            
            for i in range(len(assets)):
                for j in range(i+1, len(assets)):
                    asset1, asset2 = assets[i], assets[j]
                    
                    # Calculate rolling correlation
                    rolling_corr = returns_matrix[asset1].rolling(window=window_size).corr(
                        returns_matrix[asset2]
                    )
                    
                    rolling_correlations[f"{asset1}_{asset2}"] = rolling_corr.dropna()
            
            return rolling_correlations
            
        except Exception as e:
            st.error(f"Error calculating rolling correlations: {str(e)}")
            return {}
    
    def calculate_rolling_skewness(self, returns, window_size):
        """Calculate rolling skewness"""
        try:
            rolling_skew = returns.rolling(window=window_size).skew()
            return rolling_skew.dropna()
            
        except Exception as e:
            st.error(f"Error calculating rolling skewness: {str(e)}")
            return pd.Series()
    
    def calculate_rolling_kurtosis(self, returns, window_size):
        """Calculate rolling kurtosis"""
        try:
            rolling_kurt = returns.rolling(window=window_size).kurt()
            return rolling_kurt.dropna()
            
        except Exception as e:
            st.error(f"Error calculating rolling kurtosis: {str(e)}")
            return pd.Series()
    
    def calculate_rolling_expected_shortfall(self, returns, confidence_level, window_size):
        """Calculate rolling Expected Shortfall"""
        try:
            rolling_es = []
            dates = []
            
            for i in range(window_size, len(returns)):
                window_returns = returns.iloc[i-window_size:i]
                
                # Calculate VaR threshold
                var_percentile = (1 - confidence_level) * 100
                var_threshold = np.percentile(window_returns, var_percentile)
                
                # Calculate Expected Shortfall
                tail_returns = window_returns[window_returns <= var_threshold]
                
                if len(tail_returns) > 0:
                    expected_shortfall = -tail_returns.mean() * 100000
                else:
                    expected_shortfall = 0
                
                rolling_es.append(expected_shortfall)
                dates.append(returns.index[i])
            
            return pd.Series(rolling_es, index=dates)
            
        except Exception as e:
            st.error(f"Error calculating rolling Expected Shortfall: {str(e)}")
            return pd.Series()
    
    def generate_rolling_metrics_summary(self, returns, window_size=60, confidence_level=0.95):
        """Generate comprehensive rolling metrics summary"""
        try:
            summary = {}
            
            # Rolling VaR
            summary['Rolling_VaR'] = self.calculate_rolling_var(returns, confidence_level, window_size)
            
            # Rolling Volatility
            summary['Rolling_Volatility'] = self.calculate_rolling_volatility(returns, window_size)
            
            # Rolling Sharpe Ratio
            summary['Rolling_Sharpe'] = self.calculate_rolling_sharpe(returns, window_size)
            
            # Rolling Expected Shortfall
            summary['Rolling_ES'] = self.calculate_rolling_expected_shortfall(returns, confidence_level, window_size)
            
            # Rolling Skewness
            summary['Rolling_Skewness'] = self.calculate_rolling_skewness(returns, window_size)
            
            # Rolling Kurtosis
            summary['Rolling_Kurtosis'] = self.calculate_rolling_kurtosis(returns, window_size)
            
            # Drawdown
            summary['Drawdown'] = self.calculate_drawdown(returns)
            
            return summary
            
        except Exception as e:
            st.error(f"Error generating rolling metrics summary: {str(e)}")
            return {}
    
    def identify_risk_regimes(self, returns, volatility_threshold_low=0.10, volatility_threshold_high=0.25):
        """Identify risk regimes based on volatility"""
        try:
            # Calculate rolling volatility
            rolling_vol = self.calculate_rolling_volatility(returns, window_size=30)
            
            # Classify regimes
            regimes = pd.Series(index=rolling_vol.index, dtype='object')
            regimes[rolling_vol <= volatility_threshold_low] = 'Low Risk'
            regimes[(rolling_vol > volatility_threshold_low) & (rolling_vol <= volatility_threshold_high)] = 'Medium Risk'
            regimes[rolling_vol > volatility_threshold_high] = 'High Risk'
            
            return regimes
            
        except Exception as e:
            st.error(f"Error identifying risk regimes: {str(e)}")
            return pd.Series()