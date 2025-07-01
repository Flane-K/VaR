import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import io
import base64

class Utils:
    def __init__(self):
        pass
    
    def format_currency(self, value, currency='USD'):
        """Format currency values"""
        try:
            if currency == 'USD':
                return f"${value:,.2f}"
            elif currency == 'EUR':
                return f"€{value:,.2f}"
            elif currency == 'GBP':
                return f"£{value:,.2f}"
            else:
                return f"{value:,.2f} {currency}"
        except:
            return str(value)
    
    def format_percentage(self, value, decimals=2):
        """Format percentage values"""
        try:
            return f"{value:.{decimals}f}%"
        except:
            return str(value)
    
    def calculate_portfolio_statistics(self, returns):
        """Calculate comprehensive portfolio statistics"""
        try:
            if len(returns) == 0:
                return {}
            
            stats = {
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'annual_return': returns.mean() * 252,
                'annual_volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'min_return': returns.min(),
                'max_return': returns.max(),
                'total_observations': len(returns)
            }
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252)
                stats['sortino_ratio'] = (returns.mean() * 252) / downside_std
            else:
                stats['sortino_ratio'] = np.inf
            
            # Calmar ratio
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            if max_drawdown != 0:
                stats['calmar_ratio'] = (returns.mean() * 252) / abs(max_drawdown)
                stats['max_drawdown'] = max_drawdown
            else:
                stats['calmar_ratio'] = np.inf
                stats['max_drawdown'] = 0
            
            return stats
            
        except Exception as e:
            st.error(f"Error calculating portfolio statistics: {str(e)}")
            return {}
    
    def export_to_csv(self, data, filename):
        """Export data to CSV"""
        try:
            if isinstance(data, pd.DataFrame):
                csv_data = data.to_csv(index=True)
            elif isinstance(data, pd.Series):
                csv_data = data.to_csv()
            elif isinstance(data, dict):
                df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
                csv_data = df.to_csv(index=False)
            else:
                csv_data = str(data)
            
            return csv_data
            
        except Exception as e:
            st.error(f"Error exporting to CSV: {str(e)}")
            return ""
    
    def export_to_excel(self, data_dict, filename):
        """Export multiple datasets to Excel"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, data in data_dict.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_excel(writer, sheet_name=sheet_name, index=True)
                    elif isinstance(data, pd.Series):
                        data.to_excel(writer, sheet_name=sheet_name)
                    elif isinstance(data, dict):
                        df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            processed_data = output.getvalue()
            return processed_data
            
        except Exception as e:
            st.error(f"Error exporting to Excel: {str(e)}")
            return None
    
    def validate_portfolio_weights(self, weights):
        """Validate portfolio weights"""
        try:
            if not isinstance(weights, (list, np.ndarray, pd.Series)):
                return False, "Weights must be a list, array, or series"
            
            weights = np.array(weights)
            
            # Check for negative weights
            if np.any(weights < 0):
                return False, "Weights cannot be negative"
            
            # Check if weights sum to 1 (within tolerance)
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1.0) > 1e-6:
                return False, f"Weights must sum to 1.0, current sum: {weight_sum:.6f}"
            
            return True, "Weights are valid"
            
        except Exception as e:
            return False, f"Error validating weights: {str(e)}"
    
    def calculate_portfolio_metrics(self, returns, weights, benchmark_returns=None):
        """Calculate portfolio metrics including tracking error and information ratio"""
        try:
            if len(returns.shape) == 1:
                # Single asset
                portfolio_returns = returns
            else:
                # Multi-asset portfolio
                portfolio_returns = returns.dot(weights)
            
            metrics = self.calculate_portfolio_statistics(portfolio_returns)
            
            # Add benchmark comparison if provided
            if benchmark_returns is not None:
                # Tracking error
                active_returns = portfolio_returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)
                metrics['tracking_error'] = tracking_error
                
                # Information ratio
                if tracking_error != 0:
                    information_ratio = (active_returns.mean() * 252) / tracking_error
                    metrics['information_ratio'] = information_ratio
                else:
                    metrics['information_ratio'] = 0
                
                # Beta
                covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                if benchmark_variance != 0:
                    beta = covariance / benchmark_variance
                    metrics['beta'] = beta
                else:
                    metrics['beta'] = 0
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def create_performance_attribution(self, returns, weights, sector_mapping=None):
        """Create performance attribution analysis"""
        try:
            if len(returns.shape) == 1 or sector_mapping is None:
                return {}
            
            attribution = {}
            
            # Calculate individual asset contributions
            for i, asset in enumerate(returns.columns):
                asset_return = returns.iloc[:, i].mean() * 252  # Annualized
                asset_weight = weights[i]
                contribution = asset_return * asset_weight
                
                attribution[asset] = {
                    'weight': asset_weight,
                    'return': asset_return,
                    'contribution': contribution
                }
            
            # Calculate sector contributions if mapping provided
            if sector_mapping:
                sector_attribution = {}
                for sector in set(sector_mapping.values()):
                    sector_assets = [asset for asset, sec in sector_mapping.items() if sec == sector]
                    sector_contribution = sum(attribution[asset]['contribution'] for asset in sector_assets if asset in attribution)
                    sector_weight = sum(attribution[asset]['weight'] for asset in sector_assets if asset in attribution)
                    
                    if sector_weight > 0:
                        sector_return = sector_contribution / sector_weight
                        sector_attribution[sector] = {
                            'weight': sector_weight,
                            'return': sector_return,
                            'contribution': sector_contribution
                        }
                
                attribution['sectors'] = sector_attribution
            
            return attribution
            
        except Exception as e:
            st.error(f"Error creating performance attribution: {str(e)}")
            return {}
    
    def generate_risk_report_summary(self, portfolio_data, var_results, backtest_results=None):
        """Generate executive summary for risk report"""
        try:
            summary = {
                'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'portfolio_overview': {},
                'risk_metrics': {},
                'var_analysis': {},
                'recommendations': []
            }
            
            # Portfolio overview
            if 'returns' in portfolio_data:
                returns = portfolio_data['returns']
                stats = self.calculate_portfolio_statistics(returns)
                
                summary['portfolio_overview'] = {
                    'total_return_annual': self.format_percentage(stats['annual_return'] * 100),
                    'volatility_annual': self.format_percentage(stats['annual_volatility'] * 100),
                    'sharpe_ratio': f"{stats['sharpe_ratio']:.3f}",
                    'max_drawdown': self.format_percentage(stats['max_drawdown'] * 100),
                    'observations': stats['total_observations']
                }
            
            # VaR analysis
            if var_results:
                summary['var_analysis'] = {
                    'var_95': self.format_currency(var_results.get('var_95', 0)),
                    'expected_shortfall': self.format_currency(var_results.get('expected_shortfall', 0)),
                    'confidence_level': '95%'
                }
            
            # Backtesting results
            if backtest_results:
                summary['backtesting'] = {
                    'total_violations': backtest_results.get('violations', 0),
                    'expected_violations': f"{backtest_results.get('expected_violations', 0):.1f}",
                    'kupiec_test_pvalue': f"{backtest_results.get('kupiec_pvalue', 0):.4f}",
                    'model_performance': 'Adequate' if backtest_results.get('kupiec_pvalue', 0) > 0.05 else 'Needs Review'
                }
            
            # Generate recommendations
            recommendations = []
            
            if 'returns' in portfolio_data:
                volatility = stats.get('annual_volatility', 0)
                if volatility > 0.25:
                    recommendations.append("High portfolio volatility detected. Consider diversification.")
                
                if stats.get('sharpe_ratio', 0) < 0.5:
                    recommendations.append("Low risk-adjusted returns. Review asset allocation.")
                
                if abs(stats.get('max_drawdown', 0)) > 0.2:
                    recommendations.append("Significant drawdown risk. Implement risk management measures.")
            
            if backtest_results and backtest_results.get('kupiec_pvalue', 1) < 0.05:
                recommendations.append("VaR model failing backtests. Consider alternative methodologies.")
            
            summary['recommendations'] = recommendations
            
            return summary
            
        except Exception as e:
            st.error(f"Error generating risk report summary: {str(e)}")
            return {}
    
    def create_download_link(self, data, filename, file_type='csv'):
        """Create download link for data"""
        try:
            if file_type.lower() == 'csv':
                if isinstance(data, str):
                    file_data = data
                else:
                    file_data = self.export_to_csv(data, filename)
                
                b64 = base64.b64encode(file_data.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
                
            elif file_type.lower() == 'excel':
                if isinstance(data, bytes):
                    file_data = data
                else:
                    file_data = self.export_to_excel(data, filename)
                
                if file_data:
                    b64 = base64.b64encode(file_data).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel File</a>'
                else:
                    href = "Error creating Excel file"
            
            else:
                href = "Unsupported file type"
            
            return href
            
        except Exception as e:
            st.error(f"Error creating download link: {str(e)}")
            return "Error creating download link"
    
    def validate_data_quality(self, data):
        """Validate data quality and provide recommendations"""
        try:
            quality_report = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'recommendations': []
            }
            
            if data is None or data.empty:
                quality_report['valid'] = False
                quality_report['errors'].append("Data is empty or None")
                return quality_report
            
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
            if missing_pct > 0:
                quality_report['warnings'].append(f"Missing data: {missing_pct:.2f}%")
                if missing_pct > 10:
                    quality_report['recommendations'].append("Consider data cleaning or imputation")
            
            # Check for outliers (simple method)
            if data.select_dtypes(include=[np.number]).shape[1] > 0:
                numeric_data = data.select_dtypes(include=[np.number])
                for col in numeric_data.columns:
                    Q1 = numeric_data[col].quantile(0.25)
                    Q3 = numeric_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = numeric_data[(numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)]
                    if len(outliers) > 0:
                        outlier_pct = len(outliers) / len(numeric_data) * 100
                        quality_report['warnings'].append(f"Outliers in {col}: {outlier_pct:.2f}%")
            
            # Check data frequency consistency
            if hasattr(data.index, 'freq') and data.index.freq is None:
                quality_report['recommendations'].append("Data frequency is inconsistent")
            
            # Check for sufficient data points
            if len(data) < 252:  # Less than 1 year of daily data
                quality_report['warnings'].append("Limited historical data may affect model reliability")
            
            return quality_report
            
        except Exception as e:
            st.error(f"Error validating data quality: {str(e)}")
            return {'valid': False, 'errors': [str(e)]}