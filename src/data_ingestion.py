import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import io

class DataIngestion:
    def __init__(self):
        self.data = None
        self.returns = None
    
    def load_live_data(self, symbols, start_date, end_date, min_data_points=None):
        """Load live market data from Yahoo Finance with crypto support and sufficient data for backtesting"""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Calculate required data points for backtesting
            if min_data_points:
                # Ensure we have enough data for backtesting
                required_days = min_data_points + 100  # Add buffer for weekends/holidays
                start_date = end_date - timedelta(days=required_days)
            
            # Separate crypto and regular symbols
            crypto_symbols = [s for s in symbols if '-USD' in s]
            regular_symbols = [s for s in symbols if '-USD' not in s]
            
            all_data = pd.DataFrame()
            
            # Load regular symbols
            if regular_symbols:
                try:
                    regular_data = yf.download(regular_symbols, start=start_date, end=end_date, progress=False)
                    if not regular_data.empty:
                        if len(regular_symbols) == 1:
                            if 'Adj Close' in regular_data.columns:
                                all_data[regular_symbols[0]] = regular_data['Adj Close']
                            else:
                                all_data[regular_symbols[0]] = regular_data['Close']
                        else:
                            if 'Adj Close' in regular_data.columns:
                                all_data = pd.concat([all_data, regular_data['Adj Close']], axis=1)
                            else:
                                all_data = pd.concat([all_data, regular_data['Close']], axis=1)
                except Exception as e:
                    st.warning(f"Error loading regular symbols: {e}")
            
            # Load crypto symbols
            if crypto_symbols:
                try:
                    crypto_data = yf.download(crypto_symbols, start=start_date, end=end_date, progress=False)
                    if not crypto_data.empty:
                        if len(crypto_symbols) == 1:
                            all_data[crypto_symbols[0]] = crypto_data['Close']
                        else:
                            if isinstance(crypto_data.columns, pd.MultiIndex):
                                all_data = pd.concat([all_data, crypto_data['Close']], axis=1)
                            else:
                                all_data[crypto_symbols[0]] = crypto_data['Close']
                except Exception as e:
                    st.warning(f"Error loading crypto symbols: {e}")
            
            if not all_data.empty:
                # Ensure we have enough data points
                if min_data_points and len(all_data) < min_data_points:
                    st.warning(f"Insufficient data: got {len(all_data)} points, need {min_data_points}")
                    # Try to get more data by extending the date range
                    extended_start = start_date - timedelta(days=365)
                    return self.load_live_data(symbols, extended_start, end_date, None)
                
                self.data = all_data
                self.returns = self.data.pct_change().dropna()
                return self.data
            else:
                return None
            
        except Exception as e:
            st.error(f"Error loading market data: {str(e)}")
            return None
    
    def load_csv_data(self, uploaded_file):
        """Load data from uploaded CSV file"""
        try:
            if uploaded_file is not None:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                # Assume first column is date, rest are prices
                if df.shape[1] > 1:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df.set_index(df.columns[0], inplace=True)
                    
                    self.data = df
                    self.returns = self.data.pct_change().dropna()
                    return self.data
                else:
                    st.error("CSV file must contain at least 2 columns (date and price)")
                    return None
                    
        except Exception as e:
            st.error(f"Error loading CSV data: {str(e)}")
            return None
    
    def create_manual_data(self, symbols, prices, dates):
        """Create data manually from user inputs"""
        try:
            data_dict = {}
            
            for i, symbol in enumerate(symbols):
                data_dict[symbol] = prices[i]
            
            self.data = pd.DataFrame(data_dict, index=pd.to_datetime(dates))
            self.returns = self.data.pct_change().dropna()
            return self.data
            
        except Exception as e:
            st.error(f"Error creating manual data: {str(e)}")
            return None
    
    def generate_synthetic_data(self, num_days=500, initial_price=100, annual_return=0.08, annual_volatility=0.20, random_seed=42):
        """Generate synthetic stock price data using geometric Brownian motion"""
        try:
            np.random.seed(random_seed)
            
            # Convert annual parameters to daily
            daily_return = annual_return / 252
            daily_volatility = annual_volatility / np.sqrt(252)
            
            # Generate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=num_days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate price series using geometric Brownian motion
            prices = [initial_price]
            for i in range(1, len(dates)):
                random_shock = np.random.normal(0, 1)
                price_change = prices[-1] * (daily_return + daily_volatility * random_shock)
                new_price = prices[-1] + price_change
                prices.append(max(new_price, 0.01))  # Ensure positive prices
            
            # Create DataFrame
            self.data = pd.DataFrame({'Price': prices}, index=dates)
            self.returns = self.data.pct_change().dropna()
            return self.data
            
        except Exception as e:
            st.error(f"Error generating synthetic data: {str(e)}")
            return None
    
    def get_portfolio_returns(self, weights=None):
        """Calculate portfolio returns based on weights"""
        if self.returns is None:
            return None
        
        if weights is None:
            # Equal weights
            weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
        
        portfolio_returns = self.returns.dot(weights)
        return portfolio_returns
    
    def validate_data(self):
        """Validate loaded data"""
        if self.data is None:
            return False, "No data loaded"
        
        if self.data.empty:
            return False, "Data is empty"
        
        if self.data.isnull().all().any():
            return False, "Data contains only null values"
        
        if len(self.data) < 30:
            return False, "Insufficient data points (minimum 30 required)"
        
        return True, "Data validation passed"
    
    def get_data_summary(self):
        """Get summary statistics of the data"""
        if self.data is None or self.returns is None:
            return None
        
        summary = {
            'data_points': len(self.data),
            'date_range': f"{self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}",
            'assets': list(self.data.columns),
            'returns_mean': self.returns.mean().to_dict(),
            'returns_std': self.returns.std().to_dict(),
            'returns_min': self.returns.min().to_dict(),
            'returns_max': self.returns.max().to_dict()
        }
        
        return summary
    
    def resample_data(self, frequency='D'):
        """Resample data to different frequency"""
        if self.data is None:
            return None
        
        try:
            if frequency == 'D':
                return self.data
            elif frequency == 'W':
                resampled = self.data.resample('W').last()
            elif frequency == 'M':
                resampled = self.data.resample('M').last()
            else:
                return self.data
            
            return resampled
            
        except Exception as e:
            st.error(f"Error resampling data: {str(e)}")
            return None
    
    def handle_missing_data(self, method='forward_fill'):
        """Handle missing data in the dataset"""
        if self.data is None:
            return None
        
        try:
            if method == 'forward_fill':
                self.data = self.data.fillna(method='ffill')
            elif method == 'backward_fill':
                self.data = self.data.fillna(method='bfill')
            elif method == 'interpolate':
                self.data = self.data.interpolate()
            elif method == 'drop':
                self.data = self.data.dropna()
            
            # Recalculate returns
            self.returns = self.data.pct_change().dropna()
            return self.data
            
        except Exception as e:
            st.error(f"Error handling missing data: {str(e)}")
            return None
    
    def get_crypto_symbols(self):
        """Get list of popular crypto symbols for yfinance"""
        return [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
            'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'SHIB-USD',
            'MATIC-USD', 'LTC-USD', 'UNI-USD', 'LINK-USD', 'ATOM-USD'
        ]
    
    def validate_crypto_symbol(self, symbol):
        """Validate if a crypto symbol is properly formatted"""
        return symbol.upper().endswith('-USD')
    
    def load_mixed_portfolio(self, regular_symbols, crypto_symbols, start_date, end_date):
        """Load a mixed portfolio of regular stocks and crypto"""
        try:
            all_symbols = regular_symbols + crypto_symbols
            return self.load_live_data(all_symbols, start_date, end_date)
        except Exception as e:
            st.error(f"Error loading mixed portfolio: {str(e)}")
            return None
    
    def get_sufficient_data_for_backtesting(self, symbols, backtest_days=198):
        """Get sufficient data for backtesting requirements"""
        try:
            # Calculate required data points: backtesting days + window for VaR calculation + buffer
            required_points = backtest_days + 252 + 50  # 500 total points for safety
            
            # Calculate start date to get enough data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=required_points + 150)  # Add extra buffer for weekends/holidays
            
            return self.load_live_data(symbols, start_date, end_date, required_points)
            
        except Exception as e:
            st.error(f"Error getting sufficient data for backtesting: {str(e)}")
            return None
