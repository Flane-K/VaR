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
    
    def load_live_data(self, symbols, start_date, end_date):
        """Load live market data from Yahoo Finance"""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            
            if len(symbols) == 1:
                # Single asset
                self.data = data[['Adj Close']].copy()
                self.data.columns = [symbols[0]]
            else:
                # Multiple assets
                self.data = data['Adj Close'].copy()
            
            self.returns = self.data.pct_change().dropna()
            return self.data
            
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