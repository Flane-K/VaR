import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import json
from scipy.stats import norm
import streamlit as st

class OptionsDataFetcher:
    """Enhanced options data fetcher with rate limiting, caching, and fallback mechanisms"""
    
    def __init__(self):
        self.cache = {}
        self.last_request_time = {}
        self.min_request_interval = 2  # Minimum seconds between requests
        
    def _rate_limit_check(self, symbol):
        """Check and enforce rate limiting"""
        current_time = time.time()
        if symbol in self.last_request_time:
            time_since_last = current_time - self.last_request_time[symbol]
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
        
        self.last_request_time[symbol] = time.time()
    
    def _get_cache_key(self, symbol, expiry_date=None):
        """Generate cache key for options data"""
        if expiry_date:
            return f"{symbol}_{expiry_date.strftime('%Y-%m-%d')}"
        return f"{symbol}_all"
    
    def _is_cache_valid(self, cache_key, max_age_minutes=30):
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key].get('timestamp', 0)
        current_time = time.time()
        age_minutes = (current_time - cache_time) / 60
        
        return age_minutes < max_age_minutes
    
    def fetch_options_data_yfinance(self, symbol, max_retries=3):
        """Fetch options data using yfinance with retry logic"""
        cache_key = self._get_cache_key(symbol)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        for attempt in range(max_retries):
            try:
                self._rate_limit_check(symbol)
                
                ticker = yf.Ticker(symbol)
                
                # Get options expiration dates
                expiry_dates = ticker.options
                
                if not expiry_dates:
                    raise Exception("No options expiration dates found")
                
                # Get current stock price
                info = ticker.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 100))
                
                options_data = {
                    'current_price': current_price,
                    'expiry_dates': list(expiry_dates),
                    'options_chains': {}
                }
                
                # Fetch options chain for each expiry (limit to first 5 to avoid rate limiting)
                for expiry in expiry_dates[:5]:
                    try:
                        self._rate_limit_check(symbol)
                        chain = ticker.option_chain(expiry)
                        
                        options_data['options_chains'][expiry] = {
                            'calls': chain.calls,
                            'puts': chain.puts
                        }
                        
                        # Small delay between requests
                        time.sleep(0.5)
                        
                    except Exception as e:
                        st.warning(f"Could not fetch options chain for {expiry}: {str(e)}")
                        continue
                
                # Cache the result
                self.cache[cache_key] = {
                    'data': options_data,
                    'timestamp': time.time()
                }
                
                return options_data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    st.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.error(f"Failed to fetch options data after {max_retries} attempts: {str(e)}")
                    return None
        
        return None
    
    def generate_synthetic_options_data(self, symbol, current_price=None):
        """Generate realistic synthetic options data when live data fails"""
        if current_price is None:
            # Try to get current price, fallback to default
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 150))
            except:
                current_price = 150  # Default fallback
        
        # Generate expiry dates (weekly and monthly)
        today = datetime.now().date()
        expiry_dates = []
        
        # Add weekly expiries for next 8 weeks
        for i in range(1, 9):
            friday = today + timedelta(days=(4 - today.weekday()) % 7 + 7 * i)
            expiry_dates.append(friday.strftime('%Y-%m-%d'))
        
        # Add monthly expiries for next 12 months
        for i in range(1, 13):
            if i <= 12:
                month = today.month + i
                year = today.year
                if month > 12:
                    month -= 12
                    year += 1
                
                # Third Friday of the month
                first_day = datetime(year, month, 1).date()
                first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
                third_friday = first_friday + timedelta(days=14)
                
                if third_friday not in [datetime.strptime(d, '%Y-%m-%d').date() for d in expiry_dates]:
                    expiry_dates.append(third_friday.strftime('%Y-%m-%d'))
        
        options_data = {
            'current_price': current_price,
            'expiry_dates': expiry_dates,
            'options_chains': {}
        }
        
        # Generate options chains for each expiry
        for expiry_str in expiry_dates:
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            days_to_expiry = (expiry_date - today).days
            
            if days_to_expiry <= 0:
                continue
            
            time_to_expiry = days_to_expiry / 365.0
            
            # Generate strike prices around current price
            strike_range = np.arange(
                current_price * 0.7, 
                current_price * 1.3, 
                current_price * 0.025
            )
            strike_range = np.round(strike_range, 2)
            
            calls_data = []
            puts_data = []
            
            for strike in strike_range:
                # Calculate theoretical option prices using Black-Scholes
                risk_free_rate = 0.05
                volatility = 0.25
                
                # Black-Scholes calculations
                d1 = (np.log(current_price / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
                d2 = d1 - volatility * np.sqrt(time_to_expiry)
                
                call_price = current_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
                put_price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - current_price * norm.cdf(-d1)
                
                # Add some realistic bid-ask spread and volume
                spread_factor = 0.02
                call_bid = max(0.01, call_price * (1 - spread_factor))
                call_ask = call_price * (1 + spread_factor)
                put_bid = max(0.01, put_price * (1 - spread_factor))
                put_ask = put_price * (1 + spread_factor)
                
                # Generate realistic volume (higher for ATM options)
                atm_factor = np.exp(-0.5 * ((strike - current_price) / (current_price * 0.1))**2)
                base_volume = int(100 * atm_factor * np.random.uniform(0.5, 2.0))
                
                calls_data.append({
                    'contractSymbol': f"{symbol}{expiry_str.replace('-', '')[:6]}C{int(strike*1000):08d}",
                    'strike': strike,
                    'lastPrice': round(call_price, 2),
                    'bid': round(call_bid, 2),
                    'ask': round(call_ask, 2),
                    'volume': base_volume,
                    'openInterest': base_volume * 2,
                    'impliedVolatility': round(volatility + np.random.uniform(-0.05, 0.05), 4),
                    'inTheMoney': strike < current_price,
                    'contractSize': 'REGULAR',
                    'currency': 'USD'
                })
                
                puts_data.append({
                    'contractSymbol': f"{symbol}{expiry_str.replace('-', '')[:6]}P{int(strike*1000):08d}",
                    'strike': strike,
                    'lastPrice': round(put_price, 2),
                    'bid': round(put_bid, 2),
                    'ask': round(put_ask, 2),
                    'volume': base_volume,
                    'openInterest': base_volume * 2,
                    'impliedVolatility': round(volatility + np.random.uniform(-0.05, 0.05), 4),
                    'inTheMoney': strike > current_price,
                    'contractSize': 'REGULAR',
                    'currency': 'USD'
                })
            
            options_data['options_chains'][expiry_str] = {
                'calls': pd.DataFrame(calls_data),
                'puts': pd.DataFrame(puts_data)
            }
        
        return options_data
    
    def get_options_data(self, symbol, use_synthetic=False):
        """Main method to get options data with fallback to synthetic data"""
        if use_synthetic:
            st.info("Using synthetic options data for demonstration")
            return self.generate_synthetic_options_data(symbol)
        
        # Try to fetch live data first
        options_data = self.fetch_options_data_yfinance(symbol)
        
        if options_data is None or not options_data.get('options_chains'):
            st.warning("Live options data unavailable. Generating synthetic data for demonstration.")
            return self.generate_synthetic_options_data(symbol)
        
        return options_data
    
    def find_closest_atm_option(self, options_data, option_type='call', target_dte=None):
        """Find the closest at-the-money option"""
        current_price = options_data['current_price']
        best_option = None
        min_distance = float('inf')
        
        for expiry, chains in options_data['options_chains'].items():
            if target_dte:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                days_to_expiry = (expiry_date - datetime.now().date()).days
                if abs(days_to_expiry - target_dte) > 30:  # Allow 30-day tolerance
                    continue
            
            chain = chains['calls'] if option_type.lower() == 'call' else chains['puts']
            
            if chain.empty:
                continue
            
            # Find closest strike to current price
            chain['strike_distance'] = abs(chain['strike'] - current_price)
            closest_strike_option = chain.loc[chain['strike_distance'].idxmin()]
            
            distance = closest_strike_option['strike_distance']
            if distance < min_distance:
                min_distance = distance
                best_option = {
                    'expiry': expiry,
                    'strike': closest_strike_option['strike'],
                    'option_type': option_type,
                    'price': closest_strike_option['lastPrice'],
                    'bid': closest_strike_option['bid'],
                    'ask': closest_strike_option['ask'],
                    'volume': closest_strike_option['volume'],
                    'implied_volatility': closest_strike_option['impliedVolatility'],
                    'contract_symbol': closest_strike_option['contractSymbol']
                }
        
        return best_option
    
    def get_option_by_criteria(self, options_data, strike_price=None, expiry_date=None, option_type='call'):
        """Get specific option by strike and expiry criteria"""
        current_price = options_data['current_price']
        
        # If no specific criteria, return closest ATM option
        if strike_price is None and expiry_date is None:
            return self.find_closest_atm_option(options_data, option_type, target_dte=180)
        
        best_option = None
        min_score = float('inf')
        
        for expiry, chains in options_data['options_chains'].items():
            chain = chains['calls'] if option_type.lower() == 'call' else chains['puts']
            
            if chain.empty:
                continue
            
            # Calculate score based on criteria
            for _, option in chain.iterrows():
                score = 0
                
                # Strike price score
                if strike_price is not None:
                    strike_diff = abs(option['strike'] - strike_price)
                    score += strike_diff
                
                # Expiry date score
                if expiry_date is not None:
                    target_expiry = expiry_date.strftime('%Y-%m-%d') if hasattr(expiry_date, 'strftime') else str(expiry_date)
                    expiry_diff = abs((datetime.strptime(expiry, '%Y-%m-%d') - datetime.strptime(target_expiry, '%Y-%m-%d')).days)
                    score += expiry_diff * 0.1  # Weight expiry difference less than strike
                
                if score < min_score:
                    min_score = score
                    best_option = {
                        'expiry': expiry,
                        'strike': option['strike'],
                        'option_type': option_type,
                        'price': option['lastPrice'],
                        'bid': option['bid'],
                        'ask': option['ask'],
                        'volume': option['volume'],
                        'implied_volatility': option['impliedVolatility'],
                        'contract_symbol': option['contractSymbol']
                    }
        
        return best_option
