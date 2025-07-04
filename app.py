import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.option_data_fetcher import OptionsDataFetcher
from src.data_ingestion import DataIngestion
from src.var_engines import VaREngines
from src.backtesting import Backtesting
from src.stress_testing import StressTesting
from src.rolling_analysis import RollingAnalysis
from src.options_var import OptionsVaR
from src.visualization import Visualization
from src.utils import Utils

# Page configuration
st.set_page_config(
    page_title="VaR & Risk Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff88;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff88;
    }
    .options-config {
        background-color: #1e3a8a20;
        border: 2px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #ffd93d20;
        border: 1px solid #ffd93d;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #6bcf7f20;
        border: 1px solid #6bcf7f;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    try:
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0)
    except:
        return 0

def calculate_option_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    try:
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    except:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

def generate_option_synthetic_data(S0, K, T, r, sigma, option_type, underlying_symbol, num_days=252):
    """Generate synthetic option price data using Black-Scholes"""
    try:
        dates = pd.date_range(start=datetime.now() - timedelta(days=num_days), 
                             end=datetime.now(), freq='D')

        # Generate underlying price path using GBM
        dt = 1/252
        returns = np.random.normal((r - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), num_days)

        underlying_prices = [S0]
        for ret in returns[1:]:
            underlying_prices.append(underlying_prices[-1] * np.exp(ret))

        # Calculate option prices for each underlying price
        option_prices = []
        for i, S in enumerate(underlying_prices):
            time_to_expiry = T - (i * dt)
            if time_to_expiry > 0:
                option_price = black_scholes_price(S, K, time_to_expiry, r, sigma, option_type)
            else:
                if option_type == 'call':
                    option_price = max(S - K, 0)
                else:
                    option_price = max(K - S, 0)
            option_prices.append(option_price)

        # Create DataFrame with proper column names
        data = pd.DataFrame({
            'Date': dates[:len(option_prices)],
            underlying_symbol: underlying_prices[:len(option_prices)],
            'Option_Price': option_prices
        })
        data.set_index('Date', inplace=True)

        return data
    except Exception as e:
        st.error(f"Error generating synthetic option data: {str(e)}")
        return None

def calculate_options_var_comprehensive(option_returns, confidence_level, method='historical'):
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
        st.error(f"Error calculating options VaR: {str(e)}")
        return 0

def safe_dataframe_display(df, title="Data", key_suffix=""):
    """Safely display dataframe with proper type handling for Arrow compatibility"""
    try:
        if df is None or df.empty:
            st.warning(f"No {title.lower()} available")
            return

        # Create a copy to avoid modifying original
        display_df = df.copy()

        # Convert all columns to appropriate types
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                # Try to convert to numeric, if fails keep as string
                try:
                    display_df[col] = pd.to_numeric(display_df[col], errors='ignore')
                except:
                    display_df[col] = display_df[col].astype(str)

        # Display with unique key
        st.dataframe(display_df, key=f"df_{title.lower().replace(' ', '_')}_{key_suffix}")

    except Exception as e:
        st.error(f"Error displaying {title}: {str(e)}")
        # Fallback to simple display
        try:
            st.write(df)
        except:
            st.write(f"Unable to display {title}")

def create_metrics_dataframe(metrics_dict, title="Metrics"):
    """Create a properly formatted dataframe for metrics display"""
    try:
        if not metrics_dict:
            return pd.DataFrame()

        # Convert metrics to proper format
        formatted_metrics = []
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                if 'percentage' in key.lower() or 'ratio' in key.lower():
                    formatted_value = f"{value:.4f}"
                elif 'var' in key.lower() or 'shortfall' in key.lower():
                    formatted_value = f"${value:,.2f}"
                else:
                    formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)

            formatted_metrics.append({
                'Metric': key.replace('_', ' ').title(),
                'Value': formatted_value
            })

        return pd.DataFrame(formatted_metrics)

    except Exception as e:
        st.error(f"Error creating metrics dataframe: {str(e)}")
        return pd.DataFrame()

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'current_returns' not in st.session_state:
        st.session_state.current_returns = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "Parametric (Delta-Normal)"
    if 'var_results' not in st.session_state:
        st.session_state.var_results = {}
    if 'model_changed' not in st.session_state:
        st.session_state.model_changed = False
    if 'options_data' not in st.session_state:
        st.session_state.options_data = None
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None
    if 'options_expirations' not in st.session_state:
        st.session_state.options_expirations = None
    if 'option_params' not in st.session_state:
        st.session_state.option_params = None
    if 'symbols' not in st.session_state:
        st.session_state.symbols = []
    if 'weights' not in st.session_state:
        st.session_state.weights = []
    # Track current configuration to detect changes
    if 'current_portfolio_type' not in st.session_state:
        st.session_state.current_portfolio_type = None
    if 'current_data_source' not in st.session_state:
        st.session_state.current_data_source = None

def reset_data_on_config_change(portfolio_type, data_source):
    """Reset session state when portfolio type or data source changes"""
    config_changed = False

    if st.session_state.current_portfolio_type != portfolio_type:
        st.session_state.current_portfolio_type = portfolio_type
        config_changed = True

    if st.session_state.current_data_source != data_source:
        st.session_state.current_data_source = data_source
        config_changed = True

    if config_changed:
        # Reset data-related session state
        st.session_state.data_loaded = False
        st.session_state.current_data = None
        st.session_state.current_returns = None
        st.session_state.options_data = None
        st.session_state.selected_option = None
        st.session_state.options_expirations = None
        st.session_state.option_params = None
        st.session_state.symbols = []
        st.session_state.weights = []
        st.session_state.var_results = {}

        if config_changed:
            st.info("Configuration changed. Please load data again.")

def main():
    # Initialize session state
    initialize_session_state()

    # Main header
    st.markdown('<h1 class="main-header">📊 VaR & Risk Analytics Platform</h1>', unsafe_allow_html=True)

    # Initialize instances
    instances = {
        'data_ingestion': DataIngestion(),
        'var_engines': VaREngines(),
        'backtesting': Backtesting(),
        'stress_testing': StressTesting(),
        'rolling_analysis': RollingAnalysis(),
        'options_var': OptionsVaR(),
        'visualization': Visualization(),
        'utils': Utils(),
        'options_fetcher': OptionsDataFetcher()
    }

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Portfolio Type Selection (moved to top)
        st.subheader("💼 Portfolio Type")
        portfolio_type = st.selectbox(
            "Select Portfolio Type",
            ["Single Asset", "Multi-Asset", "Crypto Portfolio", "Options Portfolio"],
            key="portfolio_type_select"
        )

        # Data Source Selection (conditional - not shown for Options Portfolio)
        if portfolio_type != "Options Portfolio":
            st.subheader("📊 Data Source")
            data_source = st.selectbox(
                "Select Data Source",
                ["Live Market Data", "Upload File", "Manual Entry", "Synthetic Data"],
                key="data_source_select"
            )
        else:
            # For options portfolio, we'll handle data source internally
            data_source = "Options Data"

        # Check for configuration changes and reset if needed
        reset_data_on_config_change(portfolio_type, data_source)

        # VaR Model Selection
        st.subheader("🎯 VaR Model")
        if portfolio_type == "Options Portfolio":
            var_model = st.selectbox(
                "Select Options VaR Model",
                [
                    "Historical Simulation",
                    "Parametric (Delta-Normal)", 
                    "Monte Carlo"
                ],
                key="var_model_select"
            )
        else:
            var_model = st.selectbox(
                "Select VaR Model",
                [
                    "Parametric (Delta-Normal)",
                    "Historical Simulation", 
                    "Monte Carlo",
                    "GARCH",
                    "Extreme Value Theory (EVT)"
                ],
                key="var_model_select"
            )

        # Check if model changed
        if var_model != st.session_state.current_model:
            st.session_state.model_changed = True
            st.session_state.current_model = var_model

        # Risk Parameters
        st.subheader("📈 Risk Parameters")
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key="confidence_slider")
        time_horizon = st.slider("Time Horizon (days)", 1, 30, 1, key="time_horizon_slider")

        # Date Range (Default 1 year)
        st.subheader("📅 Date Range")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Default 1 year

        date_start = st.date_input("Start Date", start_date, key="start_date_input")
        date_end = st.date_input("End Date", end_date, key="end_date_input")

        # Model-specific parameters
        if var_model == "Monte Carlo":
            st.subheader("🎲 Monte Carlo Parameters")
            num_simulations = st.slider("Number of Simulations", 1000, 100000, 10000, 1000, key="mc_sims_slider")

        if var_model == "GARCH":
            st.subheader("📊 GARCH Parameters")
            garch_p = st.slider("GARCH P", 1, 3, 1, key="garch_p_slider")
            garch_q = st.slider("GARCH Q", 1, 3, 1, key="garch_q_slider")

        # Portfolio Type Specific Configuration
        if portfolio_type == "Options Portfolio":
            st.markdown('<div class="options-config">', unsafe_allow_html=True)
            st.subheader("📊 Options Configuration")

            # Options data source selection
            options_data_source = st.selectbox(
                "Options Data Source",
                ["Live Market Data", "Manual Entry"],
                key="options_data_source_select"
            )

            # Initialize default values for options parameters
            underlying = "AAPL"
            spot_price = 150.0
            strike_price = 155.0
            time_to_expiry = 0.25
            risk_free_rate = 0.05
            volatility = 0.25
            option_type_str = "Call"
            quantity = 100

            if options_data_source == "Live Market Data":
                st.write("**Live Market Options:**")
                underlying = st.text_input("Underlying Symbol", "AAPL", key="options_underlying_input")

                # Fetch options data button
                if st.button("🔄 Fetch Options Data", key="fetch_options_button"):
                    with st.spinner("Fetching options data..."):
                        try:
                            options_data = instances['options_fetcher'].get_options_data(underlying, use_synthetic=False)
                            if options_data and options_data.get('options_chains'):
                                st.session_state.options_data = options_data
                                st.session_state.options_expirations = options_data.get('expiry_dates', [])
                                spot_price = options_data.get('current_price', 150.0)
                                st.success(f"✅ Fetched options data for {underlying}")
                            else:
                                st.warning(f"Could not fetch live options data for {underlying}. Using synthetic data.")
                                options_data = instances['options_fetcher'].generate_synthetic_options_data(underlying, spot_price)
                                st.session_state.options_data = options_data
                                st.session_state.options_expirations = options_data.get('expiry_dates', [])
                        except Exception as e:
                            st.error(f"Error fetching options data: {str(e)}")
                            st.info("Generating synthetic options data for demonstration.")
                            options_data = instances['options_fetcher'].generate_synthetic_options_data(underlying, spot_price)
                            st.session_state.options_data = options_data
                            st.session_state.options_expirations = options_data.get('expiry_dates', [])

                # Options selection if data is available
                if hasattr(st.session_state, 'options_data') and st.session_state.options_data is not None:
                    option_type_str = st.selectbox("Option Type", ["Call", "Put"], key="live_option_type_select")

                    # Get available expiry dates
                    available_expiries = list(st.session_state.options_data.get('options_chains', {}).keys())
                    if available_expiries:
                        selected_expiry = st.selectbox(
                            "Expiry Date",
                            available_expiries[:5],  # Show first 5 expiries
                            key="live_expiry_select"
                        )

                        # Get the appropriate options dataframe
                        chain_data = st.session_state.options_data['options_chains'].get(selected_expiry, {})
                        if option_type_str == "Call":
                            options_df = chain_data.get('calls', pd.DataFrame())
                        else:
                            options_df = chain_data.get('puts', pd.DataFrame())

                        if not options_df.empty:
                            # Strike selection
                            available_strikes = sorted(options_df['strike'].unique())
                            default_strike_idx = len(available_strikes) // 2  # Middle strike as default

                            selected_strike = st.selectbox(
                                "Strike Price",
                                available_strikes,
                                index=default_strike_idx,
                                key="live_strike_select"
                            )

                            # Find the selected option
                            selected_option_data = options_df[options_df['strike'] == selected_strike].iloc[0]

                            # Display option details
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Strike:** ${selected_strike}")
                                st.write(f"**Last Price:** ${selected_option_data.get('lastPrice', 0):.2f}")
                            with col2:
                                st.write(f"**Bid:** ${selected_option_data.get('bid', 0):.2f}")
                                st.write(f"**Ask:** ${selected_option_data.get('ask', 0):.2f}")

                            # Set parameters for calculations
                            spot_price = st.session_state.options_data.get('current_price', 150.0)
                            strike_price = selected_strike
                            # Calculate time to expiry
                            expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d').date()
                            days_to_expiry = (expiry_date - datetime.now().date()).days
                            time_to_expiry = max(days_to_expiry / 365.0, 0.01)  # Minimum 1 day
                            risk_free_rate = 0.05
                            volatility = selected_option_data.get('impliedVolatility', 0.25)
                            quantity = st.number_input("Quantity", 1, 1000, 100, 1, key="live_quantity_input")
                        else:
                            st.warning("No options data available for the selected type and expiry")
                    else:
                        st.warning("No expiry dates available")

            if options_data_source == "Manual Entry":
                st.write("**Manual Options Entry:**")
                underlying = st.text_input("Underlying Symbol", "AAPL", key="manual_underlying_input")
                spot_price = st.number_input("Current Spot Price ($)", 50.0, 1000.0, 150.0, 1.0, key="manual_spot_price_input")
                strike_price = st.number_input("Strike Price ($)", 50.0, 1000.0, 155.0, 1.0, key="manual_strike_price_input")
                time_to_expiry = st.number_input("Time to Expiry (years)", 0.01, 2.0, 0.25, 0.01, key="manual_time_expiry_input")
                risk_free_rate = st.number_input("Risk-free Rate", 0.0, 0.1, 0.05, 0.001, key="manual_risk_free_input")
                volatility = st.number_input("Volatility", 0.1, 1.0, 0.25, 0.01, key="manual_volatility_input")
                option_type_str = st.selectbox("Option Type", ["Call", "Put"], key="manual_option_type_select")
                quantity = st.number_input("Quantity", 1, 1000, 100, 1, key="manual_quantity_input")

            symbols = [underlying]
            weights = [1.0]

            st.markdown('</div>', unsafe_allow_html=True)

        elif data_source == "Live Market Data":
            st.subheader("🔗 Market Data Configuration")

            if portfolio_type == "Single Asset":
                symbols_input = st.text_input("Symbol", "AAPL", key="single_symbol_input")
                symbols = [symbols_input.strip().upper()]
                weights = [1.0]

            elif portfolio_type == "Multi-Asset":
                symbols_input = st.text_input("Symbols (comma-separated)", "AAPL,GOOGL,MSFT,TSLA", key="multi_symbols_input")
                symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

                # Crypto symbols input
                crypto_input = st.text_input("Crypto Symbols (comma-separated, optional)", "", 
                                           help="Add crypto symbols like BTC-USD, ETH-USD", key="crypto_symbols_input")
                if crypto_input.strip():
                    crypto_symbols = [s.strip().upper() for s in crypto_input.split(",") if s.strip()]
                    symbols.extend(crypto_symbols)

                # Portfolio weights
                if len(symbols) > 1:
                    st.write("Portfolio Weights:")
                    weights = []
                    for i, symbol in enumerate(symbols):
                        weight = st.number_input(f"{symbol}", 0.0, 1.0, 1.0/len(symbols), 0.01, key=f"weight_{symbol}_{i}")
                        weights.append(weight)

                    # Normalize weights
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w/total_weight for w in weights]
                else:
                    weights = [1.0]

            elif portfolio_type == "Crypto Portfolio":
                symbols_input = st.text_input("Crypto Symbols", "BTC-USD", key="crypto_portfolio_input")
                symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
                weights = [1.0/len(symbols) for _ in symbols] if len(symbols) > 1 else [1.0]

        elif data_source == "Synthetic Data":
            st.subheader("🎲 Synthetic Data Parameters")

            # Expandable section for custom parameters
            use_custom = st.checkbox("Customize Parameters", key="custom_synthetic_checkbox")

            if use_custom:
                num_days = st.slider("Number of Days", 100, 2000, 500, key="synthetic_days_slider")
                initial_price = st.number_input("Initial Price ($)", 10.0, 1000.0, 100.0, key="synthetic_price_input")
                annual_return = st.slider("Annual Return", -0.5, 0.5, 0.08, 0.01, key="synthetic_return_slider")
                annual_volatility = st.slider("Annual Volatility", 0.05, 1.0, 0.20, 0.01, key="synthetic_vol_slider")
                random_seed = st.number_input("Random Seed", 1, 1000, 42, key="synthetic_seed_input")
            else:
                # Default parameters for good representative data
                num_days = 500
                initial_price = 100.0
                annual_return = 0.08
                annual_volatility = 0.20
                random_seed = 42

            symbols = ["Synthetic_Asset"]
            weights = [1.0]

        # Load Data Button
        if st.button("🔄 Load Data", type="primary", key="load_data_button"):
            with st.spinner("Loading data..."):
                try:
                    if portfolio_type == "Options Portfolio":
                        # Generate or load options data
                        data = generate_option_synthetic_data(
                            spot_price, strike_price, time_to_expiry, 
                            risk_free_rate, volatility, option_type_str.lower(), underlying
                        )

                        if data is not None:
                            st.session_state.current_data = data
                            st.session_state.current_returns = data['Option_Price'].pct_change().dropna()
                            st.session_state.data_loaded = True
                            st.session_state.symbols = [f"{underlying}_{option_type_str}_{strike_price}"]
                            st.session_state.weights = [1.0]
                            st.session_state.option_params = {
                                'underlying': underlying,
                                'spot_price': spot_price,
                                'strike_price': strike_price,
                                'time_to_expiry': time_to_expiry,
                                'risk_free_rate': risk_free_rate,
                                'volatility': volatility,
                                'option_type': option_type_str.lower(),
                                'quantity': quantity
                            }
                            st.success(f"✅ Successfully loaded options data for {underlying}")
                        else:
                            st.error("❌ Failed to generate options data")

                    elif data_source == "Live Market Data":
                        # Calculate required data range for backtesting
                        backtesting_window = st.session_state.get('backtesting_window', 252)
                        required_days = backtesting_window + 100  # Extra buffer

                        # Extend start date if needed for backtesting
                        extended_start = min(date_start, date_end - timedelta(days=required_days))

                        data = instances['data_ingestion'].load_live_data(symbols, extended_start, date_end)

                        if data is not None and not data.empty:
                            st.session_state.current_data = data
                            st.session_state.current_returns = instances['data_ingestion'].returns
                            st.session_state.data_loaded = True
                            st.session_state.symbols = symbols
                            st.session_state.weights = weights
                            st.success(f"✅ Successfully loaded data for {len(symbols)} asset(s)")
                        else:
                            st.error("❌ Failed to load market data")

                    elif data_source == "Synthetic Data":
                        data = instances['data_ingestion'].generate_synthetic_data(
                            num_days, initial_price, annual_return, annual_volatility, random_seed
                        )

                        if data is not None:
                            st.session_state.current_data = data
                            st.session_state.current_returns = instances['data_ingestion'].returns
                            st.session_state.data_loaded = True
                            st.session_state.symbols = symbols
                            st.session_state.weights = weights
                            st.success("✅ Successfully generated synthetic data")
                        else:
                            st.error("❌ Failed to generate synthetic data")

                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")

    # Main content area with tabs
    if st.session_state.data_loaded and st.session_state.current_returns is not None:
        # Calculate portfolio returns
        try:
            if len(st.session_state.current_returns.shape) == 1:
                var_portfolio_returns = st.session_state.current_returns
            else:
                var_portfolio_returns = st.session_state.current_returns.dot(st.session_state.weights)

            # Calculate VaR based on selected model and portfolio type
            if portfolio_type == "Options Portfolio":
                # Options-specific VaR calculation
                var_result = calculate_options_var_comprehensive(
                    var_portfolio_returns, confidence_level, var_model.lower().replace(' ', '_')
                )
                expected_shortfall = calculate_options_var_comprehensive(
                    var_portfolio_returns[var_portfolio_returns <= -var_result], 0.5, 'historical'
                )
            else:
                # Standard VaR calculation
                if var_model == "Parametric (Delta-Normal)":
                    var_result = instances['var_engines'].calculate_parametric_var(
                        var_portfolio_returns, confidence_level, time_horizon
                    )
                elif var_model == "Historical Simulation":
                    var_result = instances['var_engines'].calculate_historical_var(
                        var_portfolio_returns, confidence_level, time_horizon
                    )
                elif var_model == "Monte Carlo":
                    var_result = instances['var_engines'].calculate_monte_carlo_var(
                        var_portfolio_returns, confidence_level, time_horizon, 
                        num_simulations if 'num_simulations' in locals() else 10000
                    )
                elif var_model == "GARCH":
                    var_result = instances['var_engines'].calculate_garch_var(
                        var_portfolio_returns, confidence_level, time_horizon,
                        garch_p if 'garch_p' in locals() else 1,
                        garch_q if 'garch_q' in locals() else 1
                    )
                elif var_model == "Extreme Value Theory (EVT)":
                    var_result = instances['var_engines'].calculate_evt_var(
                        var_portfolio_returns, confidence_level
                    )

                # Calculate Expected Shortfall
                expected_shortfall = instances['var_engines'].calculate_expected_shortfall(
                    var_portfolio_returns, confidence_level
                )

            # Store results in session state
            st.session_state.var_results = {
                'var': var_result,
                'expected_shortfall': expected_shortfall,
                'model': var_model,
                'confidence_level': confidence_level
            }

        except Exception as e:
            st.error(f"Error calculating VaR: {str(e)}")
            var_result = 0
            expected_shortfall = 0

        # Create tabs - standardized to 8 tabs for both portfolio types
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Options Dashboard" if portfolio_type == "Options Portfolio" else "📊 Dashboard", 
            "📈 VaR Calculator", 
            "📋 Data Overview", 
            "🔄 Rolling Analysis", 
            "🧪 Backtesting", 
            "⚡ Stress Testing", 
            "⬇️ Export Data", 
            "❓ Help"
        ])

        with tab1:  # Dashboard (Options Dashboard for options portfolio)
            if portfolio_type == "Options Portfolio":
                st.header("📊 Options Risk Analytics Dashboard")

                # Options-specific metrics
                if hasattr(st.session_state, 'option_params') and st.session_state.option_params:
                    params = st.session_state.option_params

                    # Calculate current option price and Greeks
                    current_option_price = black_scholes_price(
                        params['spot_price'], params['strike_price'], 
                        params['time_to_expiry'], params['risk_free_rate'], 
                        params['volatility'], params['option_type']
                    )

                    greeks = calculate_option_greeks(
                        params['spot_price'], params['strike_price'], 
                        params['time_to_expiry'], params['risk_free_rate'], 
                        params['volatility'], params['option_type']
                    )

                    # Key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            label="Options VaR (95%)",
                            value=f"${var_result:,.2f}",
                            delta=f"{var_model}"
                        )

                    with col2:
                        st.metric(
                            label="Current Option Price",
                            value=f"${current_option_price:.2f}",
                            delta=f"{params['option_type'].title()} Option"
                        )

                    with col3:
                        st.metric(
                            label="Delta",
                            value=f"{greeks['delta']:.4f}",
                            delta="Price Sensitivity"
                        )

                    with col4:
                        st.metric(
                            label="Theta (Daily)",
                            value=f"${greeks['theta']:.2f}",
                            delta="Time Decay"
                        )

                    # Additional Greeks
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            label="Gamma",
                            value=f"{greeks['gamma']:.6f}",
                            delta="Delta Sensitivity"
                        )

                    with col2:
                        st.metric(
                            label="Vega",
                            value=f"${greeks['vega']:.2f}",
                            delta="Vol Sensitivity"
                        )

                    with col3:
                        portfolio_value = current_option_price * params['quantity']
                        st.metric(
                            label="Portfolio Value",
                            value=f"${portfolio_value:,.2f}",
                            delta=f"{params['quantity']} contracts"
                        )

                    with col4:
                        var_percentage = (var_result / portfolio_value) * 100 if portfolio_value > 0 else 0
                        st.metric(
                            label="VaR as % of Portfolio",
                            value=f"{var_percentage:.2f}%",
                            delta=f"{time_horizon} day horizon"
                        )
                else:
                    st.warning("No option parameters available. Please load options data first.")
            else:
                st.header("📊 Risk Analytics Dashboard")

                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label="Value at Risk (95%)",
                        value=f"${var_result:,.2f}",
                        delta=f"{var_model}"
                    )

                with col2:
                    st.metric(
                        label="Expected Shortfall",
                        value=f"${expected_shortfall:,.2f}",
                        delta=f"{confidence_level*100:.0f}% confidence"
                    )

                with col3:
                    portfolio_value = 100000  # Assumed portfolio value
                    var_percentage = (var_result / portfolio_value) * 100
                    st.metric(
                        label="VaR as % of Portfolio",
                        value=f"{var_percentage:.2f}%",
                        delta=f"{time_horizon} day horizon"
                    )

                with col4:
                    if len(st.session_state.current_returns) > 0:
                        current_vol = st.session_state.current_returns.std() * np.sqrt(252) * 100
                        st.metric(
                            label="Annualized Volatility",
                            value=f"{current_vol.iloc[0]:.2f}%",
                            delta="Historical"
                        )

            # Time range selector for dashboard charts
            st.subheader("📅 Chart Time Range")
            col1, col2 = st.columns(2)
            with col1:
                chart_start = st.date_input("Chart Start Date", date_start, key="dashboard_chart_start")
            with col2:
                chart_end = st.date_input("Chart End Date", date_end, key="dashboard_chart_end")

            # Filter data for selected time range
            mask = (st.session_state.current_data.index >= pd.to_datetime(chart_start)) & \
                   (st.session_state.current_data.index <= pd.to_datetime(chart_end))
            filtered_data = st.session_state.current_data.loc[mask]

            if portfolio_type == "Options Portfolio":
                if 'Option_Price' in filtered_data.columns:
                    filtered_returns = filtered_data['Option_Price'].pct_change().dropna()
                else:
                    st.error("Option price data not found in the dataset")
                    filtered_returns = pd.Series()
            else:
                filtered_returns = filtered_data.pct_change().dropna()
                if len(filtered_returns.shape) > 1:
                    filtered_returns = filtered_returns.dot(st.session_state.weights)

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                if portfolio_type == "Options Portfolio":
                    st.subheader("Option Value Performance")
                    if not filtered_data.empty and 'Option_Price' in filtered_data.columns:
                        fig_perf = go.Figure()
                        fig_perf.add_trace(go.Scatter(
                            x=filtered_data.index,
                            y=filtered_data['Option_Price'].values,
                            mode='lines',
                            name='Option Value',
                            line=dict(color='#00ff88', width=2)
                        ))

                        fig_perf.update_layout(
                            title="Option Value Over Time",
                            xaxis_title="Date",
                            yaxis_title="Option Value ($)",
                            template="plotly_dark",
                            height=400
                        )

                        st.plotly_chart(fig_perf, use_container_width=True, key="dashboard_option_performance_chart")
                else:
                    st.subheader("Portfolio Performance")
                    if not filtered_returns.empty:
                        cumulative_returns = (1 + filtered_returns).cumprod()

                        fig_perf = go.Figure()
                        fig_perf.add_trace(go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns.values,
                            mode='lines',
                            name='Cumulative Returns',
                            line=dict(color='#00ff88', width=2)
                        ))

                        fig_perf.update_layout(
                            title="Cumulative Portfolio Returns",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return",
                            template="plotly_dark",
                            height=400
                        )

                        st.plotly_chart(fig_perf, use_container_width=True, key="dashboard_performance_chart")

            with col2:
                st.subheader("Returns Distribution")
                if not filtered_returns.empty:
                    fig_dist = instances['visualization'].plot_var_distribution(
                        filtered_returns, confidence_level, var_result
                    )
                    st.plotly_chart(fig_dist, use_container_width=True, key="dashboard_distribution_chart")

            # Portfolio Statistics
            if portfolio_type == "Options Portfolio":
                st.subheader("Options Statistics")
                if hasattr(st.session_state, 'option_params') and st.session_state.option_params:
                    params = st.session_state.option_params
                    option_stats = {
                        'Underlying Symbol': params.get('underlying', 'N/A'),
                        'Option Type': params['option_type'].title(),
                        'Strike Price': f"${params['strike_price']:.2f}",
                        'Current Spot': f"${params['spot_price']:.2f}",
                        'Time to Expiry': f"{params['time_to_expiry']:.2f} years",
                        'Implied Volatility': f"{params['volatility']*100:.1f}%",
                        'Risk-free Rate': f"{params['risk_free_rate']*100:.1f}%",
                        'Quantity': f"{params['quantity']} contracts"
                    }
                    stats_df = create_metrics_dataframe(option_stats, "Options Statistics")
                    safe_dataframe_display(stats_df, "Options Statistics", "dashboard")
            else:
                st.subheader("Portfolio Statistics")
                if not var_portfolio_returns.empty:
                    portfolio_stats = instances['utils'].calculate_portfolio_statistics(var_portfolio_returns)
                    stats_df = create_metrics_dataframe(portfolio_stats, "Portfolio Statistics")
                    safe_dataframe_display(stats_df, "Portfolio Statistics", "dashboard")

        with tab2:  # VaR Calculator
            st.header("📈 VaR Calculator")

            # Time range selector for VaR charts
            st.subheader("📅 Analysis Time Range")
            col1, col2 = st.columns(2)
            with col1:
                var_chart_start = st.date_input("VaR Chart Start Date", date_start, key="var_chart_start")
            with col2:
                var_chart_end = st.date_input("VaR Chart End Date", date_end, key="var_chart_end")

            # Filter data for VaR analysis
            var_mask = (st.session_state.current_data.index >= pd.to_datetime(var_chart_start)) & \
                       (st.session_state.current_data.index <= pd.to_datetime(var_chart_end))
            var_filtered_data = st.session_state.current_data.loc[var_mask]

            if portfolio_type == "Options Portfolio":
                if 'Option_Price' in var_filtered_data.columns:
                    var_filtered_returns = var_filtered_data['Option_Price'].pct_change().dropna()
                else:
                    st.error("Option price data not found")
                    var_filtered_returns = pd.Series()
            else:
                var_filtered_returns = var_filtered_data.pct_change().dropna()
                if len(var_filtered_returns.shape) > 1:
                    var_filtered_returns = var_filtered_returns.dot(st.session_state.weights)

            # VaR Results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("VaR Results")
                var_metrics = {
                    'VaR Model': var_model,
                    'Value at Risk': f"${var_result:,.2f}",
                    'Expected Shortfall': f"${expected_shortfall:,.2f}",
                    'Confidence Level': f"{confidence_level*100:.0f}%",
                    'Time Horizon': f"{time_horizon} day(s)"
                }

                var_df = create_metrics_dataframe(var_metrics, "VaR Results")
                safe_dataframe_display(var_df, "VaR Results", "var_calculator")

            with col2:
                st.subheader("Model Comparison")
                # Calculate VaR using all methods for comparison
                comparison_results = {}

                try:
                    if portfolio_type == "Options Portfolio" and not var_filtered_returns.empty:
                        comparison_results['Historical'] = calculate_options_var_comprehensive(
                            var_filtered_returns, confidence_level, 'historical'
                        )
                        comparison_results['Parametric'] = calculate_options_var_comprehensive(
                            var_filtered_returns, confidence_level, 'parametric'
                        )
                        comparison_results['Monte Carlo'] = calculate_options_var_comprehensive(
                            var_filtered_returns, confidence_level, 'monte_carlo'
                        )
                    elif not var_filtered_returns.empty:
                        comparison_results['Parametric'] = instances['var_engines'].calculate_parametric_var(
                            var_filtered_returns, confidence_level, time_horizon
                        )
                        comparison_results['Historical'] = instances['var_engines'].calculate_historical_var(
                            var_filtered_returns, confidence_level, time_horizon
                        )
                        comparison_results['Monte Carlo'] = instances['var_engines'].calculate_monte_carlo_var(
                            var_filtered_returns, confidence_level, time_horizon, 5000
                        )

                    if comparison_results:
                        comparison_df = create_metrics_dataframe(comparison_results, "Model Comparison")
                        safe_dataframe_display(comparison_df, "Model Comparison", "var_comparison")

                except Exception as e:
                    st.warning(f"Error in model comparison: {str(e)}")

            # VaR Distribution Plot
            st.subheader("Returns Distribution with VaR Threshold")
            if not var_filtered_returns.empty:
                fig_var_dist = instances['visualization'].plot_var_distribution(
                    var_filtered_returns, confidence_level, var_result
                )
                st.plotly_chart(fig_var_dist, use_container_width=True, key="var_calculator_distribution")

        with tab3:  # Data Overview
            st.header("📋 Data Overview")

            # Data Summary
            st.subheader("📊 Data Summary")
            if portfolio_type == "Options Portfolio":
                if hasattr(st.session_state, 'option_params') and st.session_state.option_params:
                    params = st.session_state.option_params
                    data_summary = {
                        'Data Type': 'Options Data',
                        'Underlying': params.get('underlying', 'N/A'),
                        'Option Type': params['option_type'].title(),
                        'Data Points': len(st.session_state.current_data),
                        'Date Range': f"{st.session_state.current_data.index[0].strftime('%Y-%m-%d')} to {st.session_state.current_data.index[-1].strftime('%Y-%m-%d')}"
                    }
                    summary_df = create_metrics_dataframe(data_summary, "Options Data Summary")
                    safe_dataframe_display(summary_df, "Options Data Summary", "data_overview")
            else:
                data_summary = instances['data_ingestion'].get_data_summary()
                if data_summary:
                    summary_metrics = {
                        'Data Points': data_summary['data_points'],
                        'Date Range': data_summary['date_range'],
                        'Number of Assets': len(data_summary['assets']),
                        'Assets': ', '.join(data_summary['assets'])
                    }
                    summary_df = create_metrics_dataframe(summary_metrics, "Data Summary")
                    safe_dataframe_display(summary_df, "Data Summary", "data_overview")

            # Raw Data Display
            if portfolio_type == "Options Portfolio":
                st.subheader("📈 Options Data")
            else:
                st.subheader("📈 Price Data")

            if st.session_state.current_data is not None:
                # Time range selector for data display
                col1, col2 = st.columns(2)
                with col1:
                    data_display_start = st.date_input("Data Display Start", date_start, key="data_display_start")
                with col2:
                    data_display_end = st.date_input("Data Display End", date_end, key="data_display_end")

                # Filter data for display
                display_mask = (st.session_state.current_data.index >= pd.to_datetime(data_display_start)) & \
                              (st.session_state.current_data.index <= pd.to_datetime(data_display_end))
                display_data = st.session_state.current_data.loc[display_mask]

                safe_dataframe_display(display_data.tail(20), "Recent Data", "price_data")

            # Returns Data
            st.subheader("📊 Returns Data")
            if st.session_state.current_returns is not None:
                display_returns_mask = (st.session_state.current_returns.index >= pd.to_datetime(data_display_start)) & \
                                      (st.session_state.current_returns.index <= pd.to_datetime(data_display_end))
                display_returns = st.session_state.current_returns.loc[display_returns_mask]

                safe_dataframe_display(display_returns.tail(20), "Recent Returns Data", "returns_data")

            # Portfolio Composition
            if hasattr(st.session_state, 'weights') and hasattr(st.session_state, 'symbols'):
                if portfolio_type == "Options Portfolio":
                    st.subheader("📊 Options Position")
                else:
                    st.subheader("💼 Portfolio Composition")

                portfolio_composition = pd.DataFrame({
                    'Asset': st.session_state.symbols,
                    'Weight': [f"{w:.4f}" for w in st.session_state.weights],
                    'Weight (%)': [f"{w*100:.2f}%" for w in st.session_state.weights]
                })
                safe_dataframe_display(portfolio_composition, "Portfolio Composition", "portfolio_weights")

        with tab4:  # Rolling Analysis
            st.header("🔄 Rolling Analysis")

            # Time range selector for rolling analysis
            st.subheader("📅 Rolling Analysis Time Range")
            col1, col2 = st.columns(2)
            with col1:
                rolling_start = st.date_input("Rolling Start Date", date_start, key="rolling_start")
            with col2:
                rolling_end = st.date_input("Rolling End Date", date_end, key="rolling_end")

            # Rolling window parameter
            rolling_window = st.slider("Rolling Window (days)", 30, 252, 60, key="rolling_window_slider")

            # Filter data for rolling analysis
            rolling_mask = (st.session_state.current_data.index >= pd.to_datetime(rolling_start)) & \
                          (st.session_state.current_data.index <= pd.to_datetime(rolling_end))
            rolling_filtered_data = st.session_state.current_data.loc[rolling_mask]

            if portfolio_type == "Options Portfolio":
                if 'Option_Price' in rolling_filtered_data.columns:
                    rolling_portfolio_returns = rolling_filtered_data['Option_Price'].pct_change().dropna()
                else:
                    st.error("Option price data not found")
                    rolling_portfolio_returns = pd.Series()
            else:
                rolling_filtered_returns = rolling_filtered_data.pct_change().dropna()
                if len(rolling_filtered_returns.shape) > 1:
                    rolling_portfolio_returns = rolling_filtered_returns.dot(st.session_state.weights)
                else:
                    rolling_portfolio_returns = rolling_filtered_returns

            if len(rolling_portfolio_returns) > rolling_window:
                # Calculate rolling metrics
                rolling_var = instances['rolling_analysis'].calculate_rolling_var(
                    rolling_portfolio_returns, confidence_level, rolling_window
                )
                rolling_vol = instances['rolling_analysis'].calculate_rolling_volatility(
                    rolling_portfolio_returns, rolling_window
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Rolling VaR")
                    if not rolling_var.empty:
                        fig_rolling_var = instances['visualization'].plot_rolling_metrics(
                            rolling_var, "Rolling VaR"
                        )
                        st.plotly_chart(fig_rolling_var, use_container_width=True, key="rolling_var_chart")

                with col2:
                    st.subheader("Rolling Volatility")
                    if not rolling_vol.empty:
                        fig_rolling_vol = instances['visualization'].plot_rolling_metrics(
                            rolling_vol, "Rolling Volatility"
                        )
                        st.plotly_chart(fig_rolling_vol, use_container_width=True, key="rolling_vol_chart")

                # Drawdown Analysis
                st.subheader("Drawdown Analysis")
                drawdown_data = instances['rolling_analysis'].calculate_maximum_drawdown(rolling_portfolio_returns)
                if drawdown_data and 'drawdown_series' in drawdown_data:
                    fig_drawdown = instances['visualization'].plot_drawdown(drawdown_data['drawdown_series'])
                    st.plotly_chart(fig_drawdown, use_container_width=True, key="drawdown_chart")

                    # Drawdown metrics
                    drawdown_metrics = {
                        'Maximum Drawdown': f"{drawdown_data['max_drawdown']*100:.2f}%",
                        'Drawdown Date': str(drawdown_data.get('max_drawdown_date', 'N/A')),
                        'Peak Date': str(drawdown_data.get('peak_date', 'N/A'))
                    }
                    drawdown_df = create_metrics_dataframe(drawdown_metrics, "Drawdown Metrics")
                    safe_dataframe_display(drawdown_df, "Drawdown Metrics", "drawdown_metrics")
            else:
                st.warning(f"Insufficient data for rolling analysis. Need at least {rolling_window} data points.")

        with tab5:  # Backtesting
            st.header("🧪 Backtesting")

            # Backtesting parameters
            col1, col2 = st.columns(2)
            with col1:
                backtesting_window = st.slider("Backtesting Window", 100, 500, 252, key="backtesting_window_slider")
            with col2:
                backtesting_confidence = st.slider("Backtesting Confidence", 0.90, 0.99, 0.95, 0.01, key="backtesting_confidence_slider")

            # Time range for backtesting
            st.subheader("📅 Backtesting Time Range")
            col1, col2 = st.columns(2)
            with col1:
                backtest_start = st.date_input("Backtest Start Date", date_start, key="backtest_start")
            with col2:
                backtest_end = st.date_input("Backtest End Date", date_end, key="backtest_end")

            # Filter data for backtesting
            backtest_mask = (st.session_state.current_data.index >= pd.to_datetime(backtest_start)) & \
                           (st.session_state.current_data.index <= pd.to_datetime(backtest_end))
            backtest_filtered_data = st.session_state.current_data.loc[backtest_mask]

            if portfolio_type == "Options Portfolio":
                if 'Option_Price' in backtest_filtered_data.columns:
                    backtest_portfolio_returns = backtest_filtered_data['Option_Price'].pct_change().dropna()
                else:
                    st.error("Option price data not found")
                    backtest_portfolio_returns = pd.Series()
            else:
                backtest_filtered_returns = backtest_filtered_data.pct_change().dropna()
                if len(backtest_filtered_returns.shape) > 1:
                    backtest_portfolio_returns = backtest_filtered_returns.dot(st.session_state.weights)
                else:
                    backtest_portfolio_returns = backtest_filtered_returns

            if st.button("🔄 Run Backtesting", key="run_backtesting_button"):
                if len(backtest_portfolio_returns) >= backtesting_window + 50:
                    with st.spinner("Running backtesting..."):
                        # Create a VaR function for backtesting
                        def var_function(returns, conf_level, horizon):
                            if portfolio_type == "Options Portfolio":
                                return calculate_options_var_comprehensive(returns, conf_level, 'historical')
                            else:
                                if var_model == "Parametric (Delta-Normal)":
                                    return instances['var_engines'].calculate_parametric_var(returns, conf_level, horizon)
                                elif var_model == "Historical Simulation":
                                    return instances['var_engines'].calculate_historical_var(returns, conf_level, horizon)
                                elif var_model == "Monte Carlo":
                                    return instances['var_engines'].calculate_monte_carlo_var(returns, conf_level, horizon, 5000)
                                else:
                                    return instances['var_engines'].calculate_parametric_var(returns, conf_level, horizon)

                        backtest_results = instances['backtesting'].perform_backtesting(
                            backtest_portfolio_returns, backtesting_confidence, backtesting_window, var_function
                        )

                        if backtest_results:
                            # Display results
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("Backtesting Results")
                                backtest_metrics = {
                                    'Total Violations': backtest_results.get('violations', 0),
                                    'Expected Violations': f"{backtest_results.get('expected_violations', 0):.1f}",
                                    'Violation Rate': f"{backtest_results.get('violation_rate', 0)*100:.2f}%",
                                    'Kupiec Test p-value': f"{backtest_results.get('kupiec_pvalue', 0):.4f}",
                                    'Model Performance': 'Adequate' if backtest_results.get('kupiec_pvalue', 0) > 0.05 else 'Needs Review'
                                }
                                backtest_df = create_metrics_dataframe(backtest_metrics, "Backtesting Results")
                                safe_dataframe_display(backtest_df, "Backtesting Results", "backtesting_results")

                            with col2:
                                st.subheader("Basel Traffic Light")
                                traffic_light = instances['backtesting'].basel_traffic_light(
                                    backtest_results.get('violations', 0),
                                    backtest_results.get('expected_violations', 0)
                                )

                                if traffic_light == "Green":
                                    st.success(f"🟢 {traffic_light} Zone - Model performing well")
                                elif traffic_light == "Yellow":
                                    st.warning(f"🟡 {traffic_light} Zone - Model needs attention")
                                else:
                                    st.error(f"🔴 {traffic_light} Zone - Model needs review")

                            # Violations plot
                            if 'var_estimates' in backtest_results and 'violations_dates' in backtest_results:
                                st.subheader("VaR Violations Over Time")
                                fig_violations = instances['visualization'].plot_var_violations(
                                    backtest_portfolio_returns,
                                    backtest_results['var_estimates'],
                                    backtest_results['violations_dates']
                                )
                                st.plotly_chart(fig_violations, use_container_width=True, key="backtesting_violations_chart")
                else:
                    st.error(f"Insufficient data for backtesting. Need at least {backtesting_window + 50} data points.")

        with tab6:  # Stress Testing
            st.header("⚡ Stress Testing")

            # Time range for stress testing
            st.subheader("📅 Stress Testing Time Range")
            col1, col2 = st.columns(2)
            with col1:
                stress_start = st.date_input("Stress Test Start Date", date_start, key="stress_start")
            with col2:
                stress_end = st.date_input("Stress Test End Date", date_end, key="stress_end")

            # Filter data for stress testing
            stress_mask = (st.session_state.current_data.index >= pd.to_datetime(stress_start)) & \
                         (st.session_state.current_data.index <= pd.to_datetime(stress_end))
            stress_filtered_data = st.session_state.current_data.loc[stress_mask]

            if portfolio_type == "Options Portfolio":
                if 'Option_Price' in stress_filtered_data.columns:
                    stress_portfolio_returns = stress_filtered_data['Option_Price'].pct_change().dropna()
                else:
                    st.error("Option price data not found")
                    stress_portfolio_returns = pd.Series()
            else:
                stress_filtered_returns = stress_filtered_data.pct_change().dropna()
                if len(stress_filtered_returns.shape) > 1:
                    stress_portfolio_returns = stress_filtered_returns.dot(st.session_state.weights)
                else:
                    stress_portfolio_returns = stress_filtered_returns

            # Stress testing options
            stress_type = st.selectbox(
                "Select Stress Test Type",
                ["Historical Scenarios", "Custom Stress Test"],
                key="stress_type_select"
            )

            if stress_type == "Historical Scenarios":
                if portfolio_type == "Options Portfolio":
                    scenario = st.selectbox(
                        "Select Historical Scenario",
                        ["2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash", "High Volatility Spike"],
                        key="historical_scenario_select"
                    )
                else:
                    scenario = st.selectbox(
                        "Select Historical Scenario",
                        ["2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash"],
                        key="historical_scenario_select"
                    )

                if st.button("🔄 Run Historical Stress Test", key="run_historical_stress_button"):
                    with st.spinner("Running stress test..."):
                        stress_results = instances['stress_testing'].run_stress_test(
                            stress_portfolio_returns, scenario, confidence_level
                        )

                        if stress_results:
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("Stress Test Results")
                                stress_metrics = {
                                    'Baseline VaR': f"${stress_results.get('baseline_var', 0):,.2f}",
                                    'Stressed VaR': f"${stress_results.get('stressed_var', 0):,.2f}",
                                    'VaR Increase': f"{stress_results.get('var_increase', 0):.2f}%",
                                    'Worst Case Loss': f"${stress_results.get('worst_case', 0):,.2f}",
                                    'Scenario': stress_results.get('scenario_description', 'N/A')
                                }
                                stress_df = create_metrics_dataframe(stress_metrics, "Stress Test Results")
                                safe_dataframe_display(stress_df, "Stress Test Results", "stress_results")

                            with col2:
                                st.subheader("Scenario Impact")
                                impact_data = {
                                    'Baseline': stress_results.get('baseline_var', 0),
                                    'Stressed': stress_results.get('stressed_var', 0)
                                }

                                fig_stress = go.Figure(data=[
                                    go.Bar(name='VaR Comparison', x=list(impact_data.keys()), y=list(impact_data.values()))
                                ])
                                fig_stress.update_layout(
                                    title="VaR: Baseline vs Stressed",
                                    yaxis_title="VaR ($)",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig_stress, use_container_width=True, key="stress_comparison_chart")

            else:  # Custom Stress Test
                st.subheader("Custom Stress Parameters")
                col1, col2, col3 = st.columns(3)

                with col1:
                    vol_shock = st.slider("Volatility Shock (%)", 0, 500, 100, key="vol_shock_slider")
                with col2:
                    corr_shock = st.slider("Correlation Shock", 0.0, 1.0, 0.3, 0.1, key="corr_shock_slider")
                with col3:
                    market_shock = st.slider("Market Shock (%)", -50, 50, -20, key="market_shock_slider")

                if st.button("🔄 Run Custom Stress Test", key="run_custom_stress_button"):
                    with st.spinner("Running custom stress test..."):
                        custom_stress_results = instances['stress_testing'].run_custom_stress_test(
                            stress_portfolio_returns, vol_shock, corr_shock, market_shock, confidence_level
                        )

                        if custom_stress_results:
                            custom_stress_metrics = {
                                'Baseline VaR': f"${custom_stress_results.get('baseline_var', 0):,.2f}",
                                'Stressed VaR': f"${custom_stress_results.get('stressed_var', 0):,.2f}",
                                'VaR Increase': f"{custom_stress_results.get('var_increase', 0):.2f}%",
                                'Worst Case Loss': f"${custom_stress_results.get('worst_case', 0):,.2f}"
                            }
                            custom_stress_df = create_metrics_dataframe(custom_stress_metrics, "Custom Stress Results")
                            safe_dataframe_display(custom_stress_df, "Custom Stress Results", "custom_stress_results")

        with tab7:  # Export Data
            st.header("⬇️ Export Data")
            
            st.markdown("""
            Export your analysis data and results for further processing or reporting.
            """)

            # Export options
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Price Data Export")
                if st.session_state.current_data is not None:
                    # Show data preview
                    st.write("**Data Preview:**")
                    st.dataframe(st.session_state.current_data.head(), key="export_price_preview")
                    
                    # Export button for price data
                    csv_data = st.session_state.current_data.to_csv()
                    st.download_button(
                        label="📥 Download Price Data (CSV)",
                        data=csv_data,
                        file_name=f"price_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_price_data"
                    )
                    
                    # Data info
                    st.info(f"**Data Points:** {len(st.session_state.current_data)}")
                    st.info(f"**Date Range:** {st.session_state.current_data.index[0].strftime('%Y-%m-%d')} to {st.session_state.current_data.index[-1].strftime('%Y-%m-%d')}")
                else:
                    st.warning("No price data available for export. Please load data first.")

            with col2:
                st.subheader("📈 Returns Data Export")
                if st.session_state.current_returns is not None:
                    # Show returns preview
                    st.write("**Returns Preview:**")
                    if len(st.session_state.current_returns.shape) == 1:
                        returns_df = pd.DataFrame({'Returns': st.session_state.current_returns})
                    else:
                        returns_df = pd.DataFrame(st.session_state.current_returns)
                    
                    st.dataframe(returns_df.head(), key="export_returns_preview")
                    
                    # Export button for returns data
                    csv_returns = returns_df.to_csv()
                    st.download_button(
                        label="📥 Download Returns Data (CSV)",
                        data=csv_returns,
                        file_name=f"returns_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_returns_data"
                    )
                    
                    # Returns info
                    st.info(f"**Data Points:** {len(st.session_state.current_returns)}")
                    if hasattr(st.session_state.current_returns, 'std'):
                        volatility = st.session_state.current_returns.std() * np.sqrt(252) * 100
                        st.info(f"**Annualized Volatility:** {float(volatility):.2f}%")
                else:
                    st.warning("No returns data available for export. Please load data first.")

            # VaR Results Export
            st.subheader("🎯 VaR Results Export")
            if st.session_state.var_results:
                # Create VaR results summary
                var_summary = {
                    'Export_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Portfolio_Type': portfolio_type,
                    'VaR_Model': st.session_state.var_results.get('model', 'N/A'),
                    'Value_at_Risk': st.session_state.var_results.get('var', 0),
                    'Expected_Shortfall': st.session_state.var_results.get('expected_shortfall', 0),
                    'Confidence_Level': st.session_state.var_results.get('confidence_level', 0),
                    'Time_Horizon': time_horizon,
                    'Assets': ', '.join(st.session_state.symbols) if st.session_state.symbols else 'N/A',
                    'Weights': ', '.join([f"{w:.4f}" for w in st.session_state.weights]) if st.session_state.weights else 'N/A'
                }
                
                # Add options-specific parameters if available
                if portfolio_type == "Options Portfolio" and hasattr(st.session_state, 'option_params'):
                    params = st.session_state.option_params
                    var_summary.update({
                        'Underlying_Symbol': params.get('underlying', 'N/A'),
                        'Option_Type': params.get('option_type', 'N/A'),
                        'Strike_Price': params.get('strike_price', 0),
                        'Spot_Price': params.get('spot_price', 0),
                        'Time_to_Expiry': params.get('time_to_expiry', 0),
                        'Risk_Free_Rate': params.get('risk_free_rate', 0),
                        'Volatility': params.get('volatility', 0),
                        'Quantity': params.get('quantity', 0)
                    })
                
                var_results_df = pd.DataFrame([var_summary])
                
                # Show VaR results preview
                st.write("**VaR Results Preview:**")
                st.dataframe(var_results_df, key="export_var_preview")
                
                # Export button for VaR results
                csv_var_results = var_results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download VaR Results (CSV)",
                    data=csv_var_results,
                    file_name=f"var_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_var_results"
                )
            else:
                st.warning("No VaR results available for export. Please run VaR calculations first.")

            # Portfolio Composition Export
            if hasattr(st.session_state, 'symbols') and hasattr(st.session_state, 'weights'):
                st.subheader("💼 Portfolio Composition Export")
                portfolio_composition = pd.DataFrame({
                    'Asset': st.session_state.symbols,
                    'Weight': st.session_state.weights,
                    'Weight_Percentage': [w*100 for w in st.session_state.weights]
                })
                
                st.write("**Portfolio Composition Preview:**")
                st.dataframe(portfolio_composition, key="export_portfolio_preview")
                
                csv_portfolio = portfolio_composition.to_csv(index=False)
                st.download_button(
                    label="📥 Download Portfolio Composition (CSV)",
                    data=csv_portfolio,
                    file_name=f"portfolio_composition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_portfolio_composition"
                )

            # Export All Data
            st.subheader("📦 Export All Data")
            st.markdown("Download a comprehensive export containing all available data and results.")
            
            if st.button("📥 Prepare Complete Export", key="prepare_complete_export"):
                try:
                    # Create a comprehensive export
                    export_data = {}
                    
                    # Add price data
                    if st.session_state.current_data is not None:
                        export_data['price_data'] = st.session_state.current_data.to_dict()
                    
                    # Add returns data
                    if st.session_state.current_returns is not None:
                        if len(st.session_state.current_returns.shape) == 1:
                            export_data['returns_data'] = st.session_state.current_returns.to_dict()
                        else:
                            export_data['returns_data'] = st.session_state.current_returns.to_dict()
                    
                    # Add VaR results
                    if st.session_state.var_results:
                        export_data['var_results'] = st.session_state.var_results
                    
                    # Add portfolio info
                    export_data['portfolio_info'] = {
                        'portfolio_type': portfolio_type,
                        'symbols': st.session_state.symbols,
                        'weights': st.session_state.weights,
                        'export_timestamp': datetime.now().isoformat()
                    }
                    
                    # Add options parameters if available
                    if hasattr(st.session_state, 'option_params') and st.session_state.option_params:
                        export_data['option_parameters'] = st.session_state.option_params
                    
                    # Convert to JSON for download
                    import json
                    json_export = json.dumps(export_data, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download Complete Export (JSON)",
                        data=json_export,
                        file_name=f"complete_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_complete_export"
                    )
                    
                    st.success("✅ Complete export prepared successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error preparing complete export: {str(e)}")

        with tab8:  # Help
            st.header("❓ Help & Documentation")

            st.markdown(f"""
            ## 🚀 Welcome to the VaR & Risk Analytics Platform
            
            This comprehensive platform provides sophisticated financial risk modeling capabilities for portfolio management and risk assessment.
            
            ### 📊 Getting Started
            
            1. **Select Portfolio Type**: Choose from Single Asset, Multi-Asset, Crypto, or Options Portfolio
            2. **Configure Data Source**: For non-options portfolios, select data source
            3. **Choose VaR Model**: Select from sophisticated VaR calculation methods
            4. **Set Parameters**: Configure confidence levels, time horizons, and other settings
            5. **Load Data**: Click the "Load Data" button to begin analysis
            
            ### 💼 Portfolio Types
            
            #### Single Asset
            - **Default**: AAPL
            - **Format**: Standard ticker symbols (AAPL, GOOGL, MSFT, TSLA)
            
            #### Multi-Asset
            - **Default**: AAPL,GOOGL,MSFT,TSLA
            - **Format**: Comma-separated ticker symbols
            - **Crypto Support**: Add crypto symbols in the crypto field (BTC-USD, ETH-USD)
            
            #### Crypto Portfolio
            - **Default**: BTC-USD
            - **Format**: Crypto symbols with -USD suffix (BTC-USD, ETH-USD, ADA-USD)
            
            #### Options Portfolio
            - **Live Market**: Fetches real options data using yfinance
            - **Manual Entry**: Configure custom option parameters
            - **Default**: AAPL call option (Strike: $155, Expiry: 3 months)
            
            ### 📈 VaR Models
            
            {"#### Standard Models" if portfolio_type != "Options Portfolio" else "#### Options VaR Models"}
            {"1. **Parametric (Delta-Normal)**: Classical normal distribution approach" if portfolio_type != "Options Portfolio" else "1. **Historical Simulation**: Non-parametric historical method"}
            {"2. **Historical Simulation**: Non-parametric historical method" if portfolio_type != "Options Portfolio" else "2. **Parametric (Delta-Normal)**: Classical normal distribution approach"}
            {"3. **Monte Carlo**: Simulation-based approach (1K-100K simulations)" if portfolio_type != "Options Portfolio" else "3. **Monte Carlo**: Simulation-based approach for options"}
            {"4. **GARCH**: Advanced volatility modeling for time-varying risk" if portfolio_type != "Options Portfolio" else "4. **Historic Simulation**: Enhanced historical method for options"}
            {"5. **Extreme Value Theory (EVT)**: Tail risk modeling for extreme events" if portfolio_type != "Options Portfolio" else ""}
            
            ### ⬇️ Export Data
            
            The Export Data tab allows you to download:
            - **Price Data**: Historical price data in CSV format
            - **Returns Data**: Calculated returns data in CSV format
            - **VaR Results**: Risk metrics and model results in CSV format
            - **Portfolio Composition**: Asset weights and allocation in CSV format
            - **Complete Export**: All data and results in JSON format
            
            ### 🚨 Troubleshooting
            
            #### Common Issues
            - **"Insufficient data"**: Increase historical window or data period
            - **"GARCH model failed"**: Requires minimum 100 observations
            - **"Symbol not found"**: Verify ticker format (add -USD for crypto)
            - **"Weights don't sum to 1"**: Portfolio weights are automatically normalized
            - **"Option price data not found"**: Reload data after changing portfolio type
            
            #### Options-Specific Issues
            - **"No options data found"**: Symbol may not have listed options
            - **"Options chain empty"**: Try a different expiry date
            - **"Strike not available"**: System will find closest available strike
            - **"Time to expiry too short"**: Minimum 1 day required
            """)

    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>🚀 Welcome to the VaR & Risk Analytics Platform</h2>
            <p style="font-size: 1.2rem; color: #888;">
                Professional risk management tools for modern financial markets
            </p>
            <p style="margin-top: 2rem;">
                👈 Configure your portfolio and data source in the sidebar, then click "Load Data" to begin analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature highlights
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Advanced VaR Models</h3>
                <p>5 sophisticated VaR calculation methods including Parametric, Historical, Monte Carlo, GARCH, and EVT</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔄 Real-time Analysis</h3>
                <p>Dynamic updates across all tabs when parameters change, with intelligent data persistence</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🧪 Comprehensive Testing</h3>
                <p>Backtesting, stress testing, and scenario analysis with regulatory compliance metrics</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
