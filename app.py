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
import json
import io
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
    .feature-highlight {
        background-color: #ff6b6b20;
        border: 1px solid #ff6b6b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

def calculate_required_data_days(backtesting_window, rolling_window=60, buffer_days=100):
    """Calculate required data days for comprehensive analysis"""
    return max(backtesting_window + rolling_window + buffer_days, 500)

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
    if 'option_data_fetcher' not in st.session_state:
        st.session_state.option_data_fetcher = None
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
        st.session_state.option_data_fetcher = None
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
        'option_data_fetcher': OptionsDataFetcher()
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

        # Data Source Selection for Options Portfolio (moved here)
        if portfolio_type == "Options Portfolio":
            st.subheader("📊 Data Source")
            option_data_fetcher_source = st.selectbox(
                "Options Data Source",
                ["Live Market Data", "Manual Entry"],
                key="option_data_fetcher_source_select"
            )
            data_source = "Options Data"
        else:
            # Data Source Selection for other portfolio types
            st.subheader("📊 Data Source")
            data_source = st.selectbox(
                "Select Data Source",
                ["Live Market Data", "Upload File", "Manual Entry", "Synthetic Data"],
                key="data_source_select"
            )

        # Check for configuration changes and reset if needed
        reset_data_on_config_change(portfolio_type, data_source)

        # VaR Model Selection
        st.subheader("🎯 VaR Model")
        if portfolio_type == "Options Portfolio":
            var_model = st.selectbox(
                "Select Options VaR Model",
                [
                    "Delta-Gamma Parametric",
                    "Historical Simulation",
                    "Parametric (Delta-Normal)", 
                    "Monte Carlo",
                    "Historic Simulation"
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

        # Date Range with automatic extension for backtesting
        st.subheader("📅 Date Range")
        end_date = datetime.now()
        
        # Calculate required data period for backtesting
        backtesting_window = st.slider("Backtesting Window", 100, 500, 252, key="backtesting_window_slider")
        required_days = calculate_required_data_days(backtesting_window)
        
        # Auto-extend start date for sufficient backtesting data
        auto_start_date = end_date - timedelta(days=required_days)
        default_start_date = end_date - timedelta(days=365)  # Default 1 year for display
        
        st.info(f"💡 Auto-extending data to {required_days} days for comprehensive backtesting")
        
        date_start = st.date_input("Start Date (Display)", default_start_date, key="start_date_input")
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

            # Initialize default values for options parameters
            underlying = "AAPL"
            spot_price = 150.0
            strike_price = 155.0
            time_to_expiry = 0.25
            risk_free_rate = 0.05
            volatility = 0.25
            option_type_str = "Call"
            quantity = 100

            if option_data_fetcher_source == "Live Market Data":
                st.write("**Live Market Options:**")
                underlying = st.text_input("Underlying Symbol", "AAPL", key="options_underlying_input")

                # Fetch options data button
                if st.button("🔄 Fetch Options Data", key="fetch_options_button"):
                    with st.spinner("Fetching options data..."):
                        try:
                            option_data_fetcher = instances['option_data_fetcher'].get_options_data(underlying, use_synthetic=False)
                            if option_data_fetcher and option_data_fetcher.get('options_chains'):
                                st.session_state.option_data_fetcher = option_data_fetcher
                                st.session_state.options_expirations = option_data_fetcher.get('expiry_dates', [])
                                spot_price = option_data_fetcher.get('current_price', 150.0)
                                st.success(f"✅ Fetched options data for {underlying}")
                            else:
                                st.warning(f"Could not fetch live options data for {underlying}. Using synthetic data.")
                                option_data_fetcher = instances['option_data_fetcher'].generate_synthetic_option_data_fetcher(underlying, spot_price)
                                st.session_state.option_data_fetcher = option_data
                                st.session_state.options_expirations = option_data_fetcher.get('expiry_dates', [])
                        except Exception as e:
                            st.error(f"Error fetching options data: {str(e)}")
                            st.info("Generating synthetic options data for demonstration.")
                            option_data_fetcher = instances['option_data_fetcher'].generate_synthetic_option_data(underlying, spot_price)
                            st.session_state.option_data_fetcher = option_data_fetcher
                            st.session_state.options_expirations = option_data_fetcher.get('expiry_dates', [])

                # Options selection if data is available
                if hasattr(st.session_state, 'option_data_fetcher') and st.session_state.option_data_fetcher is not None:
                    option_type_str = st.selectbox("Option Type", ["Call", "Put"], key="live_option_type_select")

                    # Get available expiry dates
                    available_expiries = list(st.session_state.option_data_fetcher.get('options_chains', {}).keys())
                    if available_expiries:
                        selected_expiry = st.selectbox(
                            "Expiry Date",
                            available_expiries[:5],  # Show first 5 expiries
                            key="live_expiry_select"
                        )

                        # Get the appropriate options dataframe
                        chain_data = st.session_state.option_data_fetcher['options_chains'].get(selected_expiry, {})
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
                            spot_price = st.session_state.option_data_fetcher.get('current_price', 150.0)
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

            if option_data_fetcher_source == "Manual Entry":
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
                #num_days = st.slider("Number of Days", 100, 2000, max(500, required_days), key="synthetic_days_slider")
                initial_price = st.number_input("Initial Price ($)", 10.0, 1000.0, 100.0, key="synthetic_price_input")
                annual_return = st.slider("Annual Return", -0.5, 0.5, 0.08, 0.01, key="synthetic_return_slider")
                annual_volatility = st.slider("Annual Volatility", 0.05, 1.0, 0.20, 0.01, key="synthetic_vol_slider")
                random_seed = st.number_input("Random Seed", 1, 1000, 42, key="synthetic_seed_input")
            else:
                # Default parameters for good representative data
                #num_days = max(500, required_days)
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
                        data = instances['option_data_fetcher'].generate_synthetic_option_data(
                            spot_price, strike_price, time_to_expiry, 
                            risk_free_rate, volatility, option_type_str.lower(), underlying,
                        )
                        data = pd.DataFrame(data['options'])
                        if data is not None :
                            st.session_state.current_data = data
                            st.session_state.current_returns = (pd.DataFrame(data['options'])).pct_change().dropna()
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
                        # Use auto-extended start date for sufficient backtesting data
                        extended_start = auto_start_date

                        data = instances['data_ingestion'].load_live_data(symbols, extended_start, date_end)

                        if data is not None:
                            st.session_state.current_data = data
                            st.session_state.current_returns = instances['data_ingestion'].returns
                            st.session_state.data_loaded = True
                            st.session_state.symbols = symbols
                            st.session_state.weights = weights
                            st.success(f"✅ Successfully loaded data for {len(symbols)} asset(s) ({len(data)} days)")
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
                # Options-specific VaR calculation using the consolidated method
                params = st.session_state.option_params
                var_result_dict = instances['options_var'].calculate_portfolio_var(
                    symbol=params['underlying'],
                    spot_price=params['spot_price'],
                    strike_price=params['strike_price'],
                    time_to_expiry=params['time_to_expiry'],
                    risk_free_rate=params['risk_free_rate'],
                    volatility=params['volatility'],
                    option_type=params['option_type'],
                    quantity=params['quantity'],
                    confidence_level=confidence_level,
                    var_model=var_model
                )
                
                if var_result_dict['success']:
                    var_result = var_result_dict['var_dollar']
                    expected_shortfall = var_result_dict['expected_shortfall']
                else:
                    var_result = 0
                    expected_shortfall = 0
                    st.error(f"Error calculating options VaR: {var_result_dict['error']}")
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

        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Dashboard", "📈 VaR Calculator", "📋 Data Overview", "🔄 Rolling Analysis", 
            "🧪 Backtesting", "⚡ Stress Testing", "⬇️ Export Data", "❓ Help"
        ])

        with tab1:  # Dashboard
            if portfolio_type == "Options Portfolio":
                st.header("📊 Options Risk Analytics Dashboard")

                # Options-specific metrics
                if hasattr(st.session_state, 'option_params') and st.session_state.option_params:
                    params = st.session_state.option_params

                    # Calculate current option price and Greeks using consolidated methods
                    current_option_price = instances['options_var'].black_scholes_price(
                        params['spot_price'], params['strike_price'], 
                        params['time_to_expiry'], params['risk_free_rate'], 
                        params['volatility'], params['option_type']
                    )

                    greeks = instances['options_var'].black_scholes_greeks(
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
                        if hasattr(current_vol, 'iloc'):
                            vol_value = current_vol.iloc[0] if len(current_vol) > 0 else current_vol
                        else:
                            vol_value = current_vol
                        st.metric(
                            label="Annualized Volatility",
                            value=f"{vol_value:.2f}%",
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
                        params = st.session_state.option_params
                        
                        # Use consolidated options VaR methods
                        comparison_results['Historical'] = instances['options_var'].calculate_options_var_comprehensive(
                            var_filtered_returns, confidence_level, 'historical'
                        )
                        comparison_results['Parametric'] = instances['options_var'].calculate_options_var_comprehensive(
                            var_filtered_returns, confidence_level, 'parametric'
                        )
                        comparison_results['Monte Carlo'] = instances['options_var'].calculate_options_var_comprehensive(
                            var_filtered_returns, confidence_level, 'monte_carlo'
                        )
                        
                        # Add Delta-Gamma if option params available
                        if params:
                            underlying_returns = var_filtered_data[params['underlying']].pct_change().dropna()
                            comparison_results['Delta-Gamma'] = instances['options_var'].calculate_delta_gamma_var(
                                params['spot_price'], params['strike_price'], 
                                params['time_to_expiry'], params['risk_free_rate'], 
                                params['volatility'], params['option_type'], 
                                confidence_level, underlying_returns
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
                # Use the backtesting window from sidebar
                st.info(f"Using backtesting window: {backtesting_window} days")
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

            # Check data sufficiency
            total_data_points = len(st.session_state.current_data)
            required_points = backtesting_window + 50
            
            st.info(f"📊 Data Status: {total_data_points} total points, {required_points} required for backtesting")

            if st.button("🔄 Run Backtesting", key="run_backtesting_button"):
                if len(backtest_portfolio_returns) >= backtesting_window + 50:
                    with st.spinner("Running backtesting..."):
                        # Create a VaR function for backtesting
                        def var_function(returns, conf_level, horizon):
                            if portfolio_type == "Options Portfolio":
                                return instances['options_var'].calculate_options_var_comprehensive(
                                    returns, conf_level, 'historical'
                                )
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
                    st.error(f"Insufficient data for backtesting. Need at least {backtesting_window + 50} data points. Current: {len(backtest_portfolio_returns)}")
                    st.info("💡 Try loading data with a longer time period or reduce the backtesting window.")

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
                st.subheader("🎛️ Custom Stress Parameters")
                
                # Enhanced custom stress test interface
                st.markdown('<div class="feature-highlight">', unsafe_allow_html=True)
                st.write("**Design your own stress scenario by adjusting the parameters below:**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)

                with col1:
                    vol_multiplier = st.slider("Volatility Multiplier", 0.5, 5.0, 1.0, 0.1, key="vol_multiplier_slider")
                    st.caption(f"Current: {vol_multiplier:.1f}x normal volatility")
                    
                with col2:
                    correlation_shock = st.slider("Correlation Shock", 0.0, 1.0, 0.3, 0.1, key="corr_shock_slider")
                    st.caption(f"Correlation increase: +{correlation_shock:.1f}")
                    
                with col3:
                    market_shock = st.slider("Market Shock (%)", -50, 50, -20, 1, key="market_shock_slider")
                    st.caption(f"Market shift: {market_shock:+.0f}%")

                # Additional stress parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    liquidity_shock = st.slider("Liquidity Shock", 0.0, 2.0, 0.0, 0.1, key="liquidity_shock_slider")
                    st.caption("Bid-ask spread multiplier")
                    
                with col2:
                    tail_risk_multiplier = st.slider("Tail Risk Multiplier", 1.0, 3.0, 1.0, 0.1, key="tail_risk_slider")
                    st.caption("Extreme event probability")

                # Scenario description
                st.subheader("📝 Scenario Description")
                scenario_description = st.text_area(
                    "Describe your stress scenario:",
                    f"Custom stress test with {vol_multiplier:.1f}x volatility, {market_shock:+.0f}% market shock, and {correlation_shock:.1f} correlation increase.",
                    key="scenario_description_input"
                )

                if st.button("🔄 Run Custom Stress Test", key="run_custom_stress_button"):
                    with st.spinner("Running custom stress test..."):
                        custom_stress_results = instances['stress_testing'].run_custom_stress_test(
                            stress_portfolio_returns, vol_multiplier, correlation_shock, market_shock, confidence_level
                        )

                        if custom_stress_results:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Custom Stress Results")
                                custom_stress_metrics = {
                                    'Baseline VaR': f"${custom_stress_results.get('baseline_var', 0):,.2f}",
                                    'Stressed VaR': f"${custom_stress_results.get('stressed_var', 0):,.2f}",
                                    'VaR Increase': f"{custom_stress_results.get('var_increase', 0):.2f}%",
                                    'Worst Case Loss': f"${custom_stress_results.get('worst_case', 0):,.2f}",
                                    'Scenario': custom_stress_results.get('scenario_description', 'Custom')
                                }
                                custom_stress_df = create_metrics_dataframe(custom_stress_metrics, "Custom Stress Results")
                                safe_dataframe_display(custom_stress_df, "Custom Stress Results", "custom_stress_results")

                            with col2:
                                st.subheader("Stress Impact Visualization")
                                
                                # Create comparison chart
                                baseline_var = custom_stress_results.get('baseline_var', 0)
                                stressed_var = custom_stress_results.get('stressed_var', 0)
                                
                                fig_custom_stress = go.Figure()
                                
                                fig_custom_stress.add_trace(go.Bar(
                                    name='Baseline VaR',
                                    x=['VaR'],
                                    y=[baseline_var],
                                    marker_color='#00ff88'
                                ))
                                
                                fig_custom_stress.add_trace(go.Bar(
                                    name='Stressed VaR',
                                    x=['VaR'],
                                    y=[stressed_var],
                                    marker_color='#ff6b6b'
                                ))
                                
                                fig_custom_stress.update_layout(
                                    title="Custom Stress Test Impact",
                                    yaxis_title="VaR ($)",
                                    template="plotly_dark",
                                    barmode='group'
                                )
                                
                                st.plotly_chart(fig_custom_stress, use_container_width=True, key="custom_stress_chart")

                            # Returns distribution comparison
                            if 'stressed_returns' in custom_stress_results:
                                st.subheader("Returns Distribution: Normal vs Stressed")
                                
                                fig_dist_comparison = go.Figure()
                                
                                # Original returns
                                fig_dist_comparison.add_trace(go.Histogram(
                                    x=stress_portfolio_returns,
                                    name='Normal Returns',
                                    opacity=0.7,
                                    nbinsx=50,
                                    marker_color='#00ff88'
                                ))
                                
                                # Stressed returns
                                fig_dist_comparison.add_trace(go.Histogram(
                                    x=custom_stress_results['stressed_returns'],
                                    name='Stressed Returns',
                                    opacity=0.7,
                                    nbinsx=50,
                                    marker_color='#ff6b6b'
                                ))
                                
                                fig_dist_comparison.update_layout(
                                    title="Returns Distribution Comparison",
                                    xaxis_title="Returns",
                                    yaxis_title="Frequency",
                                    template="plotly_dark",
                                    barmode='overlay'
                                )
                                
                                st.plotly_chart(fig_dist_comparison, use_container_width=True, key="stress_distribution_comparison")

        with tab7:  # Export Data
            st.header("⬇️ Export Data")
            
            st.markdown("""
            Export your analysis data and results for further analysis, reporting, or compliance purposes.
            All exports include timestamps and comprehensive metadata.
            """)

            # Export options
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Price Data Export")
                if st.session_state.current_data is not None:
                    # Show data preview
                    st.write("**Data Preview:**")
                    safe_dataframe_display(st.session_state.current_data.head(), "Price Data Preview", "export_price_preview")
                    
                    # Export button
                    if st.button("📥 Download Price Data (CSV)", key="export_price_data"):
                        csv_buffer = io.StringIO()
                        st.session_state.current_data.to_csv(csv_buffer)
                        csv_data = csv_buffer.getvalue()
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"price_data_{timestamp}.csv"
                        
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            key="download_price_csv"
                        )
                else:
                    st.warning("No price data available for export")

            with col2:
                st.subheader("📈 Returns Data Export")
                if st.session_state.current_returns is not None:
                    # Show returns preview
                    st.write("**Returns Preview:**")
                    returns_df = pd.DataFrame(st.session_state.current_returns)
                    safe_dataframe_display(returns_df.head(), "Returns Data Preview", "export_returns_preview")
                    
                    # Export button
                    if st.button("📥 Download Returns Data (CSV)", key="export_returns_data"):
                        csv_buffer = io.StringIO()
                        returns_df.to_csv(csv_buffer)
                        csv_data = csv_buffer.getvalue()
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"returns_data_{timestamp}.csv"
                        
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            key="download_returns_csv"
                        )
                else:
                    st.warning("No returns data available for export")

            # VaR Results Export
            st.subheader("🎯 VaR Results Export")
            if st.session_state.var_results:
                var_results_df = create_metrics_dataframe(st.session_state.var_results, "VaR Results")
                safe_dataframe_display(var_results_df, "VaR Results Preview", "export_var_preview")
                
                if st.button("📥 Download VaR Results (CSV)", key="export_var_results"):
                    csv_buffer = io.StringIO()
                    var_results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"var_results_{timestamp}.csv"
                    
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        key="download_var_csv"
                    )
            else:
                st.warning("No VaR results available for export")

            # Portfolio Composition Export
            st.subheader("💼 Portfolio Composition Export")
            if hasattr(st.session_state, 'symbols') and hasattr(st.session_state, 'weights'):
                portfolio_df = pd.DataFrame({
                    'Asset': st.session_state.symbols,
                    'Weight': st.session_state.weights,
                    'Weight_Percentage': [w*100 for w in st.session_state.weights]
                })
                
                safe_dataframe_display(portfolio_df, "Portfolio Composition Preview", "export_portfolio_preview")
                
                if st.button("📥 Download Portfolio Composition (CSV)", key="export_portfolio_composition"):
                    csv_buffer = io.StringIO()
                    portfolio_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"portfolio_composition_{timestamp}.csv"
                    
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        key="download_portfolio_csv"
                    )

            # Complete Export (JSON)
            st.subheader("📦 Complete Export")
            st.write("Export all data and results in a comprehensive JSON format")
            
            if st.button("📥 Generate Complete Export", key="generate_complete_export"):
                # Compile all available data
                export_data = {
                    'metadata': {
                        'export_timestamp': datetime.now().isoformat(),
                        'portfolio_type': portfolio_type,
                        'data_source': data_source,
                        'var_model': var_model,
                        'confidence_level': confidence_level,
                        'time_horizon': time_horizon
                    },
                    'price_data': st.session_state.current_data.to_dict() if st.session_state.current_data is not None else None,
                    'returns_data': st.session_state.current_returns.to_dict() if st.session_state.current_returns is not None else None,
                    'var_results': st.session_state.var_results,
                    'portfolio_composition': {
                        'symbols': st.session_state.symbols,
                        'weights': st.session_state.weights
                    }
                }
                
                # Add options-specific data if available
                if portfolio_type == "Options Portfolio" and hasattr(st.session_state, 'option_params'):
                    export_data['options_parameters'] = st.session_state.option_params
                
                # Convert to JSON
                json_data = json.dumps(export_data, indent=2, default=str)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"complete_export_{timestamp}.json"
                
                st.download_button(
                    label="💾 Download Complete Export (JSON)",
                    data=json_data,
                    file_name=filename,
                    mime="application/json",
                    key="download_complete_json"
                )

            # Export Statistics
            st.subheader("📊 Export Statistics")
            if st.session_state.current_data is not None:
                export_stats = {
                    'Total Data Points': len(st.session_state.current_data),
                    'Date Range': f"{st.session_state.current_data.index[0].strftime('%Y-%m-%d')} to {st.session_state.current_data.index[-1].strftime('%Y-%m-%d')}",
                    'Portfolio Type': portfolio_type,
                    'Number of Assets': len(st.session_state.symbols),
                    'VaR Model': var_model,
                    'Current VaR': f"${var_result:,.2f}" if 'var_result' in locals() else "Not calculated"
                }
                
                if portfolio_type == "Options Portfolio":
                    export_stats['Volatility'] = f"{st.session_state.current_returns.std() * np.sqrt(252) * 100:.2f}%" if st.session_state.current_returns is not None else "N/A"
                
                stats_df = create_metrics_dataframe(export_stats, "Export Statistics")
                safe_dataframe_display(stats_df, "Export Statistics", "export_stats")

        with tab8:  # Help
            st.header("❓ Help & Documentation")

            st.markdown(f"""
            ## 🚀 Welcome to the VaR & Risk Analytics Platform
            
            This comprehensive platform provides sophisticated financial risk modeling capabilities for portfolio management and risk assessment.
            
            ### 📊 Getting Started
            
            1. **Select Portfolio Type**: Choose from Single Asset, Multi-Asset, Crypto, or Options Portfolio
            2. **Configure Data Source**: Select appropriate data source for your analysis
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
            
            {"#### Options VaR Models" if portfolio_type == "Options Portfolio" else "#### Standard Models"}
            {"1. **Delta-Gamma Parametric**: Advanced options VaR using Taylor expansion with Greeks" if portfolio_type == "Options Portfolio" else "1. **Parametric (Delta-Normal)**: Classical normal distribution approach"}
            {"2. **Historical Simulation**: Non-parametric historical method" if portfolio_type == "Options Portfolio" else "2. **Historical Simulation**: Non-parametric historical method"}
            {"3. **Parametric (Delta-Normal)**: Classical normal distribution approach" if portfolio_type == "Options Portfolio" else "3. **Monte Carlo**: Simulation-based approach (1K-100K simulations)"}
            {"4. **Monte Carlo**: Simulation-based approach for options" if portfolio_type == "Options Portfolio" else "4. **GARCH**: Advanced volatility modeling for time-varying risk"}
            {"5. **Historic Simulation**: Enhanced historical method for options" if portfolio_type == "Options Portfolio" else "5. **Extreme Value Theory (EVT)**: Tail risk modeling for extreme events"}
            
            ### 🆕 New Features
            
            #### Enhanced Options Analysis
            - **Delta-Gamma VaR**: Advanced parametric method using option Greeks
            - **Real-time Greeks Calculation**: Delta, Gamma, Theta, Vega
            - **Options Chain Integration**: Live market data support
            
            #### Improved Backtesting
            - **Auto Data Extension**: Automatically fetches sufficient data for backtesting
            - **Enhanced Validation**: Comprehensive model performance assessment
            - **Basel Compliance**: Traffic light system for regulatory compliance
            
            #### Advanced Stress Testing
            - **Custom Scenarios**: Design your own stress test parameters
            - **Enhanced Visualizations**: Distribution comparisons and impact analysis
            - **Multiple Shock Types**: Volatility, correlation, market, and liquidity shocks
            
            #### Comprehensive Export
            - **Multiple Formats**: CSV for data, JSON for complete exports
            - **Timestamped Files**: All exports include generation timestamps
            - **Metadata Inclusion**: Complete analysis parameters and settings
            
            ### 🚨 Troubleshooting
            
            #### Common Issues
            - **"Insufficient data"**: System auto-extends data period for backtesting
            - **"GARCH model failed"**: Requires minimum 100 observations
            - **"Symbol not found"**: Verify ticker format (add -USD for crypto)
            - **"Weights don't sum to 1"**: Portfolio weights are automatically normalized
            - **"Option price data not found"**: Reload data after changing portfolio type
            
            #### Options-Specific Issues
            - **"No options data found"**: Symbol may not have listed options
            - **"Options chain empty"**: Try a different expiry date
            - **"Strike not available"**: System will find closest available strike
            - **"Time to expiry too short"**: Minimum 1 day required
            - **"Delta-Gamma calculation failed"**: Check underlying data availability
            
            #### Performance Tips
            - **Large Datasets**: Use date range filters for better performance
            - **Monte Carlo**: Reduce simulations if processing is slow
            - **Rolling Analysis**: Adjust window size based on data availability
            - **Export Large Data**: Use CSV format for large datasets
            """)

            # Feature suggestions
            st.subheader("💡 Suggested Additional Features")
            
            st.markdown("""
            <div class="feature-highlight">
            <h4>🔮 Potential Enhancements</h4>
            <ul>
                <li><strong>Machine Learning VaR</strong>: LSTM/GRU models for time series prediction</li>
                <li><strong>Real-time Monitoring</strong>: Live portfolio tracking with alerts</li>
                <li><strong>Multi-currency Support</strong>: FX risk analysis and hedging</li>
                <li><strong>Regulatory Reporting</strong>: Automated compliance report generation</li>
                <li><strong>Portfolio Optimization</strong>: Mean-variance and risk parity optimization</li>
                <li><strong>Credit Risk Integration</strong>: Corporate bond and credit default analysis</li>
                <li><strong>ESG Risk Metrics</strong>: Environmental, social, governance risk factors</li>
                <li><strong>Alternative Data</strong>: Sentiment analysis and news impact</li>
                <li><strong>API Integration</strong>: Bloomberg, Refinitiv, and other data providers</li>
                <li><strong>Mobile Dashboard</strong>: Responsive design for mobile devices</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

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
                <p>6 sophisticated VaR calculation methods including Delta-Gamma for options and EVT for tail risk</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔄 Real-time Analysis</h3>
                <p>Dynamic updates with intelligent data management and automatic backtesting data extension</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🧪 Comprehensive Testing</h3>
                <p>Enhanced backtesting, custom stress testing, and regulatory compliance metrics</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
