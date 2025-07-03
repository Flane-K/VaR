import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
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
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #00ff88, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff88;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

def safe_dataframe_display(df, title="Data"):
    """Safely display dataframe with error handling"""
    try:
        if df is not None and not df.empty:
            # Convert all columns to string to avoid Arrow conversion issues
            display_df = df.copy()
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info(f"No {title.lower()} available")
    except Exception as e:
        st.error(f"Error displaying {title.lower()}: {str(e)}")
        # Fallback to simple display
        try:
            st.write(df)
        except:
            st.write(f"Unable to display {title.lower()}")

def get_options_chain(symbol):
    """Get options chain for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        options_dates = ticker.options
        
        if not options_dates:
            return None, None
        
        # Get options for the first available date (usually nearest expiry)
        options_date = options_dates[0]
        options_chain = ticker.option_chain(options_date)
        
        return options_chain, options_dates
    except Exception as e:
        st.error(f"Error fetching options chain: {str(e)}")
        return None, None

def find_closest_atm_option(options_chain, current_price, option_type='call'):
    """Find the closest at-the-money option"""
    try:
        if option_type.lower() == 'call':
            options_df = options_chain.calls
        else:
            options_df = options_chain.puts
        
        # Find strike closest to current price
        options_df['strike_diff'] = abs(options_df['strike'] - current_price)
        closest_option = options_df.loc[options_df['strike_diff'].idxmin()]
        
        return closest_option
    except Exception as e:
        st.error(f"Error finding ATM option: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'current_returns' not in st.session_state:
        st.session_state.current_returns = None
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    if 'portfolio_type' not in st.session_state:
        st.session_state.portfolio_type = None
    if 'options_data' not in st.session_state:
        st.session_state.options_data = None
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None

def main():
    # Initialize session state
    initialize_session_state()
    
    # Initialize instances
    instances = {
        'data_ingestion': DataIngestion(),
        'var_engines': VaREngines(),
        'backtesting': Backtesting(),
        'stress_testing': StressTesting(),
        'rolling_analysis': RollingAnalysis(),
        'options_var': OptionsVaR(),
        'visualization': Visualization(),
        'utils': Utils()
    }
    
    # Main header
    st.markdown('<h1 class="main-header">📊 VaR & Risk Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Portfolio Type Selection
        portfolio_type = st.selectbox(
            "Portfolio Type",
            ["Single Asset", "Multi-Asset", "Crypto Portfolio", "Options Portfolio"],
            key="portfolio_type_select"
        )
        
        # Data Source Selection
        data_source = st.selectbox(
            "Data Source",
            ["Live Market Data", "File Upload", "Manual Entry", "Synthetic Data"],
            key="data_source_select"
        )
        
        # Default date range (1 year)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        date_range = st.date_input(
            "Default Date Range",
            value=(start_date, end_date),
            max_value=datetime.now(),
            key="default_date_range"
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        
        # VaR Model Selection
        var_model = st.selectbox(
            "VaR Model",
            ["Parametric (Delta-Normal)", "Historical Simulation", "Monte Carlo", "GARCH", "Extreme Value Theory"],
            key="var_model_select"
        )
        
        # Risk Parameters
        st.subheader("Risk Parameters")
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key="confidence_level")
        time_horizon = st.slider("Time Horizon (days)", 1, 30, 1, key="time_horizon")
        
        # Model-specific parameters
        if var_model == "Monte Carlo":
            num_simulations = st.slider("Number of Simulations", 1000, 100000, 10000, 1000, key="num_simulations")
        elif var_model == "GARCH":
            garch_p = st.slider("GARCH p", 1, 3, 1, key="garch_p")
            garch_q = st.slider("GARCH q", 1, 3, 1, key="garch_q")
        
        # Backtesting Parameters
        st.subheader("Backtesting Parameters")
        backtest_window = st.slider("Backtesting Window", 30, 1000, 252, key="backtest_window")
    
    # Data Loading Section
    data_loaded = False
    
    if portfolio_type == "Options Portfolio":
        # Options Portfolio Handling
        if data_source == "Live Market Data":
            st.subheader("📊 Options Data Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                underlying_symbol = st.text_input("Underlying Symbol", value="AAPL", key="underlying_symbol")
                option_type = st.selectbox("Option Type", ["Call", "Put"], key="option_type")
            
            with col2:
                custom_strike = st.number_input("Custom Strike (0 for ATM)", value=0.0, key="custom_strike")
                custom_expiry_days = st.number_input("Days to Expiry (0 for default)", value=0, min_value=0, key="custom_expiry")
            
            if st.button("Load Options Data", key="load_options_data"):
                with st.spinner("Loading options data..."):
                    try:
                        # Get current stock price
                        ticker = yf.Ticker(underlying_symbol)
                        current_price = ticker.history(period="1d")['Close'].iloc[-1]
                        
                        # Get options chain
                        options_chain, options_dates = get_options_chain(underlying_symbol)
                        
                        if options_chain is not None:
                            # Find appropriate option
                            if custom_strike > 0:
                                # Find closest to custom strike
                                if option_type.lower() == 'call':
                                    options_df = options_chain.calls
                                else:
                                    options_df = options_chain.puts
                                
                                options_df['strike_diff'] = abs(options_df['strike'] - custom_strike)
                                selected_option = options_df.loc[options_df['strike_diff'].idxmin()]
                            else:
                                # Find ATM option
                                selected_option = find_closest_atm_option(options_chain, current_price, option_type)
                            
                            if selected_option is not None:
                                # Calculate time to expiry
                                expiry_date = pd.to_datetime(options_dates[0])
                                time_to_expiry = (expiry_date - pd.Timestamp.now()).days / 365.0
                                
                                # Store options data
                                st.session_state.options_data = {
                                    'spot_price': current_price,
                                    'strike_price': selected_option['strike'],
                                    'time_to_expiry': time_to_expiry,
                                    'risk_free_rate': 0.05,  # Default 5%
                                    'volatility': selected_option.get('impliedVolatility', 0.25),
                                    'option_type': option_type.lower(),
                                    'market_price': selected_option['lastPrice'],
                                    'underlying_symbol': underlying_symbol
                                }
                                
                                st.session_state.selected_option = selected_option
                                data_loaded = True
                                st.success(f"Loaded {option_type} option: Strike ${selected_option['strike']:.2f}, Expiry: {options_dates[0]}")
                            else:
                                st.error("Could not find suitable option")
                        else:
                            st.error("No options data available for this symbol")
                    except Exception as e:
                        st.error(f"Error loading options data: {str(e)}")
        
        elif data_source == "Manual Entry":
            st.subheader("📊 Manual Options Entry")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                spot_price = st.number_input("Spot Price ($)", value=100.0, min_value=0.01, key="manual_spot")
                strike_price = st.number_input("Strike Price ($)", value=105.0, min_value=0.01, key="manual_strike")
            
            with col2:
                time_to_expiry = st.number_input("Time to Expiry (years)", value=0.25, min_value=0.01, max_value=5.0, key="manual_expiry")
                risk_free_rate = st.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=1.0, key="manual_rate")
            
            with col3:
                volatility = st.number_input("Volatility", value=0.25, min_value=0.01, max_value=5.0, key="manual_vol")
                option_type = st.selectbox("Option Type", ["call", "put"], key="manual_option_type")
            
            if st.button("Generate Options Data", key="generate_options_data"):
                st.session_state.options_data = {
                    'spot_price': spot_price,
                    'strike_price': strike_price,
                    'time_to_expiry': time_to_expiry,
                    'risk_free_rate': risk_free_rate,
                    'volatility': volatility,
                    'option_type': option_type,
                    'market_price': None,
                    'underlying_symbol': 'MANUAL'
                }
                data_loaded = True
                st.success("Manual options data generated successfully!")
    
    else:
        # Regular Portfolio Handling
        if data_source == "Live Market Data":
            if portfolio_type == "Single Asset":
                symbol = st.text_input("Enter Symbol", value="AAPL", key="single_symbol")
                symbols = [symbol]
                weights = [1.0]
            elif portfolio_type == "Multi-Asset":
                symbols_input = st.text_input("Enter Symbols (comma-separated)", value="AAPL,GOOGL,MSFT,TSLA", key="multi_symbols")
                symbols = [s.strip().upper() for s in symbols_input.split(",")]
                
                # Portfolio weights
                st.subheader("Portfolio Weights")
                weights = []
                cols = st.columns(len(symbols))
                for i, symbol in enumerate(symbols):
                    with cols[i]:
                        weight = st.number_input(f"{symbol}", value=1.0/len(symbols), min_value=0.0, max_value=1.0, key=f"weight_{symbol}")
                        weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
                    st.info(f"Normalized weights: {[f'{w:.3f}' for w in weights]}")
            
            elif portfolio_type == "Crypto Portfolio":
                symbols_input = st.text_input("Enter Crypto Symbols", value="BTC-USD,ETH-USD,ADA-USD", key="crypto_symbols")
                symbols = [s.strip().upper() for s in symbols_input.split(",")]
                
                # Portfolio weights
                st.subheader("Portfolio Weights")
                weights = []
                cols = st.columns(len(symbols))
                for i, symbol in enumerate(symbols):
                    with cols[i]:
                        weight = st.number_input(f"{symbol}", value=1.0/len(symbols), min_value=0.0, max_value=1.0, key=f"crypto_weight_{symbol}")
                        weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
            
            if st.button("Load Data", key="load_live_data"):
                with st.spinner("Loading market data..."):
                    data = instances['data_ingestion'].load_live_data(symbols, start_date, end_date)
                    if data is not None:
                        st.session_state.current_data = data
                        st.session_state.current_returns = instances['data_ingestion'].get_portfolio_returns(weights)
                        st.session_state.portfolio_weights = weights
                        st.session_state.data_source = data_source
                        st.session_state.portfolio_type = portfolio_type
                        data_loaded = True
                        st.success("Data loaded successfully!")
                    else:
                        st.error("Failed to load data")
        
        elif data_source == "File Upload":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="file_upload")
            if uploaded_file is not None:
                data = instances['data_ingestion'].load_csv_data(uploaded_file)
                if data is not None:
                    st.session_state.current_data = data
                    
                    # Equal weights for uploaded data
                    num_assets = len(data.columns)
                    weights = [1.0/num_assets] * num_assets
                    st.session_state.current_returns = instances['data_ingestion'].get_portfolio_returns(weights)
                    st.session_state.portfolio_weights = weights
                    st.session_state.data_source = data_source
                    st.session_state.portfolio_type = portfolio_type
                    data_loaded = True
                    st.success("File uploaded successfully!")
        
        elif data_source == "Synthetic Data":
            col1, col2 = st.columns(2)
            with col1:
                num_days = st.number_input("Number of Days", value=500, min_value=100, max_value=2000, key="synth_days")
                initial_price = st.number_input("Initial Price", value=100.0, min_value=1.0, key="synth_price")
            with col2:
                annual_return = st.number_input("Annual Return", value=0.08, min_value=-0.5, max_value=0.5, key="synth_return")
                annual_volatility = st.number_input("Annual Volatility", value=0.20, min_value=0.01, max_value=1.0, key="synth_vol")
            
            if st.button("Generate Synthetic Data", key="generate_synth_data"):
                data = instances['data_ingestion'].generate_synthetic_data(num_days, initial_price, annual_return, annual_volatility)
                if data is not None:
                    st.session_state.current_data = data
                    st.session_state.current_returns = instances['data_ingestion'].returns
                    st.session_state.portfolio_weights = [1.0]
                    st.session_state.data_source = data_source
                    st.session_state.portfolio_type = portfolio_type
                    data_loaded = True
                    st.success("Synthetic data generated successfully!")
    
    # Check if we have data (either newly loaded or from session state)
    has_data = data_loaded or st.session_state.current_data is not None or st.session_state.options_data is not None
    
    if has_data:
        # Create tabs based on portfolio type
        if portfolio_type == "Options Portfolio" and st.session_state.options_data is not None:
            # Options Portfolio Tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Options Dashboard", "📈 Options VaR", "📋 Data Overview", 
                "🔄 Rolling Analysis", "🧪 Backtesting", "⚡ Stress Testing", "❓ Help"
            ])
            
            # Options Dashboard Tab
            with tab1:
                st.header("📊 Options Portfolio Dashboard")
                
                options_data = st.session_state.options_data
                
                # Calculate option price and Greeks
                option_price = instances['options_var'].black_scholes_price(
                    options_data['spot_price'], options_data['strike_price'],
                    options_data['time_to_expiry'], options_data['risk_free_rate'],
                    options_data['volatility'], options_data['option_type']
                )
                
                greeks = instances['options_var'].calculate_greeks(
                    options_data['spot_price'], options_data['strike_price'],
                    options_data['time_to_expiry'], options_data['risk_free_rate'],
                    options_data['volatility'], options_data['option_type']
                )
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Option Price", f"${option_price:.2f}")
                with col2:
                    st.metric("Delta", f"{greeks.get('delta', 0):.4f}")
                with col3:
                    st.metric("Gamma", f"{greeks.get('gamma', 0):.4f}")
                with col4:
                    st.metric("Theta", f"{greeks.get('theta', 0):.4f}")
                
                # Option Payoff Chart
                S_range = np.linspace(options_data['spot_price'] * 0.7, options_data['spot_price'] * 1.3, 100)
                fig_payoff = instances['visualization'].plot_options_payoff(
                    S_range, options_data['strike_price'], options_data['option_type'],
                    options_data['time_to_expiry'], options_data['risk_free_rate'], options_data['volatility']
                )
                st.plotly_chart(fig_payoff, use_container_width=True, key="options_dashboard_payoff")
                
                # Greeks Table
                st.subheader("Greeks Analysis")
                greeks_df = pd.DataFrame([greeks]).T
                greeks_df.columns = ['Value']
                safe_dataframe_display(greeks_df, "Greeks")
            
            # Options VaR Tab
            with tab2:
                st.header("📈 Options VaR Analysis")
                
                # VaR Method Selection for Options
                var_method_options = st.selectbox(
                    "VaR Method",
                    ["Delta-Normal", "Delta-Gamma", "Full Revaluation Monte Carlo", "Historical Simulation"],
                    key="options_var_method"
                )
                
                # Calculate Options VaR
                if var_method_options == "Historical Simulation":
                    # For historical simulation, we need underlying price data
                    if 'underlying_symbol' in options_data and options_data['underlying_symbol'] != 'MANUAL':
                        try:
                            # Get underlying price data
                            ticker = yf.Ticker(options_data['underlying_symbol'])
                            hist_data = ticker.history(period="1y")
                            underlying_returns = hist_data['Close'].pct_change().dropna()
                            
                            # Calculate historical VaR for options
                            var_result = instances['options_var'].calculate_options_var_historical(
                                options_data['spot_price'], options_data['strike_price'],
                                options_data['time_to_expiry'], options_data['risk_free_rate'],
                                options_data['volatility'], options_data['option_type'],
                                underlying_returns, confidence_level
                            )
                        except Exception as e:
                            st.error(f"Error calculating historical VaR: {str(e)}")
                            var_result = {}
                    else:
                        st.warning("Historical simulation requires live market data for underlying asset")
                        var_result = {}
                else:
                    var_result = instances['options_var'].calculate_options_var(
                        options_data['spot_price'], options_data['strike_price'],
                        options_data['time_to_expiry'], options_data['risk_free_rate'],
                        options_data['volatility'], options_data['option_type'],
                        var_method_options, confidence_level
                    )
                
                if var_result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("VaR", f"${var_result.get('var', 0):.2f}")
                    with col2:
                        st.metric("Current Price", f"${var_result.get('current_price', 0):.2f}")
                    with col3:
                        st.metric("Method", var_result.get('method', 'N/A'))
                    
                    # VaR Results Table
                    st.subheader("Detailed VaR Results")
                    var_df = pd.DataFrame([var_result]).T
                    var_df.columns = ['Value']
                    safe_dataframe_display(var_df, "VaR Results")
        
        else:
            # Regular Portfolio Tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Dashboard", "📈 VaR Calculator", "📋 Data Overview", 
                "🔄 Rolling Analysis", "🧪 Backtesting", "⚡ Stress Testing", 
                "📊 Options Analysis", "❓ Help"
            ])
            
            # Dashboard Tab
            with tab1:
                st.header("📊 Portfolio Dashboard")
                
                if st.session_state.current_returns is not None:
                    # Key Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_return = float((1 + st.session_state.current_returns).prod() - 1)
                        st.metric(
                            label="Total Return",
                            value=f"{total_return:.2%}",
                            delta="Since Inception"
                        )
                    
                    with col2:
                        annual_return = float(st.session_state.current_returns.mean() * 252)
                        st.metric(
                            label="Annualized Return",
                            value=f"{annual_return:.2%}",
                            delta="Historical"
                        )
                    
                    with col3:
                        current_vol = float(st.session_state.current_returns.std() * np.sqrt(252) * 100)
                        st.metric(
                            label="Annualized Volatility",
                            value=f"{current_vol:.2f}%",
                            delta="Historical"
                        )
                    
                    with col4:
                        if current_vol > 0:
                            sharpe_ratio = annual_return / (current_vol / 100)
                        else:
                            sharpe_ratio = 0
                        st.metric(
                            label="Sharpe Ratio",
                            value=f"{sharpe_ratio:.3f}",
                            delta="Risk-Adjusted"
                        )
                    
                    # Performance Chart
                    cumulative_returns = (1 + st.session_state.current_returns).cumprod()
                    
                    # Time range selector for dashboard
                    st.subheader("📈 Performance Chart")
                    dash_col1, dash_col2 = st.columns(2)
                    with dash_col1:
                        dash_start = st.date_input("Chart Start Date", value=start_date, key="dash_start")
                    with dash_col2:
                        dash_end = st.date_input("Chart End Date", value=end_date, key="dash_end")
                    
                    # Filter data for selected period
                    mask = (cumulative_returns.index >= pd.Timestamp(dash_start)) & (cumulative_returns.index <= pd.Timestamp(dash_end))
                    filtered_cumulative = cumulative_returns[mask]
                    
                    fig_perf = instances['visualization'].plot_performance_chart(filtered_cumulative)
                    st.plotly_chart(fig_perf, use_container_width=True, key="dashboard_performance_chart")
                    
                    # Distribution Chart
                    filtered_returns = st.session_state.current_returns[mask]
                    if len(filtered_returns) > 0:
                        # Calculate VaR for the filtered period
                        if var_model == "Parametric (Delta-Normal)":
                            var_value = instances['var_engines'].calculate_parametric_var(filtered_returns, confidence_level, time_horizon)
                        elif var_model == "Historical Simulation":
                            var_value = instances['var_engines'].calculate_historical_var(filtered_returns, confidence_level, time_horizon)
                        elif var_model == "Monte Carlo":
                            var_value = instances['var_engines'].calculate_monte_carlo_var(filtered_returns, confidence_level, time_horizon, num_simulations)
                        elif var_model == "GARCH":
                            var_value = instances['var_engines'].calculate_garch_var(filtered_returns, confidence_level, time_horizon, garch_p, garch_q)
                        elif var_model == "Extreme Value Theory":
                            var_value = instances['var_engines'].calculate_evt_var(filtered_returns, confidence_level)
                        
                        fig_dist = instances['visualization'].plot_var_distribution(
                            filtered_returns, confidence_level, var_value
                        )
                        st.plotly_chart(fig_dist, use_container_width=True, key="dashboard_distribution_chart")
                        
                        # Portfolio Statistics
                        st.subheader("Portfolio Statistics")
                        stats = instances['utils'].calculate_portfolio_statistics(filtered_returns)
                        
                        stats_data = {
                            'Annual Return': f"{stats.get('annual_return', 0):.2%}",
                            'Annual Volatility': f"{stats.get('annual_volatility', 0):.2%}",
                            'Sharpe Ratio': f"{stats.get('sharpe_ratio', 0):.3f}",
                            'Sortino Ratio': f"{stats.get('sortino_ratio', 0):.3f}",
                            'Max Drawdown': f"{stats.get('max_drawdown', 0):.2%}",
                            'Skewness': f"{stats.get('skewness', 0):.3f}",
                            'Kurtosis': f"{stats.get('kurtosis', 0):.3f}"
                        }
                        
                        stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
                        safe_dataframe_display(stats_df, "Portfolio Statistics")
            
            # VaR Calculator Tab
            with tab2:
                st.header("📈 VaR Calculator")
                
                if st.session_state.current_returns is not None:
                    # Time range selector for VaR
                    st.subheader("📅 Analysis Period")
                    var_col1, var_col2 = st.columns(2)
                    with var_col1:
                        var_start = st.date_input("VaR Start Date", value=start_date, key="var_start")
                    with var_col2:
                        var_end = st.date_input("VaR End Date", value=end_date, key="var_end")
                    
                    # Filter returns for selected period
                    mask = (st.session_state.current_returns.index >= pd.Timestamp(var_start)) & (st.session_state.current_returns.index <= pd.Timestamp(var_end))
                    var_portfolio_returns = st.session_state.current_returns[mask]
                    
                    if len(var_portfolio_returns) > 0:
                        # Calculate VaR using selected model
                        if var_model == "Parametric (Delta-Normal)":
                            var_value = instances['var_engines'].calculate_parametric_var(var_portfolio_returns, confidence_level, time_horizon)
                            expected_shortfall = instances['var_engines'].calculate_expected_shortfall(var_portfolio_returns, confidence_level)
                        elif var_model == "Historical Simulation":
                            var_value = instances['var_engines'].calculate_historical_var(var_portfolio_returns, confidence_level, time_horizon)
                            expected_shortfall = instances['var_engines'].calculate_expected_shortfall(var_portfolio_returns, confidence_level)
                        elif var_model == "Monte Carlo":
                            var_value = instances['var_engines'].calculate_monte_carlo_var(var_portfolio_returns, confidence_level, time_horizon, num_simulations)
                            expected_shortfall = instances['var_engines'].calculate_expected_shortfall(var_portfolio_returns, confidence_level)
                        elif var_model == "GARCH":
                            var_value = instances['var_engines'].calculate_garch_var(var_portfolio_returns, confidence_level, time_horizon, garch_p, garch_q)
                            expected_shortfall = instances['var_engines'].calculate_expected_shortfall(var_portfolio_returns, confidence_level)
                        elif var_model == "Extreme Value Theory":
                            var_value = instances['var_engines'].calculate_evt_var(var_portfolio_returns, confidence_level)
                            expected_shortfall = instances['var_engines'].calculate_expected_shortfall(var_portfolio_returns, confidence_level)
                        
                        # Display VaR Results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Value at Risk", f"${var_value:,.2f}")
                        with col2:
                            st.metric("Expected Shortfall", f"${expected_shortfall:,.2f}")
                        with col3:
                            st.metric("Confidence Level", f"{confidence_level:.0%}")
                        
                        # VaR Distribution Plot
                        fig_dist = instances['visualization'].plot_var_distribution(
                            var_portfolio_returns, confidence_level, var_value
                        )
                        st.plotly_chart(fig_dist, use_container_width=True, key="var_calculator_distribution")
                        
                        # Model Comparison
                        st.subheader("Model Comparison")
                        comparison_results = {}
                        
                        models = ["Parametric (Delta-Normal)", "Historical Simulation", "Monte Carlo"]
                        for model in models:
                            try:
                                if model == "Parametric (Delta-Normal)":
                                    comparison_results[model] = instances['var_engines'].calculate_parametric_var(var_portfolio_returns, confidence_level, time_horizon)
                                elif model == "Historical Simulation":
                                    comparison_results[model] = instances['var_engines'].calculate_historical_var(var_portfolio_returns, confidence_level, time_horizon)
                                elif model == "Monte Carlo":
                                    comparison_results[model] = instances['var_engines'].calculate_monte_carlo_var(var_portfolio_returns, confidence_level, time_horizon, 10000)
                            except Exception as e:
                                comparison_results[model] = 0
                                st.warning(f"Error calculating {model}: {str(e)}")
                        
                        if comparison_results:
                            fig_comparison = instances['visualization'].plot_var_comparison(comparison_results)
                            st.plotly_chart(fig_comparison, use_container_width=True, key="var_model_comparison")
                            
                            # Comparison Table
                            comparison_df = pd.DataFrame(list(comparison_results.items()), columns=['Model', 'VaR ($)'])
                            safe_dataframe_display(comparison_df, "Model Comparison")
                    else:
                        st.warning("No data available for the selected time period")
        
        # Data Overview Tab (for all portfolio types)
        with tab3:
            st.header("📋 Data Overview")
            
            if portfolio_type == "Options Portfolio" and st.session_state.options_data is not None:
                # Options Data Overview
                options_data = st.session_state.options_data
                
                st.subheader("📊 Options Parameters")
                options_params = {
                    'Underlying Symbol': options_data.get('underlying_symbol', 'N/A'),
                    'Option Type': options_data['option_type'].title(),
                    'Spot Price': f"${options_data['spot_price']:.2f}",
                    'Strike Price': f"${options_data['strike_price']:.2f}",
                    'Time to Expiry': f"{options_data['time_to_expiry']:.4f} years",
                    'Risk-Free Rate': f"{options_data['risk_free_rate']:.2%}",
                    'Volatility': f"{options_data['volatility']:.2%}",
                    'Market Price': f"${options_data.get('market_price', 0):.2f}" if options_data.get('market_price') else 'N/A'
                }
                
                params_df = pd.DataFrame(list(options_params.items()), columns=['Parameter', 'Value'])
                safe_dataframe_display(params_df, "Options Parameters")
                
                # Calculate and display Greeks
                st.subheader("📈 Greeks Analysis")
                greeks = instances['options_var'].calculate_greeks(
                    options_data['spot_price'], options_data['strike_price'],
                    options_data['time_to_expiry'], options_data['risk_free_rate'],
                    options_data['volatility'], options_data['option_type']
                )
                
                greeks_df = pd.DataFrame([greeks]).T
                greeks_df.columns = ['Value']
                safe_dataframe_display(greeks_df, "Greeks")
            
            else:
                # Regular Portfolio Data Overview
                if st.session_state.current_data is not None:
                    # Data Summary
                    st.subheader("📊 Data Summary")
                    summary = instances['data_ingestion'].get_data_summary()
                    
                    if summary:
                        summary_data = {
                            'Data Points': summary['data_points'],
                            'Date Range': summary['date_range'],
                            'Number of Assets': len(summary['assets']),
                            'Assets': ', '.join(summary['assets'])
                        }
                        
                        summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
                        safe_dataframe_display(summary_df, "Data Summary")
                    
                    # Recent Price Data
                    st.subheader("📈 Recent Price Data")
                    safe_dataframe_display(st.session_state.current_data.tail(20), "Recent Prices")
                    
                    # Recent Returns Data
                    if st.session_state.current_returns is not None:
                        st.subheader("📊 Recent Returns Data")
                        if hasattr(st.session_state.current_returns, 'tail'):
                            safe_dataframe_display(st.session_state.current_returns.tail(20).to_frame('Returns'), "Recent Returns")
                        else:
                            st.write("Returns data available but cannot display in table format")
                    
                    # Portfolio Composition
                    if st.session_state.portfolio_weights is not None and len(st.session_state.current_data.columns) > 1:
                        st.subheader("💼 Portfolio Composition")
                        composition_data = {
                            'Asset': st.session_state.current_data.columns,
                            'Weight': st.session_state.portfolio_weights,
                            'Percentage': [f"{w:.2%}" for w in st.session_state.portfolio_weights]
                        }
                        
                        composition_df = pd.DataFrame(composition_data)
                        safe_dataframe_display(composition_df, "Portfolio Composition")
                    
                    # Data Quality Report
                    st.subheader("🔍 Data Quality Report")
                    quality_report = instances['utils'].validate_data_quality(st.session_state.current_data)
                    
                    if quality_report['valid']:
                        st.success("✅ Data quality validation passed")
                    else:
                        st.error("❌ Data quality issues detected")
                    
                    if quality_report['warnings']:
                        st.warning("⚠️ Warnings:")
                        for warning in quality_report['warnings']:
                            st.write(f"• {warning}")
                    
                    if quality_report['recommendations']:
                        st.info("💡 Recommendations:")
                        for rec in quality_report['recommendations']:
                            st.write(f"• {rec}")
        
        # Continue with other tabs...
        # (Rolling Analysis, Backtesting, Stress Testing tabs would follow similar pattern)
        
        # Help Tab
        with tab7 if portfolio_type == "Options Portfolio" else tab8:
            st.header("❓ Help & Documentation")
            
            st.markdown("""
            ## 🚀 Getting Started
            
            ### Portfolio Types
            - **Single Asset**: Analyze risk for individual stocks or assets
            - **Multi-Asset**: Create diversified portfolios with custom weights
            - **Crypto Portfolio**: Specialized analysis for cryptocurrency portfolios
            - **Options Portfolio**: Advanced derivatives risk modeling
            
            ### Data Sources
            - **Live Market Data**: Real-time data from Yahoo Finance
            - **File Upload**: CSV files with price data
            - **Manual Entry**: Custom data input
            - **Synthetic Data**: Generated data for testing
            
            ### VaR Models
            - **Parametric**: Normal distribution assumption
            - **Historical Simulation**: Non-parametric historical approach
            - **Monte Carlo**: Simulation-based method
            - **GARCH**: Time-varying volatility modeling
            - **Extreme Value Theory**: Tail risk modeling
            
            ### Options Analysis
            - **Delta-Normal**: Linear approximation
            - **Delta-Gamma**: Second-order approximation
            - **Full Revaluation**: Complete option repricing
            - **Historical Simulation**: Historical price movements
            
            ### Symbol Formats
            - **Stocks**: AAPL, GOOGL, MSFT
            - **Crypto**: BTC-USD, ETH-USD, ADA-USD
            - **Options**: Automatically fetched for underlying symbols
            
            ## 📊 Features
            
            ### Risk Metrics
            - Value at Risk (VaR)
            - Expected Shortfall (ES)
            - Maximum Drawdown
            - Sharpe Ratio
            - Sortino Ratio
            
            ### Analysis Tools
            - Rolling risk metrics
            - Backtesting validation
            - Stress testing scenarios
            - Correlation analysis
            - Performance attribution
            
            ## 🔧 Tips
            
            1. **Data Quality**: Ensure sufficient historical data for reliable results
            2. **Model Selection**: Choose appropriate VaR model based on data characteristics
            3. **Backtesting**: Validate model performance before making decisions
            4. **Stress Testing**: Consider extreme scenarios in risk assessment
            5. **Regular Updates**: Refresh data and recalibrate models regularly
            
            ## 📞 Support
            
            For technical support or feature requests, please refer to the documentation
            or contact the development team.
            """)
    
    else:
        st.info("👆 Please configure your portfolio and load data using the sidebar to begin analysis.")
        
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        1. **Select Portfolio Type** in the sidebar
        2. **Choose Data Source** (Live Market, File Upload, Manual Entry, or Synthetic)
        3. **Configure Parameters** (symbols, weights, dates)
        4. **Load Data** using the appropriate button
        5. **Explore Analysis** across different tabs
        
        ### 📊 Available Analysis
        - **Dashboard**: Key metrics and performance overview
        - **VaR Calculator**: Risk measurement and model comparison
        - **Rolling Analysis**: Time-varying risk metrics
        - **Backtesting**: Model validation and testing
        - **Stress Testing**: Scenario analysis and sensitivity
        - **Options Analysis**: Advanced derivatives modeling
        
        ### 🎯 Portfolio Types
        - **Single Asset**: Individual stock/crypto analysis
        - **Multi-Asset**: Diversified portfolio with custom weights
        - **Crypto Portfolio**: Cryptocurrency-focused analysis
        - **Options Portfolio**: Advanced derivatives risk modeling
        """)

if __name__ == "__main__":
    main()