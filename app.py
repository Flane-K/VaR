import streamlit as st
import pandas as pd
import numpy as np
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
from src.option_data_fetcher import OptionsDataFetcher
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
        color: #00ff88;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4ecdc4;
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00ff88;
        margin: 0.5rem 0;
    }
    
    .stMetric {
        background: transparent;
    }
    
    .risk-warning {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .success-message {
        background: linear-gradient(135deg, #6bcf7f 0%, #4ecdc4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #262730;
        border-radius: 10px 10px 0px 0px;
        color: #ffffff;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00ff88;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'current_returns' not in st.session_state:
        st.session_state.current_returns = None
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = None
    if 'var_results' not in st.session_state:
        st.session_state.var_results = {}
    if 'options_data' not in st.session_state:
        st.session_state.options_data = None

def create_instances():
    """Create instances of all classes"""
    return {
        'data_ingestion': DataIngestion(),
        'var_engines': VaREngines(),
        'backtesting': Backtesting(),
        'stress_testing': StressTesting(),
        'rolling_analysis': RollingAnalysis(),
        'options_var': OptionsVaR(),
        'options_fetcher': OptionsDataFetcher(),
        'visualization': Visualization(),
        'utils': Utils()
    }

def sidebar_configuration():
    """Create sidebar configuration"""
    st.sidebar.markdown('<div class="sub-header">📊 Configuration</div>', unsafe_allow_html=True)
    
    # Portfolio Type Selection
    portfolio_type = st.sidebar.selectbox(
        "Portfolio Type",
        ["Single Asset", "Multi-Asset", "Crypto Portfolio", "Options Portfolio"],
        help="Select the type of portfolio to analyze"
    )
    
    # Data Source Selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Live Market Data", "File Upload", "Manual Entry", "Synthetic Data"],
        help="Choose how to load your data"
    )
    
    # VaR Model Selection
    var_model = st.sidebar.selectbox(
        "VaR Model",
        ["Parametric (Delta-Normal)", "Historical Simulation", "Monte Carlo", "GARCH", "Extreme Value Theory"],
        help="Select the VaR calculation method"
    )
    
    # Risk Parameters
    st.sidebar.markdown("### Risk Parameters")
    
    confidence_level = st.sidebar.slider(
        "Confidence Level",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f",
        help="Confidence level for VaR calculation"
    )
    
    time_horizon = st.sidebar.slider(
        "Time Horizon (days)",
        min_value=1,
        max_value=30,
        value=1,
        help="Time horizon for VaR calculation"
    )
    
    historical_window = st.sidebar.slider(
        "Historical Window (days)",
        min_value=30,
        max_value=1000,
        value=252,
        help="Number of historical days to use"
    )
    
    return {
        'portfolio_type': portfolio_type,
        'data_source': data_source,
        'var_model': var_model,
        'confidence_level': confidence_level,
        'time_horizon': time_horizon,
        'historical_window': historical_window
    }

def load_data_section(config, instances):
    """Handle data loading based on configuration"""
    data_ingestion = instances['data_ingestion']
    
    if config['portfolio_type'] == "Options Portfolio":
        return load_options_data(config, instances)
    
    if config['data_source'] == "Live Market Data":
        return load_live_data(config, data_ingestion)
    elif config['data_source'] == "File Upload":
        return load_file_data(data_ingestion)
    elif config['data_source'] == "Manual Entry":
        return load_manual_data(data_ingestion)
    elif config['data_source'] == "Synthetic Data":
        return load_synthetic_data(data_ingestion)
    
    return None, None, None

def load_options_data(config, instances):
    """Load options data based on data source"""
    options_fetcher = instances['options_fetcher']
    
    if config['data_source'] == "Live Market Data":
        st.sidebar.markdown("### Options Configuration")
        
        # Symbol input
        symbol = st.sidebar.text_input("Underlying Symbol", value="AAPL", help="Enter stock symbol")
        
        # Fetch options data
        if st.sidebar.button("Fetch Options Data"):
            with st.spinner("Fetching options data..."):
                options_data = options_fetcher.get_options_data(symbol)
                
                if options_data:
                    st.session_state.options_data = options_data
                    st.sidebar.success("Options data loaded successfully!")
                    
                    # Option selection
                    st.sidebar.markdown("### Option Selection")
                    
                    # Expiry selection
                    expiry_dates = list(options_data['options_chains'].keys())
                    selected_expiry = st.sidebar.selectbox("Expiry Date", expiry_dates)
                    
                    # Option type
                    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
                    
                    # Get available strikes for selected expiry
                    if selected_expiry in options_data['options_chains']:
                        chain = options_data['options_chains'][selected_expiry]
                        if option_type in chain and not chain[option_type].empty:
                            strikes = sorted(chain[option_type]['strike'].unique())
                            
                            # Find closest ATM strike
                            current_price = options_data['current_price']
                            closest_strike = min(strikes, key=lambda x: abs(x - current_price))
                            default_index = strikes.index(closest_strike)
                            
                            selected_strike = st.sidebar.selectbox(
                                "Strike Price", 
                                strikes, 
                                index=default_index
                            )
                            
                            # Get option details
                            option_row = chain[option_type][chain[option_type]['strike'] == selected_strike].iloc[0]
                            
                            # Calculate time to expiry
                            expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d').date()
                            time_to_expiry = (expiry_date - datetime.now().date()).days / 365.0
                            
                            # Store option parameters
                            option_params = {
                                'spot_price': current_price,
                                'strike_price': selected_strike,
                                'time_to_expiry': max(time_to_expiry, 1/365),
                                'risk_free_rate': 0.05,
                                'volatility': option_row.get('impliedVolatility', 0.25),
                                'option_type': option_type,
                                'market_price': option_row.get('lastPrice', 0)
                            }
                            
                            return None, None, option_params
                else:
                    st.sidebar.error("Failed to fetch options data")
    
    elif config['data_source'] == "Manual Entry":
        st.sidebar.markdown("### Manual Options Entry")
        
        spot_price = st.sidebar.number_input("Spot Price", value=100.0, min_value=0.01)
        strike_price = st.sidebar.number_input("Strike Price", value=105.0, min_value=0.01)
        time_to_expiry = st.sidebar.number_input("Time to Expiry (years)", value=0.25, min_value=0.001, max_value=5.0)
        risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=1.0)
        volatility = st.sidebar.number_input("Volatility", value=0.25, min_value=0.01, max_value=5.0)
        option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
        
        option_params = {
            'spot_price': spot_price,
            'strike_price': strike_price,
            'time_to_expiry': time_to_expiry,
            'risk_free_rate': risk_free_rate,
            'volatility': volatility,
            'option_type': option_type
        }
        
        return None, None, option_params
    
    elif config['data_source'] == "Synthetic Data":
        # Generate synthetic underlying data for options analysis
        data_ingestion = instances['data_ingestion']
        data = data_ingestion.generate_synthetic_data()
        
        if data is not None:
            returns = data.pct_change().dropna()
            
            # Default option parameters
            option_params = {
                'spot_price': 100.0,
                'strike_price': 105.0,
                'time_to_expiry': 0.25,
                'risk_free_rate': 0.05,
                'volatility': 0.25,
                'option_type': 'call'
            }
            
            return data, returns, option_params
    
    return None, None, None

def load_live_data(config, data_ingestion):
    """Load live market data"""
    st.sidebar.markdown("### Market Data Configuration")
    
    if config['portfolio_type'] == "Single Asset":
        symbol = st.sidebar.text_input("Symbol", value="AAPL", help="Enter stock symbol")
        symbols = [symbol]
        weights = [1.0]
        
    elif config['portfolio_type'] == "Multi-Asset":
        symbols_input = st.sidebar.text_area(
            "Symbols (comma-separated)", 
            value="AAPL,GOOGL,MSFT,TSLA",
            help="Enter stock symbols separated by commas"
        )
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        # Portfolio weights
        st.sidebar.markdown("### Portfolio Weights")
        weights = []
        for symbol in symbols:
            weight = st.sidebar.number_input(f"Weight for {symbol}", value=1.0/len(symbols), min_value=0.0, max_value=1.0)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        
    elif config['portfolio_type'] == "Crypto Portfolio":
        crypto_symbols = st.sidebar.multiselect(
            "Crypto Assets",
            ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD'],
            default=['BTC-USD']
        )
        symbols = crypto_symbols
        weights = [1.0/len(symbols) for _ in symbols] if symbols else [1.0]
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config['historical_window'] + 50)
    
    start_date_input = st.sidebar.date_input("Start Date", value=start_date.date())
    end_date_input = st.sidebar.date_input("End Date", value=end_date.date())
    
    if st.sidebar.button("Load Data"):
        with st.spinner("Loading market data..."):
            data = data_ingestion.load_live_data(symbols, start_date_input, end_date_input)
            
            if data is not None:
                st.session_state.current_data = data
                st.session_state.current_returns = data_ingestion.get_portfolio_returns(weights)
                st.session_state.portfolio_weights = weights
                st.session_state.data_loaded = True
                st.sidebar.success("Data loaded successfully!")
                return data, st.session_state.current_returns, weights
            else:
                st.sidebar.error("Failed to load data")
    
    return st.session_state.current_data, st.session_state.current_returns, st.session_state.portfolio_weights

def load_file_data(data_ingestion):
    """Load data from uploaded file"""
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = data_ingestion.load_csv_data(uploaded_file)
        if data is not None:
            returns = data_ingestion.returns
            weights = [1.0/len(data.columns) for _ in data.columns]
            
            st.session_state.current_data = data
            st.session_state.current_returns = data_ingestion.get_portfolio_returns(weights)
            st.session_state.portfolio_weights = weights
            st.session_state.data_loaded = True
            st.sidebar.success("File uploaded successfully!")
            return data, st.session_state.current_returns, weights
    
    return None, None, None

def load_manual_data(data_ingestion):
    """Load manually entered data"""
    st.sidebar.markdown("### Manual Data Entry")
    
    num_assets = st.sidebar.number_input("Number of Assets", min_value=1, max_value=10, value=1)
    num_days = st.sidebar.number_input("Number of Days", min_value=30, max_value=1000, value=100)
    
    if st.sidebar.button("Generate Manual Data Template"):
        # This would typically open a data entry interface
        # For now, we'll generate synthetic data as a placeholder
        data = data_ingestion.generate_synthetic_data(num_days=num_days)
        if data is not None:
            returns = data_ingestion.returns
            weights = [1.0]
            
            st.session_state.current_data = data
            st.session_state.current_returns = returns
            st.session_state.portfolio_weights = weights
            st.session_state.data_loaded = True
            st.sidebar.success("Manual data template generated!")
            return data, returns, weights
    
    return None, None, None

def load_synthetic_data(data_ingestion):
    """Load synthetic data"""
    st.sidebar.markdown("### Synthetic Data Parameters")
    
    num_days = st.sidebar.slider("Number of Days", 100, 1000, 500)
    initial_price = st.sidebar.number_input("Initial Price", value=100.0)
    annual_return = st.sidebar.slider("Annual Return", -0.5, 0.5, 0.08, 0.01)
    annual_volatility = st.sidebar.slider("Annual Volatility", 0.05, 1.0, 0.20, 0.01)
    
    if st.sidebar.button("Generate Synthetic Data"):
        data = data_ingestion.generate_synthetic_data(
            num_days=num_days,
            initial_price=initial_price,
            annual_return=annual_return,
            annual_volatility=annual_volatility
        )
        
        if data is not None:
            returns = data_ingestion.returns
            weights = [1.0]
            
            st.session_state.current_data = data
            st.session_state.current_returns = returns
            st.session_state.portfolio_weights = weights
            st.session_state.data_loaded = True
            st.sidebar.success("Synthetic data generated!")
            return data, returns, weights
    
    return None, None, None

def calculate_var_metrics(config, instances, returns=None, options_params=None):
    """Calculate VaR metrics based on configuration"""
    var_engines = instances['var_engines']
    options_var = instances['options_var']
    
    if config['portfolio_type'] == "Options Portfolio" and options_params:
        # Calculate options VaR
        if config['var_model'] == "Historical Simulation" and returns is not None:
            var_result = options_var.calculate_options_var_historical(
                options_params['spot_price'],
                options_params['strike_price'],
                options_params['time_to_expiry'],
                options_params['risk_free_rate'],
                options_params['volatility'],
                options_params['option_type'],
                returns,
                config['confidence_level']
            )
        else:
            # Map VaR model names to options methods
            options_method_map = {
                "Parametric (Delta-Normal)": "Delta-Normal",
                "Historical Simulation": "Delta-Gamma",
                "Monte Carlo": "Full Revaluation Monte Carlo",
                "GARCH": "Delta-Gamma",
                "Extreme Value Theory": "Delta-Gamma"
            }
            
            method = options_method_map.get(config['var_model'], "Delta-Gamma")
            var_result = options_var.calculate_options_var(
                options_params['spot_price'],
                options_params['strike_price'],
                options_params['time_to_expiry'],
                options_params['risk_free_rate'],
                options_params['volatility'],
                options_params['option_type'],
                method,
                config['confidence_level']
            )
        
        # Calculate Expected Shortfall for options (approximation)
        var_value = var_result.get('var', 0)
        expected_shortfall = var_value * 1.2  # Simple approximation
        
        return {
            'var': var_value,
            'expected_shortfall': expected_shortfall,
            'method': var_result.get('method', 'Options VaR'),
            'option_details': var_result
        }
    
    elif returns is not None and len(returns) > 0:
        # Calculate regular portfolio VaR
        if config['var_model'] == "Parametric (Delta-Normal)":
            var_value = var_engines.calculate_parametric_var(returns, config['confidence_level'], config['time_horizon'])
        elif config['var_model'] == "Historical Simulation":
            var_value = var_engines.calculate_historical_var(returns, config['confidence_level'], config['time_horizon'])
        elif config['var_model'] == "Monte Carlo":
            var_value = var_engines.calculate_monte_carlo_var(returns, config['confidence_level'], config['time_horizon'])
        elif config['var_model'] == "GARCH":
            var_value = var_engines.calculate_garch_var(returns, config['confidence_level'], config['time_horizon'])
        elif config['var_model'] == "Extreme Value Theory":
            var_value = var_engines.calculate_evt_var(returns, config['confidence_level'])
        else:
            var_value = 0
        
        # Calculate Expected Shortfall
        expected_shortfall = var_engines.calculate_expected_shortfall(returns, config['confidence_level'])
        
        return {
            'var': var_value,
            'expected_shortfall': expected_shortfall,
            'method': config['var_model']
        }
    
    return {'var': 0, 'expected_shortfall': 0, 'method': 'No data'}

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Create instances
    instances = create_instances()
    
    # Header
    st.markdown('<div class="main-header">📊 VaR & Risk Analytics Platform</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    config = sidebar_configuration()
    
    # Load data
    data, returns, weights_or_options = load_data_section(config, instances)
    
    # Main content area
    if config['portfolio_type'] == "Options Portfolio":
        options_params = weights_or_options
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", 
            "📈 Data Overview", 
            "🔄 Backtesting", 
            "⚡ Stress Testing", 
            "📊 Rolling Analysis",
            "📁 Export Data"
        ])
        
        with tab1:
            st.markdown('<div class="sub-header">Options Portfolio Dashboard</div>', unsafe_allow_html=True)
            
            if options_params:
                # Calculate options VaR
                var_results = calculate_var_metrics(config, instances, returns, options_params)
                st.session_state.var_results = var_results
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Option VaR",
                        value=f"${var_results['var']:.2f}",
                        delta=f"{config['confidence_level']*100:.0f}% confidence"
                    )
                
                with col2:
                    st.metric(
                        label="Expected Shortfall",
                        value=f"${var_results['expected_shortfall']:.2f}",
                        delta="Tail risk"
                    )
                
                with col3:
                    if 'option_details' in var_results and 'current_price' in var_results['option_details']:
                        st.metric(
                            label="Option Price",
                            value=f"${var_results['option_details']['current_price']:.2f}",
                            delta="Theoretical"
                        )
                
                with col4:
                    if 'option_details' in var_results and 'delta' in var_results['option_details']:
                        st.metric(
                            label="Delta",
                            value=f"{var_results['option_details']['delta']:.3f}",
                            delta="Price sensitivity"
                        )
                
                # Option details
                st.subheader("Option Details")
                option_details_df = pd.DataFrame([
                    ["Spot Price", f"${options_params['spot_price']:.2f}"],
                    ["Strike Price", f"${options_params['strike_price']:.2f}"],
                    ["Time to Expiry", f"{options_params['time_to_expiry']:.3f} years"],
                    ["Risk-Free Rate", f"{options_params['risk_free_rate']*100:.2f}%"],
                    ["Volatility", f"{options_params['volatility']*100:.2f}%"],
                    ["Option Type", options_params['option_type'].title()]
                ], columns=["Parameter", "Value"])
                
                st.dataframe(option_details_df, use_container_width=True, hide_index=True)
                
                # Greeks (if available)
                if 'option_details' in var_results:
                    option_data = var_results['option_details']
                    if any(greek in option_data for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']):
                        st.subheader("Option Greeks")
                        greeks_data = []
                        
                        greek_names = ['delta', 'gamma', 'theta', 'vega', 'rho']
                        greek_descriptions = ['Price sensitivity', 'Delta sensitivity', 'Time decay', 'Volatility sensitivity', 'Interest rate sensitivity']
                        
                        for greek, desc in zip(greek_names, greek_descriptions):
                            if greek in option_data:
                                greeks_data.append([greek.title(), f"{option_data[greek]:.4f}", desc])
                        
                        if greeks_data:
                            greeks_df = pd.DataFrame(greeks_data, columns=["Greek", "Value", "Description"])
                            st.dataframe(greeks_df, use_container_width=True, hide_index=True)
                
                # VaR Method Information
                st.subheader("VaR Method Information")
                method_info = {
                    "Delta-Normal": "Linear approximation using option delta",
                    "Delta-Gamma": "Second-order approximation including gamma effects",
                    "Full Revaluation Monte Carlo": "Complete option repricing for each scenario",
                    "Historical Simulation": "Uses historical underlying price movements"
                }
                
                current_method = var_results.get('method', 'Unknown')
                if current_method in method_info:
                    st.info(f"**{current_method}**: {method_info[current_method]}")
                
            else:
                st.warning("Please configure and load options data from the sidebar.")
        
        with tab2:
            st.markdown('<div class="sub-header">Options Data Overview</div>', unsafe_allow_html=True)
            
            if options_params:
                # Option payoff diagram
                if st.button("Generate Option Payoff Diagram", key="payoff_diagram"):
                    spot_range = np.linspace(
                        options_params['spot_price'] * 0.7,
                        options_params['spot_price'] * 1.3,
                        100
                    )
                    
                    fig_payoff = instances['visualization'].plot_options_payoff(
                        spot_range,
                        options_params['strike_price'],
                        options_params['option_type'],
                        options_params['time_to_expiry'],
                        options_params['risk_free_rate'],
                        options_params['volatility']
                    )
                    st.plotly_chart(fig_payoff, use_container_width=True, key="options_payoff_chart")
                
                # Underlying data (if available)
                if returns is not None and len(returns) > 0:
                    st.subheader("Underlying Asset Returns")
                    st.line_chart(returns)
                    
                    # Basic statistics
                    stats_data = [
                        ["Mean Return (Daily)", f"{returns.mean():.4f}"],
                        ["Volatility (Daily)", f"{returns.std():.4f}"],
                        ["Skewness", f"{returns.skew():.4f}"],
                        ["Kurtosis", f"{returns.kurtosis():.4f}"],
                        ["Min Return", f"{returns.min():.4f}"],
                        ["Max Return", f"{returns.max():.4f}"]
                    ]
                    
                    stats_df = pd.DataFrame(stats_data, columns=["Statistic", "Value"])
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No options data available. Please configure options in the sidebar.")
        
        with tab3:
            st.markdown('<div class="sub-header">Options Backtesting</div>', unsafe_allow_html=True)
            
            if options_params and returns is not None:
                # Backtesting parameters
                col1, col2 = st.columns(2)
                with col1:
                    backtest_window = st.slider("Backtesting Window", 30, 250, 60, key="options_backtest_window")
                with col2:
                    backtest_confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key="options_backtest_confidence")
                
                if st.button("Run Options Backtesting", key="run_options_backtest"):
                    with st.spinner("Running options backtesting..."):
                        # Create a simple VaR method function for backtesting
                        def options_var_method(hist_returns, conf_level, time_hor):
                            return instances['options_var'].calculate_options_var_historical(
                                options_params['spot_price'],
                                options_params['strike_price'],
                                options_params['time_to_expiry'],
                                options_params['risk_free_rate'],
                                options_params['volatility'],
                                options_params['option_type'],
                                hist_returns,
                                conf_level
                            ).get('var', 0)
                        
                        backtest_results = instances['backtesting'].perform_backtesting(
                            returns, backtest_confidence, backtest_window, 
                            options_var_method, "options", options_params
                        )
                        
                        if backtest_results:
                            # Display results
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Violations", backtest_results.get('violations', 0))
                            with col2:
                                st.metric("Expected Violations", f"{backtest_results.get('expected_violations', 0):.1f}")
                            with col3:
                                st.metric("Violation Rate", f"{backtest_results.get('violation_rate', 0)*100:.2f}%")
                            with col4:
                                kupiec_p = backtest_results.get('kupiec_pvalue', 0)
                                st.metric("Kupiec Test p-value", f"{kupiec_p:.4f}")
                            
                            # Test results interpretation
                            if kupiec_p > 0.05:
                                st.success("✅ Model passes Kupiec test (p > 0.05)")
                            else:
                                st.error("❌ Model fails Kupiec test (p ≤ 0.05)")
                            
                            # Basel Traffic Light
                            traffic_light = instances['backtesting'].basel_traffic_light(
                                backtest_results.get('violations', 0),
                                backtest_results.get('expected_violations', 0)
                            )
                            
                            if traffic_light == "Green":
                                st.success(f"🟢 Basel Traffic Light: {traffic_light}")
                            elif traffic_light == "Yellow":
                                st.warning(f"🟡 Basel Traffic Light: {traffic_light}")
                            else:
                                st.error(f"🔴 Basel Traffic Light: {traffic_light}")
            else:
                st.warning("Options data and underlying returns required for backtesting.")
        
        with tab4:
            st.markdown('<div class="sub-header">Options Stress Testing</div>', unsafe_allow_html=True)
            
            if options_params:
                # Stress testing options
                stress_type = st.selectbox(
                    "Stress Test Type",
                    ["Historical Scenarios", "Custom Stress"],
                    key="options_stress_type"
                )
                
                if stress_type == "Historical Scenarios":
                    scenario = st.selectbox(
                        "Select Scenario",
                        ["2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash"],
                        key="options_scenario"
                    )
                    
                    if st.button("Run Historical Stress Test", key="run_options_historical_stress"):
                        with st.spinner("Running stress test..."):
                            stress_results = instances['stress_testing'].run_stress_test(
                                returns, scenario, config['confidence_level'],
                                portfolio_type="options", options_data=options_params
                            )
                            
                            if stress_results:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Baseline VaR", f"${stress_results.get('baseline_var', 0):.2f}")
                                with col2:
                                    st.metric("Stressed VaR", f"${stress_results.get('stressed_var', 0):.2f}")
                                with col3:
                                    st.metric("VaR Increase", f"{stress_results.get('var_increase', 0):.1f}%")
                
                else:  # Custom Stress
                    col1, col2 = st.columns(2)
                    with col1:
                        vol_shock = st.slider("Volatility Shock (%)", -50, 200, 50, key="options_vol_shock")
                    with col2:
                        market_shock = st.slider("Market Shock (%)", -50, 50, -20, key="options_market_shock")
                    
                    if st.button("Run Custom Stress Test", key="run_options_custom_stress"):
                        with st.spinner("Running custom stress test..."):
                            stress_results = instances['stress_testing'].run_custom_stress_test(
                                returns, vol_shock, 0, market_shock, config['confidence_level'],
                                portfolio_type="options", options_data=options_params
                            )
                            
                            if stress_results:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Baseline VaR", f"${stress_results.get('baseline_var', 0):.2f}")
                                with col2:
                                    st.metric("Stressed VaR", f"${stress_results.get('stressed_var', 0):.2f}")
                                with col3:
                                    st.metric("VaR Increase", f"{stress_results.get('var_increase', 0):.1f}%")
            else:
                st.warning("Options data required for stress testing.")
        
        with tab5:
            st.markdown('<div class="sub-header">Options Rolling Analysis</div>', unsafe_allow_html=True)
            
            if returns is not None and len(returns) > 0:
                # Rolling analysis parameters
                rolling_window = st.slider("Rolling Window", 20, 100, 30, key="options_rolling_window")
                
                if st.button("Calculate Rolling Metrics", key="calc_options_rolling"):
                    with st.spinner("Calculating rolling metrics..."):
                        # Calculate rolling volatility of underlying
                        rolling_vol = instances['rolling_analysis'].calculate_rolling_volatility(returns, rolling_window)
                        
                        if not rolling_vol.empty:
                            st.subheader("Rolling Volatility of Underlying")
                            st.line_chart(rolling_vol)
                            
                            # Rolling VaR (simplified for underlying)
                            rolling_var = instances['rolling_analysis'].calculate_rolling_var(returns, config['confidence_level'], rolling_window)
                            
                            if not rolling_var.empty:
                                st.subheader("Rolling VaR (Underlying)")
                                st.line_chart(rolling_var)
            else:
                st.warning("Underlying returns data required for rolling analysis.")
        
        with tab6:
            st.markdown('<div class="sub-header">Export Options Data</div>', unsafe_allow_html=True)
            
            if options_params and 'var_results' in st.session_state:
                # Prepare export data
                export_data = {}
                
                # Option parameters
                export_data['Option_Parameters'] = pd.DataFrame([
                    ["Spot Price", options_params['spot_price']],
                    ["Strike Price", options_params['strike_price']],
                    ["Time to Expiry", options_params['time_to_expiry']],
                    ["Risk-Free Rate", options_params['risk_free_rate']],
                    ["Volatility", options_params['volatility']],
                    ["Option Type", options_params['option_type']]
                ], columns=["Parameter", "Value"])
                
                # VaR results
                var_results = st.session_state.var_results
                export_data['VaR_Results'] = pd.DataFrame([
                    ["VaR", var_results.get('var', 0)],
                    ["Expected Shortfall", var_results.get('expected_shortfall', 0)],
                    ["Method", var_results.get('method', 'N/A')]
                ], columns=["Metric", "Value"])
                
                # Greeks (if available)
                if 'option_details' in var_results:
                    option_details = var_results['option_details']
                    greeks_data = []
                    for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                        if greek in option_details:
                            greeks_data.append([greek.title(), option_details[greek]])
                    
                    if greeks_data:
                        export_data['Greeks'] = pd.DataFrame(greeks_data, columns=["Greek", "Value"])
                
                # Underlying returns (if available)
                if returns is not None:
                    export_data['Underlying_Returns'] = returns.to_frame('Returns')
                
                # Export options
                st.subheader("Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Export to CSV", key="export_options_csv"):
                        csv_data = instances['utils'].export_to_csv(export_data['Option_Parameters'], "options_data")
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"options_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("Export to Excel", key="export_options_excel"):
                        excel_data = instances['utils'].export_to_excel(export_data, "options_analysis")
                        if excel_data:
                            st.download_button(
                                label="Download Excel",
                                data=excel_data,
                                file_name=f"options_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                
                # Display export preview
                st.subheader("Export Preview")
                for sheet_name, df in export_data.items():
                    with st.expander(f"Preview: {sheet_name}"):
                        st.dataframe(df, use_container_width=True)
            else:
                st.warning("No options data available for export.")
    
    else:
        # Regular portfolio tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", 
            "📈 Data Overview", 
            "🔄 Backtesting", 
            "⚡ Stress Testing", 
            "📊 Rolling Analysis",
            "📁 Export Data"
        ])
        
        with tab1:
            st.markdown('<div class="sub-header">Portfolio Risk Dashboard</div>', unsafe_allow_html=True)
            
            if returns is not None and len(returns) > 0:
                # Calculate VaR metrics
                var_results = calculate_var_metrics(config, instances, returns)
                st.session_state.var_results = var_results
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label=f"VaR ({config['confidence_level']*100:.0f}%)",
                        value=f"${var_results['var']:,.2f}",
                        delta=f"{config['time_horizon']} day"
                    )
                
                with col2:
                    st.metric(
                        label="Expected Shortfall",
                        value=f"${var_results['expected_shortfall']:,.2f}",
                        delta="Tail risk"
                    )
                
                with col3:
                    current_vol = returns.std() * np.sqrt(252) * 100
                    st.metric(
                        label="Annualized Volatility",
                        value=f"{current_vol:.2f}%",
                        delta="Historical"
                    )
                
                with col4:
                    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                    st.metric(
                        label="Sharpe Ratio",
                        value=f"{sharpe_ratio:.3f}",
                        delta="Risk-adjusted"
                    )
                
                # VaR Distribution Plot
                if st.button("Show VaR Distribution", key="show_var_dist"):
                    fig_dist = instances['visualization'].plot_var_distribution(
                        returns, config['confidence_level'], var_results['var']
                    )
                    st.plotly_chart(fig_dist, use_container_width=True, key="var_distribution_chart")
                
                # Portfolio Statistics
                st.subheader("Portfolio Statistics")
                stats = instances['utils'].calculate_portfolio_statistics(returns)
                
                stats_data = [
                    ["Mean Daily Return", f"{stats.get('mean_return', 0):.4f}"],
                    ["Daily Volatility", f"{stats.get('std_return', 0):.4f}"],
                    ["Annual Return", f"{stats.get('annual_return', 0)*100:.2f}%"],
                    ["Annual Volatility", f"{stats.get('annual_volatility', 0)*100:.2f}%"],
                    ["Skewness", f"{stats.get('skewness', 0):.4f}"],
                    ["Kurtosis", f"{stats.get('kurtosis', 0):.4f}"],
                    ["Max Drawdown", f"{stats.get('max_drawdown', 0)*100:.2f}%"],
                    ["Sortino Ratio", f"{stats.get('sortino_ratio', 0):.3f}"]
                ]
                
                stats_df = pd.DataFrame(stats_data, columns=["Metric", "Value"])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Model Information
                st.subheader("Model Information")
                model_info = {
                    "Parametric (Delta-Normal)": "Assumes normal distribution of returns",
                    "Historical Simulation": "Uses actual historical return distribution",
                    "Monte Carlo": "Simulates future returns based on historical parameters",
                    "GARCH": "Models time-varying volatility",
                    "Extreme Value Theory": "Focuses on tail risk and extreme events"
                }
                
                current_model = config['var_model']
                if current_model in model_info:
                    st.info(f"**{current_model}**: {model_info[current_model]}")
                
            else:
                st.warning("Please load data from the sidebar to view the dashboard.")
        
        with tab2:
            st.markdown('<div class="sub-header">Data Overview</div>', unsafe_allow_html=True)
            
            if data is not None and not data.empty:
                # Data summary
                st.subheader("Data Summary")
                
                summary_data = [
                    ["Data Points", len(data)],
                    ["Date Range", f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}"],
                    ["Assets", len(data.columns)],
                    ["Portfolio Type", config['portfolio_type']]
                ]
                
                summary_df = pd.DataFrame(summary_data, columns=["Attribute", "Value"])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Price chart
                st.subheader("Price Chart")
                st.line_chart(data)
                
                # Returns chart
                if returns is not None:
                    st.subheader("Returns Chart")
                    st.line_chart(returns)
                
                # Data table
                st.subheader("Raw Data")
                st.dataframe(data.tail(20), use_container_width=True)
                
                # Correlation matrix (for multi-asset portfolios)
                if len(data.columns) > 1:
                    st.subheader("Correlation Matrix")
                    corr_matrix = data.pct_change().corr()
                    fig_corr = instances['visualization'].plot_correlation_heatmap(corr_matrix)
                    st.plotly_chart(fig_corr, use_container_width=True, key="correlation_heatmap")
            else:
                st.warning("No data available. Please load data from the sidebar.")
        
        with tab3:
            st.markdown('<div class="sub-header">VaR Model Backtesting</div>', unsafe_allow_html=True)
            
            if returns is not None and len(returns) > 0:
                # Backtesting parameters
                col1, col2 = st.columns(2)
                with col1:
                    backtest_window = st.slider("Backtesting Window", 30, 250, 60, key="backtest_window")
                with col2:
                    backtest_confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key="backtest_confidence")
                
                if st.button("Run Backtesting", key="run_backtest"):
                    with st.spinner("Running backtesting..."):
                        # Create VaR method function
                        var_engines = instances['var_engines']
                        
                        def var_method_func(hist_returns, conf_level, time_hor):
                            if config['var_model'] == "Parametric (Delta-Normal)":
                                return var_engines.calculate_parametric_var(hist_returns, conf_level, time_hor)
                            elif config['var_model'] == "Historical Simulation":
                                return var_engines.calculate_historical_var(hist_returns, conf_level, time_hor)
                            elif config['var_model'] == "Monte Carlo":
                                return var_engines.calculate_monte_carlo_var(hist_returns, conf_level, time_hor)
                            elif config['var_model'] == "GARCH":
                                return var_engines.calculate_garch_var(hist_returns, conf_level, time_hor)
                            elif config['var_model'] == "Extreme Value Theory":
                                return var_engines.calculate_evt_var(hist_returns, conf_level)
                            else:
                                return var_engines.calculate_parametric_var(hist_returns, conf_level, time_hor)
                        
                        backtest_results = instances['backtesting'].perform_backtesting(
                            returns, backtest_confidence, backtest_window, var_method_func
                        )
                        
                        if backtest_results:
                            # Display results
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Violations", backtest_results.get('violations', 0))
                            with col2:
                                st.metric("Expected Violations", f"{backtest_results.get('expected_violations', 0):.1f}")
                            with col3:
                                st.metric("Violation Rate", f"{backtest_results.get('violation_rate', 0)*100:.2f}%")
                            with col4:
                                kupiec_p = backtest_results.get('kupiec_pvalue', 0)
                                st.metric("Kupiec Test p-value", f"{kupiec_p:.4f}")
                            
                            # Test results interpretation
                            if kupiec_p > 0.05:
                                st.success("✅ Model passes Kupiec test (p > 0.05)")
                            else:
                                st.error("❌ Model fails Kupiec test (p ≤ 0.05)")
                            
                            # Basel Traffic Light
                            traffic_light = instances['backtesting'].basel_traffic_light(
                                backtest_results.get('violations', 0),
                                backtest_results.get('expected_violations', 0)
                            )
                            
                            if traffic_light == "Green":
                                st.success(f"🟢 Basel Traffic Light: {traffic_light}")
                            elif traffic_light == "Yellow":
                                st.warning(f"🟡 Basel Traffic Light: {traffic_light}")
                            else:
                                st.error(f"🔴 Basel Traffic Light: {traffic_light}")
                            
                            # Violations plot
                            if 'var_estimates' in backtest_results and 'violations_dates' in backtest_results:
                                fig_violations = instances['visualization'].plot_var_violations(
                                    returns, 
                                    backtest_results['var_estimates'],
                                    backtest_results['violations_dates']
                                )
                                st.plotly_chart(fig_violations, use_container_width=True, key="violations_chart")
            else:
                st.warning("Returns data required for backtesting.")
        
        with tab4:
            st.markdown('<div class="sub-header">Stress Testing</div>', unsafe_allow_html=True)
            
            if returns is not None and len(returns) > 0:
                # Stress testing options
                stress_type = st.selectbox(
                    "Stress Test Type",
                    ["Historical Scenarios", "Custom Stress"],
                    key="stress_type"
                )
                
                if stress_type == "Historical Scenarios":
                    scenario = st.selectbox(
                        "Select Scenario",
                        ["2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash"],
                        key="scenario"
                    )
                    
                    if st.button("Run Historical Stress Test", key="run_historical_stress"):
                        with st.spinner("Running stress test..."):
                            stress_results = instances['stress_testing'].run_stress_test(
                                returns, scenario, config['confidence_level']
                            )
                            
                            if stress_results:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Baseline VaR", f"${stress_results.get('baseline_var', 0):,.2f}")
                                with col2:
                                    st.metric("Stressed VaR", f"${stress_results.get('stressed_var', 0):,.2f}")
                                with col3:
                                    st.metric("VaR Increase", f"{stress_results.get('var_increase', 0):.1f}%")
                                
                                st.subheader("Scenario Description")
                                st.info(f"**{scenario}**: {stress_results.get('scenario_description', 'N/A')}")
                
                else:  # Custom Stress
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        vol_shock = st.slider("Volatility Shock (%)", -50, 200, 50, key="vol_shock")
                    with col2:
                        corr_shock = st.slider("Correlation Shock", 0.0, 1.0, 0.3, key="corr_shock")
                    with col3:
                        market_shock = st.slider("Market Shock (%)", -50, 50, -20, key="market_shock")
                    
                    if st.button("Run Custom Stress Test", key="run_custom_stress"):
                        with st.spinner("Running custom stress test..."):
                            stress_results = instances['stress_testing'].run_custom_stress_test(
                                returns, vol_shock, corr_shock, market_shock, config['confidence_level']
                            )
                            
                            if stress_results:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Baseline VaR", f"${stress_results.get('baseline_var', 0):,.2f}")
                                with col2:
                                    st.metric("Stressed VaR", f"${stress_results.get('stressed_var', 0):,.2f}")
                                with col3:
                                    st.metric("VaR Increase", f"{stress_results.get('var_increase', 0):.1f}%")
                
                # Sensitivity analysis
                st.subheader("Sensitivity Analysis")
                if st.button("Run Sensitivity Analysis", key="run_sensitivity"):
                    with st.spinner("Running sensitivity analysis..."):
                        sensitivity_data = instances['stress_testing'].sensitivity_analysis(
                            returns, config['confidence_level']
                        )
                        
                        if not sensitivity_data.empty:
                            fig_sensitivity = instances['visualization'].plot_sensitivity_analysis(sensitivity_data)
                            st.plotly_chart(fig_sensitivity, use_container_width=True, key="sensitivity_chart")
            else:
                st.warning("Returns data required for stress testing.")
        
        with tab5:
            st.markdown('<div class="sub-header">Rolling Risk Analysis</div>', unsafe_allow_html=True)
            
            if returns is not None and len(returns) > 0:
                # Rolling analysis parameters
                rolling_window = st.slider("Rolling Window", 20, 100, 30, key="rolling_window")
                
                if st.button("Calculate Rolling Metrics", key="calc_rolling"):
                    with st.spinner("Calculating rolling metrics..."):
                        # Rolling VaR
                        rolling_var = instances['rolling_analysis'].calculate_rolling_var(
                            returns, config['confidence_level'], rolling_window
                        )
                        
                        if not rolling_var.empty:
                            st.subheader("Rolling VaR")
                            st.line_chart(rolling_var)
                        
                        # Rolling volatility
                        rolling_vol = instances['rolling_analysis'].calculate_rolling_volatility(
                            returns, rolling_window
                        )
                        
                        if not rolling_vol.empty:
                            st.subheader("Rolling Volatility")
                            st.line_chart(rolling_vol)
                        
                        # Rolling Sharpe ratio
                        rolling_sharpe = instances['rolling_analysis'].calculate_rolling_sharpe(
                            returns, rolling_window
                        )
                        
                        if not rolling_sharpe.empty:
                            st.subheader("Rolling Sharpe Ratio")
                            st.line_chart(rolling_sharpe)
                        
                        # Drawdown analysis
                        drawdown = instances['rolling_analysis'].calculate_drawdown(returns)
                        
                        if not drawdown.empty:
                            st.subheader("Drawdown")
                            fig_drawdown = instances['visualization'].plot_drawdown(drawdown)
                            st.plotly_chart(fig_drawdown, use_container_width=True, key="drawdown_chart")
                        
                        # Maximum drawdown metrics
                        max_dd_info = instances['rolling_analysis'].calculate_maximum_drawdown(returns)
                        if max_dd_info:
                            st.subheader("Maximum Drawdown Analysis")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Max Drawdown", f"{max_dd_info.get('max_drawdown', 0)*100:.2f}%")
                            with col2:
                                max_dd_date = max_dd_info.get('max_drawdown_date')
                                if max_dd_date:
                                    st.metric("Max DD Date", max_dd_date.strftime('%Y-%m-%d'))
                            with col3:
                                peak_date = max_dd_info.get('peak_date')
                                if peak_date:
                                    st.metric("Peak Date", peak_date.strftime('%Y-%m-%d'))
            else:
                st.warning("Returns data required for rolling analysis.")
        
        with tab6:
            st.markdown('<div class="sub-header">Export Data</div>', unsafe_allow_html=True)
            
            if data is not None and returns is not None:
                # Prepare export data
                export_data = {}
                
                # Price data
                export_data['Price_Data'] = data
                
                # Returns data
                export_data['Returns_Data'] = returns.to_frame('Returns') if isinstance(returns, pd.Series) else returns
                
                # Portfolio statistics
                if len(returns) > 0:
                    stats = instances['utils'].calculate_portfolio_statistics(returns)
                    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
                    export_data['Portfolio_Statistics'] = stats_df
                
                # VaR results
                if 'var_results' in st.session_state:
                    var_results = st.session_state.var_results
                    var_df = pd.DataFrame([
                        ['VaR', var_results.get('var', 0)],
                        ['Expected Shortfall', var_results.get('expected_shortfall', 0)],
                        ['Method', var_results.get('method', 'N/A')],
                        ['Confidence Level', config['confidence_level']],
                        ['Time Horizon', config['time_horizon']]
                    ], columns=['Metric', 'Value'])
                    export_data['VaR_Results'] = var_df
                
                # Configuration
                config_df = pd.DataFrame([
                    ['Portfolio Type', config['portfolio_type']],
                    ['Data Source', config['data_source']],
                    ['VaR Model', config['var_model']],
                    ['Confidence Level', config['confidence_level']],
                    ['Time Horizon', config['time_horizon']],
                    ['Historical Window', config['historical_window']]
                ], columns=['Parameter', 'Value'])
                export_data['Configuration'] = config_df
                
                # Export options
                st.subheader("Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Export to CSV", key="export_csv"):
                        # Export main data as CSV
                        csv_data = instances['utils'].export_to_csv(data, "portfolio_data")
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("Export to Excel", key="export_excel"):
                        excel_data = instances['utils'].export_to_excel(export_data, "portfolio_analysis")
                        if excel_data:
                            st.download_button(
                                label="Download Excel",
                                data=excel_data,
                                file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                
                # Display export preview
                st.subheader("Export Preview")
                for sheet_name, df in export_data.items():
                    with st.expander(f"Preview: {sheet_name}"):
                        st.dataframe(df, use_container_width=True)
            else:
                st.warning("No data available for export. Please load data first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #888; padding: 20px;">
            <p>🔬 <strong>VaR & Risk Analytics Platform</strong> | Built with Streamlit | © 2024</p>
            <p><em>Professional risk management tools for modern financial markets</em></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()