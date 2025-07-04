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
                    st.success(f"Options data loaded for {symbol}")
                else:
                    st.error("Failed to fetch options data")
                    return None, None, None
        
        # Options selection
        if st.session_state.options_data:
            options_data = st.session_state.options_data
            
            # Expiry selection
            expiry_dates = options_data['expiry_dates']
            selected_expiry = st.sidebar.selectbox("Expiry Date", expiry_dates)
            
            # Option type
            option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
            
            # Strike selection
            if selected_expiry in options_data['options_chains']:
                chain = options_data['options_chains'][selected_expiry]
                strikes = chain['calls']['strike'].tolist() if option_type == 'call' else chain['puts']['strike'].tolist()
                
                # Find closest ATM strike
                current_price = options_data['current_price']
                closest_strike = min(strikes, key=lambda x: abs(x - current_price))
                closest_index = strikes.index(closest_strike)
                
                selected_strike = st.sidebar.selectbox(
                    "Strike Price", 
                    strikes, 
                    index=closest_index,
                    format_func=lambda x: f"${x:.2f}"
                )
                
                # Get option details
                option_chain = chain['calls'] if option_type == 'call' else chain['puts']
                option_row = option_chain[option_chain['strike'] == selected_strike].iloc[0]
                
                # Calculate time to expiry
                expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
                time_to_expiry = (expiry_date - datetime.now()).days / 365.0
                
                options_params = {
                    'spot_price': current_price,
                    'strike_price': selected_strike,
                    'time_to_expiry': max(time_to_expiry, 0.01),
                    'risk_free_rate': 0.05,
                    'volatility': option_row['impliedVolatility'],
                    'option_type': option_type,
                    'market_price': option_row['lastPrice']
                }
                
                # Load underlying data for historical simulation
                end_date = datetime.now()
                start_date = end_date - timedelta(days=config['historical_window'])
                
                underlying_data = instances['data_ingestion'].load_live_data(
                    [symbol], start_date, end_date
                )
                
                if underlying_data is not None:
                    underlying_returns = underlying_data.pct_change().dropna()
                    return underlying_returns, [1.0], options_params
                
    elif config['data_source'] == "Manual Entry":
        st.sidebar.markdown("### Manual Options Entry")
        
        spot_price = st.sidebar.number_input("Spot Price", value=100.0, min_value=0.01)
        strike_price = st.sidebar.number_input("Strike Price", value=105.0, min_value=0.01)
        time_to_expiry = st.sidebar.number_input("Time to Expiry (years)", value=0.25, min_value=0.01, max_value=5.0)
        risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=1.0)
        volatility = st.sidebar.number_input("Volatility", value=0.25, min_value=0.01, max_value=5.0)
        option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
        
        options_params = {
            'spot_price': spot_price,
            'strike_price': strike_price,
            'time_to_expiry': time_to_expiry,
            'risk_free_rate': risk_free_rate,
            'volatility': volatility,
            'option_type': option_type
        }
        
        # Generate synthetic underlying data
        synthetic_data = instances['data_ingestion'].generate_synthetic_data(
            num_days=config['historical_window'],
            initial_price=spot_price,
            annual_return=0.08,
            annual_volatility=volatility
        )
        
        if synthetic_data is not None:
            synthetic_returns = synthetic_data.pct_change().dropna()
            return synthetic_returns, [1.0], options_params
    
    return None, None, None

def load_live_data(config, data_ingestion):
    """Load live market data"""
    st.sidebar.markdown("### Live Data Configuration")
    
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
        st.sidebar.markdown("**Portfolio Weights**")
        weights = []
        for symbol in symbols:
            weight = st.sidebar.number_input(
                f"{symbol} Weight", 
                value=1.0/len(symbols), 
                min_value=0.0, 
                max_value=1.0,
                key=f"weight_{symbol}"
            )
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        
    elif config['portfolio_type'] == "Crypto Portfolio":
        crypto_symbols = st.sidebar.multiselect(
            "Crypto Symbols",
            ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"],
            default=["BTC-USD"],
            help="Select cryptocurrency symbols"
        )
        symbols = crypto_symbols
        weights = [1.0/len(symbols) for _ in symbols] if symbols else [1.0]
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config['historical_window'])
    
    # Load data
    if st.sidebar.button("Load Data"):
        with st.spinner("Loading market data..."):
            data = data_ingestion.load_live_data(symbols, start_date, end_date)
            
            if data is not None:
                returns = data.pct_change().dropna()
                st.session_state.data_loaded = True
                st.session_state.current_data = data
                st.session_state.current_returns = returns
                st.session_state.portfolio_weights = weights
                st.success("Data loaded successfully!")
                return returns, weights, None
            else:
                st.error("Failed to load data")
    
    return None, None, None

def load_file_data(data_ingestion):
    """Load data from uploaded file"""
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV/Excel file",
        type=['csv', 'xlsx'],
        help="Upload a file with Date column and price columns"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = data_ingestion.load_csv_data(uploaded_file)
            else:
                # Handle Excel files
                df = pd.read_excel(uploaded_file)
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df.set_index(df.columns[0], inplace=True)
                data_ingestion.data = df
                data_ingestion.returns = df.pct_change().dropna()
                data = df
            
            if data is not None:
                returns = data.pct_change().dropna()
                weights = [1.0/len(data.columns) for _ in data.columns]
                
                st.session_state.data_loaded = True
                st.session_state.current_data = data
                st.session_state.current_returns = returns
                st.session_state.portfolio_weights = weights
                st.success("File uploaded successfully!")
                return returns, weights, None
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    return None, None, None

def load_manual_data(data_ingestion):
    """Load manually entered data"""
    st.sidebar.markdown("### Manual Data Entry")
    
    num_assets = st.sidebar.number_input("Number of Assets", min_value=1, max_value=10, value=1)
    num_days = st.sidebar.number_input("Number of Days", min_value=30, max_value=1000, value=252)
    
    if st.sidebar.button("Generate Manual Data"):
        # For simplicity, generate synthetic data with user parameters
        data = data_ingestion.generate_synthetic_data(
            num_days=num_days,
            initial_price=100,
            annual_return=0.08,
            annual_volatility=0.20
        )
        
        if data is not None:
            returns = data.pct_change().dropna()
            weights = [1.0]
            
            st.session_state.data_loaded = True
            st.session_state.current_data = data
            st.session_state.current_returns = returns
            st.session_state.portfolio_weights = weights
            st.success("Manual data generated!")
            return returns, weights, None
    
    return None, None, None

def load_synthetic_data(data_ingestion):
    """Load synthetic data"""
    st.sidebar.markdown("### Synthetic Data Parameters")
    
    num_days = st.sidebar.slider("Number of Days", 100, 1000, 500)
    initial_price = st.sidebar.number_input("Initial Price", value=100.0)
    annual_return = st.sidebar.slider("Annual Return", -0.5, 0.5, 0.08, 0.01)
    annual_volatility = st.sidebar.slider("Annual Volatility", 0.1, 1.0, 0.20, 0.01)
    
    if st.sidebar.button("Generate Synthetic Data"):
        data = data_ingestion.generate_synthetic_data(
            num_days=num_days,
            initial_price=initial_price,
            annual_return=annual_return,
            annual_volatility=annual_volatility
        )
        
        if data is not None:
            returns = data.pct_change().dropna()
            weights = [1.0]
            
            st.session_state.data_loaded = True
            st.session_state.current_data = data
            st.session_state.current_returns = returns
            st.session_state.portfolio_weights = weights
            st.success("Synthetic data generated!")
            return returns, weights, None
    
    return None, None, None

def calculate_var(returns, weights, config, instances, options_params=None):
    """Calculate VaR based on selected model"""
    var_engines = instances['var_engines']
    options_var = instances['options_var']
    
    if options_params is not None:
        # Options VaR calculation
        if config['var_model'] == "Historical Simulation":
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
            # Map VaR model to options method
            options_method_map = {
                "Parametric (Delta-Normal)": "Delta-Normal",
                "Monte Carlo": "Full Revaluation Monte Carlo",
                "GARCH": "Delta-Gamma",
                "Extreme Value Theory": "Delta-Gamma"
            }
            
            options_method = options_method_map.get(config['var_model'], "Delta-Gamma")
            
            var_result = options_var.calculate_options_var(
                options_params['spot_price'],
                options_params['strike_price'],
                options_params['time_to_expiry'],
                options_params['risk_free_rate'],
                options_params['volatility'],
                options_params['option_type'],
                options_method,
                config['confidence_level']
            )
        
        return var_result.get('var', 0), var_result
    
    else:
        # Regular portfolio VaR calculation
        if len(returns.shape) > 1:
            portfolio_returns = returns.dot(weights)
        else:
            portfolio_returns = returns
        
        if config['var_model'] == "Parametric (Delta-Normal)":
            var_value = var_engines.calculate_parametric_var(
                portfolio_returns, config['confidence_level'], config['time_horizon']
            )
        elif config['var_model'] == "Historical Simulation":
            var_value = var_engines.calculate_historical_var(
                portfolio_returns, config['confidence_level'], config['time_horizon']
            )
        elif config['var_model'] == "Monte Carlo":
            var_value = var_engines.calculate_monte_carlo_var(
                portfolio_returns, config['confidence_level'], config['time_horizon']
            )
        elif config['var_model'] == "GARCH":
            var_value = var_engines.calculate_garch_var(
                portfolio_returns, config['confidence_level'], config['time_horizon']
            )
        elif config['var_model'] == "Extreme Value Theory":
            var_value = var_engines.calculate_evt_var(
                portfolio_returns, config['confidence_level']
            )
        else:
            var_value = 0
        
        # Calculate Expected Shortfall
        expected_shortfall = var_engines.calculate_expected_shortfall(
            portfolio_returns, config['confidence_level']
        )
        
        return var_value, {'var': var_value, 'expected_shortfall': expected_shortfall}

def main():
    """Main application function"""
    # Initialize
    initialize_session_state()
    instances = create_instances()
    
    # Header
    st.markdown('<div class="main-header">📊 VaR & Risk Analytics Platform</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    config = sidebar_configuration()
    
    # Load data
    returns, weights, options_params = load_data_section(config, instances)
    
    # Update session state if new data is loaded
    if returns is not None:
        st.session_state.current_returns = returns
        st.session_state.portfolio_weights = weights
        st.session_state.data_loaded = True
        if options_params:
            st.session_state.options_data = options_params
    
    # Main content area
    if st.session_state.data_loaded and st.session_state.current_returns is not None:
        # Calculate VaR
        var_value, var_details = calculate_var(
            st.session_state.current_returns, 
            st.session_state.portfolio_weights, 
            config, 
            instances,
            st.session_state.options_data if config['portfolio_type'] == "Options Portfolio" else None
        )
        
        st.session_state.var_results = var_details
        
        # Create tabs
        if config['portfolio_type'] == "Options Portfolio":
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Dashboard", "📈 Data Overview", "🔄 Backtesting", 
                "⚡ Stress Testing", "📊 Rolling Analysis", "💾 Export Data"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Dashboard", "📈 Data Overview", "🔄 Backtesting", 
                "⚡ Stress Testing", "📊 Rolling Analysis", "💾 Export Data"
            ])
        
        with tab1:
            dashboard_tab(config, instances, var_value, var_details)
        
        with tab2:
            data_overview_tab(config, instances)
        
        with tab3:
            backtesting_tab(config, instances)
        
        with tab4:
            stress_testing_tab(config, instances)
        
        with tab5:
            rolling_analysis_tab(config, instances)
        
        with tab6:
            export_data_tab(config, instances)
    
    else:
        # Welcome message
        st.markdown("""
        <div class="success-message">
            <h3>🚀 Welcome to the VaR & Risk Analytics Platform</h3>
            <p>Configure your portfolio settings in the sidebar and load data to begin risk analysis.</p>
            <ul>
                <li><strong>Portfolio Types:</strong> Single Asset, Multi-Asset, Crypto, Options</li>
                <li><strong>Data Sources:</strong> Live Market, File Upload, Manual Entry, Synthetic</li>
                <li><strong>VaR Models:</strong> Parametric, Historical, Monte Carlo, GARCH, EVT</li>
                <li><strong>Analysis Tools:</strong> Backtesting, Stress Testing, Rolling Analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def dashboard_tab(config, instances, var_value, var_details):
    """Dashboard tab content"""
    if config['portfolio_type'] == "Options Portfolio":
        options_dashboard(config, instances, var_value, var_details)
    else:
        regular_dashboard(config, instances, var_value, var_details)

def regular_dashboard(config, instances, var_value, var_details):
    """Regular portfolio dashboard"""
    st.markdown('<div class="sub-header">📊 Risk Dashboard</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"VaR ({config['confidence_level']*100:.0f}%)",
            value=f"${var_value:,.0f}",
            delta=f"{config['var_model']}"
        )
    
    with col2:
        expected_shortfall = var_details.get('expected_shortfall', 0)
        st.metric(
            label="Expected Shortfall",
            value=f"${expected_shortfall:,.0f}",
            delta="Tail Risk"
        )
    
    with col3:
        if st.session_state.current_returns is not None:
            if len(st.session_state.current_returns.shape) > 1:
                var_portfolio_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
            else:
                var_portfolio_returns = st.session_state.current_returns
            
            current_vol = var_portfolio_returns.std() * np.sqrt(252) * 100
            st.metric(
                label="Annualized Volatility",
                value=f"{current_vol:.2f}%",
                delta="Historical"
            )
    
    with col4:
        if st.session_state.current_returns is not None:
            if len(st.session_state.current_returns.shape) > 1:
                var_portfolio_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
            else:
                var_portfolio_returns = st.session_state.current_returns
            
            sharpe_ratio = (var_portfolio_returns.mean() * 252) / (var_portfolio_returns.std() * np.sqrt(252))
            st.metric(
                label="Sharpe Ratio",
                value=f"{sharpe_ratio:.3f}",
                delta="Risk-Adjusted"
            )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.current_returns is not None:
            if len(st.session_state.current_returns.shape) > 1:
                var_portfolio_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
            else:
                var_portfolio_returns = st.session_state.current_returns
            
            # Performance chart
            cumulative_returns = (1 + var_portfolio_returns).cumprod()
            fig_perf = instances['visualization'].plot_performance_chart(cumulative_returns)
            st.plotly_chart(fig_perf, use_container_width=True, key="dashboard_performance")
    
    with col2:
        if st.session_state.current_returns is not None:
            if len(st.session_state.current_returns.shape) > 1:
                var_portfolio_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
            else:
                var_portfolio_returns = st.session_state.current_returns
            
            # VaR distribution
            fig_dist = instances['visualization'].plot_var_distribution(
                var_portfolio_returns, config['confidence_level'], var_value
            )
            st.plotly_chart(fig_dist, use_container_width=True, key="dashboard_distribution")
    
    # Portfolio Statistics
    st.subheader("Portfolio Statistics")
    
    if st.session_state.current_returns is not None:
        if len(st.session_state.current_returns.shape) > 1:
            var_portfolio_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
        else:
            var_portfolio_returns = st.session_state.current_returns
        
        stats = instances['utils'].calculate_portfolio_statistics(var_portfolio_returns)
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("Annual Return", f"{stats.get('annual_return', 0)*100:.2f}%")
            st.metric("Skewness", f"{stats.get('skewness', 0):.3f}")
        
        with stats_col2:
            st.metric("Annual Volatility", f"{stats.get('annual_volatility', 0)*100:.2f}%")
            st.metric("Kurtosis", f"{stats.get('kurtosis', 0):.3f}")
        
        with stats_col3:
            st.metric("Max Drawdown", f"{stats.get('max_drawdown', 0)*100:.2f}%")
            st.metric("Sortino Ratio", f"{stats.get('sortino_ratio', 0):.3f}")

def options_dashboard(config, instances, var_value, var_details):
    """Options portfolio dashboard"""
    st.markdown('<div class="sub-header">📊 Options Risk Dashboard</div>', unsafe_allow_html=True)
    
    options_data = st.session_state.options_data
    
    if options_data:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=f"Options VaR ({config['confidence_level']*100:.0f}%)",
                value=f"${var_value:,.2f}",
                delta=f"{config['var_model']}"
            )
        
        with col2:
            current_price = var_details.get('current_price', 0)
            st.metric(
                label="Option Price",
                value=f"${current_price:.2f}",
                delta="Current"
            )
        
        with col3:
            delta = var_details.get('delta', 0)
            st.metric(
                label="Delta",
                value=f"{delta:.4f}",
                delta="Price Sensitivity"
            )
        
        with col4:
            gamma = var_details.get('gamma', 0)
            st.metric(
                label="Gamma",
                value=f"{gamma:.6f}",
                delta="Convexity"
            )
        
        # Greeks display
        st.subheader("Option Greeks")
        
        greeks_col1, greeks_col2, greeks_col3 = st.columns(3)
        
        with greeks_col1:
            theta = var_details.get('theta', 0)
            st.metric("Theta", f"{theta:.4f}", "Time Decay")
        
        with greeks_col2:
            vega = var_details.get('vega', 0)
            st.metric("Vega", f"{vega:.4f}", "Vol Sensitivity")
        
        with greeks_col3:
            rho = var_details.get('rho', 0) if 'rho' in var_details else 0
            st.metric("Rho", f"{rho:.4f}", "Rate Sensitivity")
        
        # Option details
        st.subheader("Option Details")
        
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            st.write(f"**Spot Price:** ${options_data['spot_price']:.2f}")
            st.write(f"**Strike Price:** ${options_data['strike_price']:.2f}")
            st.write(f"**Time to Expiry:** {options_data['time_to_expiry']:.4f} years")
        
        with details_col2:
            st.write(f"**Risk-Free Rate:** {options_data['risk_free_rate']*100:.2f}%")
            st.write(f"**Volatility:** {options_data['volatility']*100:.2f}%")
            st.write(f"**Option Type:** {options_data['option_type'].title()}")
        
        # Option payoff diagram
        st.subheader("Option Payoff Diagram")
        
        S_min = options_data['spot_price'] * 0.7
        S_max = options_data['spot_price'] * 1.3
        S_range = np.linspace(S_min, S_max, 100)
        
        fig_payoff = instances['visualization'].plot_options_payoff(
            S_range,
            options_data['strike_price'],
            options_data['option_type'],
            options_data['time_to_expiry'],
            options_data['risk_free_rate'],
            options_data['volatility']
        )
        st.plotly_chart(fig_payoff, use_container_width=True, key="options_payoff")

def data_overview_tab(config, instances):
    """Data overview tab content"""
    st.markdown('<div class="sub-header">📈 Data Overview</div>', unsafe_allow_html=True)
    
    if st.session_state.current_data is not None:
        # Data summary
        st.subheader("Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Data Points:** {len(st.session_state.current_data)}")
            st.write(f"**Date Range:** {st.session_state.current_data.index.min().strftime('%Y-%m-%d')} to {st.session_state.current_data.index.max().strftime('%Y-%m-%d')}")
            st.write(f"**Assets:** {len(st.session_state.current_data.columns)}")
        
        with col2:
            if config['portfolio_type'] != "Options Portfolio":
                st.write("**Portfolio Weights:**")
                for i, col in enumerate(st.session_state.current_data.columns):
                    weight = st.session_state.portfolio_weights[i] if i < len(st.session_state.portfolio_weights) else 0
                    st.write(f"- {col}: {weight:.2%}")
        
        # Raw data display
        st.subheader("Raw Data")
        st.dataframe(st.session_state.current_data.tail(20), use_container_width=True)
        
        # Returns data
        st.subheader("Returns Data")
        st.dataframe(st.session_state.current_returns.tail(20), use_container_width=True)
        
        # Correlation matrix (for multi-asset portfolios)
        if len(st.session_state.current_data.columns) > 1 and config['portfolio_type'] != "Options Portfolio":
            st.subheader("Correlation Matrix")
            corr_matrix = st.session_state.current_returns.corr()
            fig_corr = instances['visualization'].plot_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True, key="data_correlation")
    
    else:
        st.info("No data loaded. Please configure and load data from the sidebar.")

def backtesting_tab(config, instances):
    """Backtesting tab content"""
    st.markdown('<div class="sub-header">🔄 Backtesting Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.current_returns is not None:
        # Backtesting parameters
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_window = st.slider("Backtesting Window", 50, 500, 252, key="backtest_window")
        
        with col2:
            backtest_confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key="backtest_confidence")
        
        if st.button("Run Backtesting", key="run_backtest"):
            with st.spinner("Running backtesting..."):
                # Determine portfolio type and prepare data
                if config['portfolio_type'] == "Options Portfolio" and st.session_state.options_data:
                    # Options backtesting
                    backtest_results = instances['backtesting'].perform_backtesting(
                        st.session_state.current_returns,
                        backtest_confidence,
                        backtest_window,
                        config['var_model'],
                        portfolio_type="options",
                        options_data=st.session_state.options_data
                    )
                else:
                    # Regular portfolio backtesting
                    if len(st.session_state.current_returns.shape) > 1:
                        portfolio_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
                    else:
                        portfolio_returns = st.session_state.current_returns
                    
                    # Create VaR calculation function
                    def var_method(returns, conf_level, horizon):
                        if config['var_model'] == "Parametric (Delta-Normal)":
                            return instances['var_engines'].calculate_parametric_var(returns, conf_level, horizon)
                        elif config['var_model'] == "Historical Simulation":
                            return instances['var_engines'].calculate_historical_var(returns, conf_level, horizon)
                        elif config['var_model'] == "Monte Carlo":
                            return instances['var_engines'].calculate_monte_carlo_var(returns, conf_level, horizon)
                        elif config['var_model'] == "GARCH":
                            return instances['var_engines'].calculate_garch_var(returns, conf_level, horizon)
                        else:
                            return instances['var_engines'].calculate_parametric_var(returns, conf_level, horizon)
                    
                    backtest_results = instances['backtesting'].perform_backtesting(
                        portfolio_returns,
                        backtest_confidence,
                        backtest_window,
                        var_method
                    )
                
                if backtest_results:
                    # Display results
                    st.subheader("Backtesting Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Violations", backtest_results['violations'])
                    
                    with col2:
                        st.metric("Expected Violations", f"{backtest_results['expected_violations']:.1f}")
                    
                    with col3:
                        st.metric("Violation Rate", f"{backtest_results['violation_rate']:.2%}")
                    
                    with col4:
                        kupiec_pvalue = backtest_results.get('kupiec_pvalue', 0)
                        status = "Pass" if kupiec_pvalue > 0.05 else "Fail"
                        st.metric("Kupiec Test", status, f"p-value: {kupiec_pvalue:.4f}")
                    
                    # Statistical tests
                    st.subheader("Statistical Tests")
                    
                    test_col1, test_col2, test_col3 = st.columns(3)
                    
                    with test_col1:
                        st.write("**Kupiec Test (Unconditional Coverage)**")
                        st.write(f"LR Statistic: {backtest_results.get('kupiec_lr', 0):.4f}")
                        st.write(f"P-value: {backtest_results.get('kupiec_pvalue', 0):.4f}")
                        st.write(f"Result: {'Pass' if backtest_results.get('kupiec_pvalue', 0) > 0.05 else 'Fail'}")
                    
                    with test_col2:
                        st.write("**Independence Test**")
                        st.write(f"LR Statistic: {backtest_results.get('independence_lr', 0):.4f}")
                        st.write(f"P-value: {backtest_results.get('independence_pvalue', 0):.4f}")
                        st.write(f"Result: {'Pass' if backtest_results.get('independence_pvalue', 0) > 0.05 else 'Fail'}")
                    
                    with test_col3:
                        st.write("**Conditional Coverage Test**")
                        st.write(f"LR Statistic: {backtest_results.get('cc_lr', 0):.4f}")
                        st.write(f"P-value: {backtest_results.get('cc_pvalue', 0):.4f}")
                        st.write(f"Result: {'Pass' if backtest_results.get('cc_pvalue', 0) > 0.05 else 'Fail'}")
                    
                    # Basel Traffic Light
                    st.subheader("Basel Traffic Light System")
                    
                    traffic_light = instances['backtesting'].basel_traffic_light(
                        backtest_results['violations'],
                        backtest_results['expected_violations']
                    )
                    
                    if traffic_light == "Green":
                        st.success(f"🟢 **{traffic_light} Zone** - Model performance is adequate")
                    elif traffic_light == "Yellow":
                        st.warning(f"🟡 **{traffic_light} Zone** - Model requires attention")
                    else:
                        st.error(f"🔴 **{traffic_light} Zone** - Model requires review")
                    
                    # Violations chart
                    if 'violations_dates' in backtest_results and backtest_results['violations_dates']:
                        st.subheader("VaR Violations Over Time")
                        
                        if config['portfolio_type'] == "Options Portfolio":
                            returns_for_plot = st.session_state.current_returns
                        else:
                            if len(st.session_state.current_returns.shape) > 1:
                                returns_for_plot = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
                            else:
                                returns_for_plot = st.session_state.current_returns
                        
                        fig_violations = instances['visualization'].plot_var_violations(
                            returns_for_plot,
                            backtest_results['var_estimates'],
                            backtest_results['violations_dates']
                        )
                        st.plotly_chart(fig_violations, use_container_width=True, key="backtest_violations")
    
    else:
        st.info("No data available for backtesting. Please load data first.")

def stress_testing_tab(config, instances):
    """Stress testing tab content"""
    st.markdown('<div class="sub-header">⚡ Stress Testing</div>', unsafe_allow_html=True)
    
    if st.session_state.current_returns is not None:
        # Stress testing options
        stress_type = st.selectbox(
            "Stress Test Type",
            ["Historical Scenarios", "Custom Scenario"],
            key="stress_type"
        )
        
        if stress_type == "Historical Scenarios":
            scenario = st.selectbox(
                "Select Scenario",
                ["2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash"],
                key="historical_scenario"
            )
            
            if st.button("Run Historical Stress Test", key="run_historical_stress"):
                with st.spinner("Running stress test..."):
                    if config['portfolio_type'] == "Options Portfolio" and st.session_state.options_data:
                        stress_results = instances['stress_testing'].run_stress_test(
                            st.session_state.current_returns,
                            scenario,
                            config['confidence_level'],
                            portfolio_type="options",
                            options_data=st.session_state.options_data
                        )
                    else:
                        if len(st.session_state.current_returns.shape) > 1:
                            portfolio_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
                        else:
                            portfolio_returns = st.session_state.current_returns
                        
                        stress_results = instances['stress_testing'].run_stress_test(
                            portfolio_returns,
                            scenario,
                            config['confidence_level']
                        )
                    
                    if stress_results:
                        display_stress_results(stress_results, instances)
        
        else:  # Custom Scenario
            st.subheader("Custom Stress Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                vol_shock = st.slider("Volatility Shock (%)", -50, 200, 50, key="vol_shock")
            
            with col2:
                corr_shock = st.slider("Correlation Shock", 0.0, 1.0, 0.3, key="corr_shock")
            
            with col3:
                market_shock = st.slider("Market Shock (%)", -50, 50, -20, key="market_shock")
            
            if st.button("Run Custom Stress Test", key="run_custom_stress"):
                with st.spinner("Running custom stress test..."):
                    if config['portfolio_type'] == "Options Portfolio" and st.session_state.options_data:
                        stress_results = instances['stress_testing'].run_custom_stress_test(
                            st.session_state.current_returns,
                            vol_shock,
                            corr_shock,
                            market_shock,
                            config['confidence_level'],
                            portfolio_type="options",
                            options_data=st.session_state.options_data
                        )
                    else:
                        if len(st.session_state.current_returns.shape) > 1:
                            portfolio_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
                        else:
                            portfolio_returns = st.session_state.current_returns
                        
                        stress_results = instances['stress_testing'].run_custom_stress_test(
                            portfolio_returns,
                            vol_shock,
                            corr_shock,
                            market_shock,
                            config['confidence_level']
                        )
                    
                    if stress_results:
                        display_stress_results(stress_results, instances)
        
        # Sensitivity Analysis
        st.subheader("Sensitivity Analysis")
        
        if st.button("Run Sensitivity Analysis", key="run_sensitivity"):
            with st.spinner("Running sensitivity analysis..."):
                if config['portfolio_type'] == "Options Portfolio" and st.session_state.options_data:
                    sensitivity_data = instances['stress_testing'].sensitivity_analysis(
                        st.session_state.current_returns,
                        config['confidence_level'],
                        portfolio_type="options",
                        options_data=st.session_state.options_data
                    )
                else:
                    if len(st.session_state.current_returns.shape) > 1:
                        portfolio_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
                    else:
                        portfolio_returns = st.session_state.current_returns
                    
                    sensitivity_data = instances['stress_testing'].sensitivity_analysis(
                        portfolio_returns,
                        config['confidence_level']
                    )
                
                if not sensitivity_data.empty:
                    st.subheader("Sensitivity Analysis Results")
                    
                    fig_sensitivity = instances['visualization'].plot_sensitivity_analysis(sensitivity_data)
                    st.plotly_chart(fig_sensitivity, use_container_width=True, key="sensitivity_analysis")
                    
                    st.dataframe(sensitivity_data, use_container_width=True)
    
    else:
        st.info("No data available for stress testing. Please load data first.")

def display_stress_results(stress_results, instances):
    """Display stress testing results"""
    st.subheader("Stress Test Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Baseline VaR",
            f"${stress_results['baseline_var']:,.0f}",
            "Normal Conditions"
        )
    
    with col2:
        st.metric(
            "Stressed VaR",
            f"${stress_results['stressed_var']:,.0f}",
            "Stress Scenario"
        )
    
    with col3:
        st.metric(
            "VaR Increase",
            f"{stress_results['var_increase']:.1f}%",
            "Relative Change"
        )
    
    # Worst case scenario
    if 'worst_case' in stress_results:
        st.metric(
            "Worst Case Loss",
            f"${stress_results['worst_case']:,.0f}",
            "1st Percentile"
        )
    
    # Scenario description
    st.info(f"**Scenario:** {stress_results['scenario_description']}")

def rolling_analysis_tab(config, instances):
    """Rolling analysis tab content"""
    st.markdown('<div class="sub-header">📊 Rolling Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.current_returns is not None:
        # Rolling analysis parameters
        col1, col2 = st.columns(2)
        
        with col1:
            rolling_window = st.slider("Rolling Window (days)", 30, 252, 60, key="rolling_window")
        
        with col2:
            rolling_metric = st.selectbox(
                "Metric to Analyze",
                ["VaR", "Volatility", "Sharpe Ratio", "Drawdown", "All Metrics"],
                key="rolling_metric"
            )
        
        if st.button("Calculate Rolling Metrics", key="calc_rolling"):
            with st.spinner("Calculating rolling metrics..."):
                if config['portfolio_type'] == "Options Portfolio":
                    # For options, use underlying returns
                    analysis_returns = st.session_state.current_returns
                else:
                    if len(st.session_state.current_returns.shape) > 1:
                        analysis_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
                    else:
                        analysis_returns = st.session_state.current_returns
                
                if rolling_metric == "VaR":
                    rolling_var = instances['rolling_analysis'].calculate_rolling_var(
                        analysis_returns, config['confidence_level'], rolling_window
                    )
                    
                    if not rolling_var.empty:
                        st.subheader("Rolling VaR")
                        fig_rolling = instances['visualization'].plot_rolling_metrics(rolling_var, "VaR")
                        st.plotly_chart(fig_rolling, use_container_width=True, key="rolling_var")
                
                elif rolling_metric == "Volatility":
                    rolling_vol = instances['rolling_analysis'].calculate_rolling_volatility(
                        analysis_returns, rolling_window
                    )
                    
                    if not rolling_vol.empty:
                        st.subheader("Rolling Volatility")
                        fig_rolling = instances['visualization'].plot_rolling_metrics(rolling_vol, "Volatility")
                        st.plotly_chart(fig_rolling, use_container_width=True, key="rolling_vol")
                
                elif rolling_metric == "Sharpe Ratio":
                    rolling_sharpe = instances['rolling_analysis'].calculate_rolling_sharpe(
                        analysis_returns, rolling_window
                    )
                    
                    if not rolling_sharpe.empty:
                        st.subheader("Rolling Sharpe Ratio")
                        fig_rolling = instances['visualization'].plot_rolling_metrics(rolling_sharpe, "Sharpe Ratio")
                        st.plotly_chart(fig_rolling, use_container_width=True, key="rolling_sharpe")
                
                elif rolling_metric == "Drawdown":
                    drawdown = instances['rolling_analysis'].calculate_drawdown(analysis_returns)
                    
                    if not drawdown.empty:
                        st.subheader("Drawdown Analysis")
                        fig_drawdown = instances['visualization'].plot_drawdown(drawdown)
                        st.plotly_chart(fig_drawdown, use_container_width=True, key="rolling_drawdown")
                        
                        # Maximum drawdown info
                        max_dd_info = instances['rolling_analysis'].calculate_maximum_drawdown(analysis_returns)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Maximum Drawdown", f"{max_dd_info['max_drawdown']*100:.2f}%")
                        
                        with col2:
                            if max_dd_info['max_drawdown_date']:
                                st.metric("Max DD Date", max_dd_info['max_drawdown_date'].strftime('%Y-%m-%d'))
                        
                        with col3:
                            if max_dd_info['peak_date']:
                                st.metric("Peak Date", max_dd_info['peak_date'].strftime('%Y-%m-%d'))
                
                else:  # All Metrics
                    rolling_summary = instances['rolling_analysis'].generate_rolling_metrics_summary(
                        analysis_returns, rolling_window, config['confidence_level']
                    )
                    
                    if rolling_summary:
                        st.subheader("Comprehensive Rolling Analysis")
                        
                        # Display each metric
                        for metric_name, metric_data in rolling_summary.items():
                            if not metric_data.empty:
                                st.write(f"**{metric_name.replace('_', ' ')}**")
                                fig_metric = instances['visualization'].plot_rolling_metrics(
                                    metric_data, metric_name.replace('_', ' ')
                                )
                                st.plotly_chart(fig_metric, use_container_width=True, key=f"rolling_{metric_name}")
        
        # Correlation analysis for multi-asset portfolios
        if len(st.session_state.current_returns.columns) > 1 and config['portfolio_type'] != "Options Portfolio":
            st.subheader("Rolling Correlation Analysis")
            
            if st.button("Calculate Rolling Correlations", key="calc_rolling_corr"):
                with st.spinner("Calculating rolling correlations..."):
                    rolling_correlations = instances['rolling_analysis'].calculate_rolling_correlations(
                        st.session_state.current_returns, rolling_window
                    )
                    
                    if rolling_correlations:
                        for pair, corr_data in rolling_correlations.items():
                            if not corr_data.empty:
                                st.write(f"**{pair} Correlation**")
                                fig_corr = instances['visualization'].plot_rolling_metrics(
                                    corr_data, f"{pair} Correlation"
                                )
                                st.plotly_chart(fig_corr, use_container_width=True, key=f"rolling_corr_{pair}")
    
    else:
        st.info("No data available for rolling analysis. Please load data first.")

def export_data_tab(config, instances):
    """Export data tab content"""
    st.markdown('<div class="sub-header">💾 Export Data</div>', unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        st.subheader("Available Data for Export")
        
        # Data selection
        export_options = st.multiselect(
            "Select data to export:",
            [
                "Raw Price Data",
                "Returns Data", 
                "VaR Results",
                "Portfolio Statistics",
                "Correlation Matrix"
            ],
            default=["Raw Price Data", "Returns Data"],
            key="export_options"
        )
        
        # Export format
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "Excel"],
            key="export_format"
        )
        
        if st.button("Generate Export", key="generate_export"):
            with st.spinner("Preparing export..."):
                export_data = {}
                
                # Prepare data based on selection
                if "Raw Price Data" in export_options and st.session_state.current_data is not None:
                    export_data["Price_Data"] = st.session_state.current_data
                
                if "Returns Data" in export_options and st.session_state.current_returns is not None:
                    export_data["Returns_Data"] = st.session_state.current_returns
                
                if "VaR Results" in export_options and st.session_state.var_results:
                    var_df = pd.DataFrame([st.session_state.var_results])
                    export_data["VaR_Results"] = var_df
                
                if "Portfolio Statistics" in export_options and st.session_state.current_returns is not None:
                    if config['portfolio_type'] == "Options Portfolio":
                        stats_returns = st.session_state.current_returns
                    else:
                        if len(st.session_state.current_returns.shape) > 1:
                            stats_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
                        else:
                            stats_returns = st.session_state.current_returns
                    
                    stats = instances['utils'].calculate_portfolio_statistics(stats_returns)
                    stats_df = pd.DataFrame([stats])
                    export_data["Portfolio_Statistics"] = stats_df
                
                if "Correlation Matrix" in export_options and st.session_state.current_returns is not None:
                    if len(st.session_state.current_returns.columns) > 1:
                        corr_matrix = st.session_state.current_returns.corr()
                        export_data["Correlation_Matrix"] = corr_matrix
                
                # Generate download
                if export_data:
                    if export_format == "CSV":
                        # For CSV, combine all data into one file
                        combined_data = pd.DataFrame()
                        for name, data in export_data.items():
                            if isinstance(data, pd.DataFrame):
                                data_copy = data.copy()
                                data_copy.columns = [f"{name}_{col}" for col in data_copy.columns]
                                if combined_data.empty:
                                    combined_data = data_copy
                                else:
                                    combined_data = pd.concat([combined_data, data_copy], axis=1)
                        
                        csv_data = instances['utils'].export_to_csv(combined_data, "risk_analysis_export")
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"risk_analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_csv"
                        )
                    
                    else:  # Excel
                        excel_data = instances['utils'].export_to_excel(export_data, "risk_analysis_export")
                        
                        if excel_data:
                            st.download_button(
                                label="📥 Download Excel",
                                data=excel_data,
                                file_name=f"risk_analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_excel"
                            )
                    
                    st.success("Export generated successfully!")
                
                else:
                    st.warning("No data selected for export.")
        
        # Data preview
        if export_options:
            st.subheader("Data Preview")
            
            for option in export_options:
                if option == "Raw Price Data" and st.session_state.current_data is not None:
                    st.write("**Raw Price Data (Last 10 rows):**")
                    st.dataframe(st.session_state.current_data.tail(10), use_container_width=True)
                
                elif option == "Returns Data" and st.session_state.current_returns is not None:
                    st.write("**Returns Data (Last 10 rows):**")
                    st.dataframe(st.session_state.current_returns.tail(10), use_container_width=True)
                
                elif option == "VaR Results" and st.session_state.var_results:
                    st.write("**VaR Results:**")
                    var_display = {}
                    for key, value in st.session_state.var_results.items():
                        if isinstance(value, (int, float)):
                            var_display[key] = f"{value:.4f}"
                        else:
                            var_display[key] = str(value)
                    
                    var_df = pd.DataFrame([var_display])
                    st.dataframe(var_df, use_container_width=True)
                
                elif option == "Portfolio Statistics" and st.session_state.current_returns is not None:
                    st.write("**Portfolio Statistics:**")
                    
                    if config['portfolio_type'] == "Options Portfolio":
                        stats_returns = st.session_state.current_returns
                    else:
                        if len(st.session_state.current_returns.shape) > 1:
                            stats_returns = st.session_state.current_returns.dot(st.session_state.portfolio_weights)
                        else:
                            stats_returns = st.session_state.current_returns
                    
                    stats = instances['utils'].calculate_portfolio_statistics(stats_returns)
                    stats_display = {}
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            stats_display[key] = f"{value:.4f}"
                        else:
                            stats_display[key] = str(value)
                    
                    stats_df = pd.DataFrame([stats_display])
                    st.dataframe(stats_df, use_container_width=True)
                
                elif option == "Correlation Matrix" and st.session_state.current_returns is not None:
                    if len(st.session_state.current_returns.columns) > 1:
                        st.write("**Correlation Matrix:**")
                        corr_matrix = st.session_state.current_returns.corr()
                        st.dataframe(corr_matrix, use_container_width=True)
    
    else:
        st.info("No data available for export. Please load data first.")
    
    # Export documentation
    st.subheader("Export Information")
    
    st.markdown("""
    **Available Export Options:**
    
    - **Raw Price Data**: Historical price data for all assets
    - **Returns Data**: Calculated returns for all assets  
    - **VaR Results**: Value at Risk calculations and metrics
    - **Portfolio Statistics**: Comprehensive portfolio performance metrics
    - **Correlation Matrix**: Asset correlation analysis (multi-asset portfolios only)
    
    **Export Formats:**
    
    - **CSV**: Single file with all selected data combined
    - **Excel**: Multiple sheets with organized data sections
    
    **File Naming**: Files are automatically timestamped for easy organization.
    """)

if __name__ == "__main__":
    main()