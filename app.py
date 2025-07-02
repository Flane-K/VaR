import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #404040;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        color: #4CAF50;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0 0.5rem 0;
    }
    .help-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data persistence
if 'data_ingestion' not in st.session_state:
    st.session_state.data_ingestion = DataIngestion()
if 'var_engines' not in st.session_state:
    st.session_state.var_engines = VaREngines()
if 'backtesting' not in st.session_state:
    st.session_state.backtesting = Backtesting()
if 'stress_testing' not in st.session_state:
    st.session_state.stress_testing = StressTesting()
if 'rolling_analysis' not in st.session_state:
    st.session_state.rolling_analysis = RollingAnalysis()
if 'options_var' not in st.session_state:
    st.session_state.options_var = OptionsVaR()
if 'visualization' not in st.session_state:
    st.session_state.visualization = Visualization()
if 'utils' not in st.session_state:
    st.session_state.utils = Utils()

# Initialize persistent data storage
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'returns' not in st.session_state:
    st.session_state.returns = None
if 'portfolio_returns' not in st.session_state:
    st.session_state.portfolio_returns = None
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'data_source_changed' not in st.session_state:
    st.session_state.data_source_changed = False

# Function to generate synthetic data
def generate_synthetic_data(num_days, initial_price, annual_return, annual_volatility, random_seed=42):
    """Generate synthetic stock price data"""
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
    df = pd.DataFrame({'Price': prices}, index=dates)
    return df

# Function to calculate VaR based on selected model
def calculate_var_by_model(returns, var_model, confidence_level, time_horizon, **kwargs):
    """Calculate VaR using the selected model"""
    if returns is None or len(returns) == 0:
        return 0
    
    try:
        if var_model == "Parametric (Delta-Normal)":
            cornish_fisher = kwargs.get('cornish_fisher', False)
            return st.session_state.var_engines.calculate_parametric_var(
                returns, confidence_level, time_horizon, cornish_fisher
            )
        elif var_model == "Historical Simulation":
            return st.session_state.var_engines.calculate_historical_var(
                returns, confidence_level, time_horizon
            )
        elif var_model == "Monte Carlo":
            num_simulations = kwargs.get('num_simulations', 10000)
            return st.session_state.var_engines.calculate_monte_carlo_var(
                returns, confidence_level, time_horizon, num_simulations
            )
        elif var_model == "GARCH-Based":
            garch_p = kwargs.get('garch_p', 1)
            garch_q = kwargs.get('garch_q', 1)
            return st.session_state.var_engines.calculate_garch_var(
                returns, confidence_level, time_horizon, garch_p, garch_q
            )
        elif var_model == "Extreme Value Theory":
            return st.session_state.var_engines.calculate_evt_var(
                returns, confidence_level
            )
        else:
            return st.session_state.var_engines.calculate_parametric_var(
                returns, confidence_level, time_horizon, False
            )
    except Exception as e:
        st.error(f"Error calculating VaR with {var_model}: {str(e)}")
        return 0

# Sidebar Controls
with st.sidebar:
    st.title("🔧 Risk Analytics Controls")
    
    # Data Source Selection
    st.markdown('<div class="sidebar-header">📊 Data Source</div>', unsafe_allow_html=True)
    data_source = st.selectbox(
        "Select Data Source",
        ["Live Market Data", "CSV/XLSX Upload", "Manual Entry"],
        key="data_source"
    )
    
    # Portfolio Definition
    st.markdown('<div class="sidebar-header">💼 Portfolio Settings</div>', unsafe_allow_html=True)
    portfolio_type = st.selectbox(
        "Portfolio Type",
        ["Single Asset", "Multi-Asset", "Crypto Portfolio", "Options Portfolio"],
        key="portfolio_type"
    )
    
    # Dynamic symbol defaults based on portfolio type
    if data_source == "Live Market Data":
        if portfolio_type == "Single Asset":
            default_symbols = "AAPL"
        elif portfolio_type == "Multi-Asset":
            default_symbols = "AAPL,GOOGL,MSFT,TSLA"
        elif portfolio_type == "Crypto Portfolio":
            default_symbols = "BTC-USD"
        else:  # Options Portfolio
            default_symbols = "AAPL"
        
        symbols = st.text_input("Enter symbols (comma-separated)", default_symbols)
        symbols_list = [s.strip().upper() for s in symbols.split(",")]
        
        # Crypto symbols for multi-asset
        if portfolio_type == "Multi-Asset":
            crypto_symbols = st.text_input("Crypto symbols (optional)", "BTC-USD,ETH-USD")
            if crypto_symbols.strip():
                crypto_list = [s.strip().upper() for s in crypto_symbols.split(",")]
                symbols_list.extend(crypto_list)
    
    elif data_source == "CSV/XLSX Upload":
        uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=["csv", "xlsx"])
        
    elif data_source == "Manual Entry":
        st.subheader("Manual Data Entry")
        
        # Option to generate synthetic data
        use_generated_data = st.checkbox("Use Generated Synthetic Data", value=True)
        
        if use_generated_data:
            with st.expander("📊 Data Generation Parameters"):
                num_days = st.number_input("Number of Days", 100, 2000, 500)
                initial_price = st.number_input("Initial Price ($)", 10.0, 1000.0, 100.0)
                annual_return = st.slider("Annual Return (%)", -20, 30, 8) / 100
                annual_volatility = st.slider("Annual Volatility (%)", 5, 50, 20) / 100
                random_seed = st.number_input("Random Seed", 1, 1000, 42)
                
                if st.button("Generate Data"):
                    st.session_state.generated_data = generate_synthetic_data(
                        num_days, initial_price, annual_return, annual_volatility, random_seed
                    )
                    st.success("✅ Synthetic data generated!")
        else:
            manual_data_input = st.text_area(
                "Enter historical prices (Date, Price per line)",
                "2023-01-01,100\n2023-01-02,101\n2023-01-03,102\n2023-01-04,99\n2023-01-05,103"
            )
    
    # VaR Model Selection
    st.markdown('<div class="sidebar-header">⚙️ VaR Model</div>', unsafe_allow_html=True)
    var_model = st.selectbox(
        "VaR Calculation Method",
        ["Parametric (Delta-Normal)", "Historical Simulation", "Monte Carlo", "GARCH-Based", "Extreme Value Theory"],
        key="var_model"
    )
    
    # Risk Parameters
    st.markdown('<div class="sidebar-header">📈 Risk Parameters</div>', unsafe_allow_html=True)
    confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100
    time_horizon = st.number_input("Time Horizon (days)", 1, 30, 1)
    window_size = st.number_input("Historical Window (days)", 30, 1000, 252)
    
    # Model-specific parameters
    if var_model == "Monte Carlo":
        num_simulations = st.number_input("Number of Simulations", 1000, 100000, 10000)
    
    if var_model == "GARCH-Based":
        garch_p = st.number_input("GARCH P", 1, 5, 1)
        garch_q = st.number_input("GARCH Q", 1, 5, 1)
    
    # Date Range (default to 2 years)
    st.markdown('<div class="sidebar-header">📅 Date Range</div>', unsafe_allow_html=True)
    end_date = st.date_input("End Date", datetime.now())
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=730))  # 2 years
    
    # Advanced Settings
    st.markdown('<div class="sidebar-header">🔧 Advanced Settings</div>', unsafe_allow_html=True)
    decay_factor = st.slider("Decay Factor", 0.9, 0.99, 0.94)
    cornish_fisher = st.checkbox("Apply Cornish-Fisher Adjustment")
    
    # Portfolio Weights (for multi-asset)
    weights = {}
    if portfolio_type == "Multi-Asset" and data_source == "Live Market Data" and symbols_list:
        st.markdown('<div class="sidebar-header">⚖️ Portfolio Weights</div>', unsafe_allow_html=True)
        for symbol in symbols_list:
            weights[symbol] = st.slider(f"{symbol} Weight", 0.0, 1.0, 1.0/len(symbols_list))
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            st.warning("Total weight is zero. Please adjust weights.")
            weights = {s: 1.0/len(symbols_list) for s in symbols_list}

# Main Content Area
st.title("📊 VaR & Risk Analytics Platform")

# Tab Navigation
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Dashboard", 
    "🧮 VaR Calculator", 
    "🧪 Backtest & Validate", 
    "⚡ Scenario & Stress", 
    "📈 Rolling Analysis", 
    "📊 Option VaR", 
    "📄 Reports & Exports",
    "❓ Help"
])

# Enhanced data loading functions with crypto support
@st.cache_data(ttl=3600)
def load_market_data(symbols_list, start_date, end_date):
    if not symbols_list:
        return None
    try:
        # Separate crypto and regular symbols
        crypto_symbols = [s for s in symbols_list if '-USD' in s]
        regular_symbols = [s for s in symbols_list if '-USD' not in s]
        
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
        
        return all_data if not all_data.empty else None
        
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        return None

@st.cache_data
def load_uploaded_file_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        else:
            df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            return df[numeric_cols]
        else:
            st.error("Uploaded file does not contain numeric price data.")
            return None
    except Exception as e:
        st.error(f"Error reading uploaded file: {str(e)}")
        return None

@st.cache_data
def load_manual_data(manual_data_input):
    try:
        data = [line.split(',') for line in manual_data_input.strip().split('\n') if line.strip()]
        df = pd.DataFrame(data, columns=['Date', 'Price'])
        df['Date'] = pd.to_datetime(df['Date'])
        df['Price'] = pd.to_numeric(df['Price'])
        df = df.set_index('Date')
        return df
    except Exception as e:
        st.error(f"Error parsing manual data: {str(e)}")
        return None

# Data Loading Logic with persistence
def load_data():
    """Load data and update session state"""
    market_data = None
    returns = None
    portfolio_returns = None
    
    if data_source == "Live Market Data":
        if symbols_list:
            market_data = load_market_data(symbols_list, start_date, end_date)
            if market_data is not None and not market_data.empty:
                returns = market_data.pct_change().dropna()
                
                if portfolio_type == "Multi-Asset":
                    if all(col in returns.columns for col in weights.keys()):
                        aligned_weights = pd.Series(weights).reindex(returns.columns, fill_value=0).values
                        portfolio_returns = returns.dot(aligned_weights)
                    else:
                        available_symbols = [s for s in symbols_list if s in returns.columns]
                        if available_symbols:
                            portfolio_returns = returns[available_symbols].mean(axis=1)
                        else:
                            portfolio_returns = None
                elif portfolio_type in ["Single Asset", "Crypto Portfolio"]:
                    if len(returns.columns) > 0:
                        portfolio_returns = returns.iloc[:, 0]
                    else:
                        portfolio_returns = None
                else:  # Options Portfolio
                    portfolio_returns = None
    
    elif data_source == "CSV/XLSX Upload":
        if 'uploaded_file' in locals() and uploaded_file is not None:
            uploaded_df = load_uploaded_file_data(uploaded_file)
            if uploaded_df is not None and not uploaded_df.empty:
                market_data = uploaded_df
                returns = market_data.pct_change().dropna()
                if len(market_data.columns) > 0:
                    portfolio_returns = market_data.iloc[:, 0].pct_change().dropna()
    
    elif data_source == "Manual Entry":
        if use_generated_data and st.session_state.generated_data is not None:
            market_data = st.session_state.generated_data
            portfolio_returns = market_data['Price'].pct_change().dropna()
            returns = pd.DataFrame({'Asset': portfolio_returns})
        elif not use_generated_data and 'manual_data_input' in locals() and manual_data_input:
            manual_df = load_manual_data(manual_data_input)
            if manual_df is not None and not manual_df.empty:
                market_data = manual_df
                portfolio_returns = manual_df['Price'].pct_change().dropna()
                returns = pd.DataFrame({'Asset': portfolio_returns})
    
    # Update session state
    st.session_state.market_data = market_data
    st.session_state.returns = returns
    st.session_state.portfolio_returns = portfolio_returns
    
    return market_data, returns, portfolio_returns

# Load data
market_data, returns, portfolio_returns = load_data()

# Dashboard Tab
with tab1:
    st.header("📊 Risk Dashboard")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        # Key Metrics with dynamic VaR model
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_var = calculate_var_by_model(
                portfolio_returns, var_model, confidence_level, time_horizon,
                cornish_fisher=cornish_fisher,
                num_simulations=num_simulations if var_model == "Monte Carlo" else 10000,
                garch_p=garch_p if var_model == "GARCH-Based" else 1,
                garch_q=garch_q if var_model == "GARCH-Based" else 1
            )
            model_short = var_model.split()[0]  # Get first word of model name
            st.metric(f"VaR (95%) - {model_short}", f"${current_var:,.2f}")
        
        with col2:
            expected_shortfall = st.session_state.var_engines.calculate_expected_shortfall(
                portfolio_returns, confidence_level
            )
            st.metric("Expected Shortfall", f"${expected_shortfall:,.2f}")
        
        with col3:
            volatility = portfolio_returns.std() * np.sqrt(252) * 100
            st.metric("Annual Volatility", f"{volatility:.2f}%")
        
        with col4:
            risk_free_rate_annual = 0.02
            excess_returns = portfolio_returns.mean() * 252 - risk_free_rate_annual
            sharpe_ratio = excess_returns / (portfolio_returns.std() * np.sqrt(252))
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
        
        # Model-specific insights
        st.subheader(f"📊 {var_model} Model Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            if var_model == "Parametric (Delta-Normal)":
                st.info("📈 **Normal Distribution Assumption**: Assumes returns follow a normal distribution. Best for stable markets.")
            elif var_model == "Historical Simulation":
                st.info("📊 **Non-parametric Method**: Uses actual historical returns. Captures real market behavior.")
            elif var_model == "Monte Carlo":
                st.info(f"🎲 **Simulation-based**: Using {num_simulations:,} simulations for robust estimates.")
            elif var_model == "GARCH-Based":
                st.info(f"📈 **Volatility Clustering**: GARCH({garch_p},{garch_q}) model captures changing volatility.")
            elif var_model == "Extreme Value Theory":
                st.info("⚡ **Tail Risk Focus**: Specialized for extreme market events and tail risks.")
        
        with col2:
            # Quick model comparison highlighting current selection
            comparison_data = {}
            for model in ["Parametric (Delta-Normal)", "Historical Simulation", "Monte Carlo"]:
                try:
                    var_val = calculate_var_by_model(portfolio_returns, model, confidence_level, time_horizon)
                    comparison_data[model.split()[0]] = var_val
                except:
                    comparison_data[model.split()[0]] = 0
            
            if comparison_data:
                fig = px.bar(
                    x=list(comparison_data.keys()),
                    y=list(comparison_data.values()),
                    title="VaR Model Comparison",
                    template="plotly_dark"
                )
                # Highlight current model
                colors = ['#ff6b6b' if model.split()[0] == var_model.split()[0] else '#4ecdc4' 
                         for model in comparison_data.keys()]
                fig.update_traces(marker_color=colors)
                st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio Performance Chart
        st.subheader("📈 Portfolio Performance")
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00ff88', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns Distribution with VaR
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 Returns Distribution - {var_model}")
            fig = st.session_state.visualization.plot_var_distribution(
                portfolio_returns, confidence_level, current_var
            )
            fig.update_layout(title=f"Returns Distribution - {var_model}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Risk Metrics Over Time")
            rolling_vol = portfolio_returns.rolling(30).std() * np.sqrt(252) * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='30-Day Rolling Volatility',
                line=dict(color='#4ecdc4', width=2)
            ))
            
            fig.update_layout(
                title="Rolling 30-Day Volatility (%)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please load data to view the dashboard.")

# VaR Calculator Tab
with tab2:
    st.header("🧮 VaR Calculator")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 VaR Calculation Results")
            
            # Calculate VaR using selected method
            calculated_var_value = calculate_var_by_model(
                portfolio_returns, var_model, confidence_level, time_horizon,
                cornish_fisher=cornish_fisher,
                num_simulations=num_simulations if var_model == "Monte Carlo" else 10000,
                garch_p=garch_p if var_model == "GARCH-Based" else 1,
                garch_q=garch_q if var_model == "GARCH-Based" else 1
            )
            
            st.metric(f"{var_model} VaR", f"${calculated_var_value:,.2f}")
            
            # Expected Shortfall
            es_value = st.session_state.var_engines.calculate_expected_shortfall(
                portfolio_returns, confidence_level
            )
            st.metric("Expected Shortfall", f"${es_value:,.2f}")
        
        with col2:
            st.subheader("📈 VaR Visualization")
            fig = st.session_state.visualization.plot_var_distribution(
                portfolio_returns, confidence_level, calculated_var_value
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model Comparison
        st.subheader("🔍 VaR Model Comparison")
        all_var_results = {}
        
        for model in ["Parametric (Delta-Normal)", "Historical Simulation", "Monte Carlo", "GARCH-Based", "Extreme Value Theory"]:
            try:
                var_val = calculate_var_by_model(portfolio_returns, model, confidence_level, time_horizon)
                all_var_results[model] = var_val
            except Exception as e:
                all_var_results[model] = f"Error: {e}"
        
        # Create comparison DataFrame and highlight current model
        comparison_df = pd.DataFrame(list(all_var_results.items()), columns=['Method', 'VaR'])
        comparison_df['Current Model'] = comparison_df['Method'] == var_model
        
        # Style the dataframe
        def highlight_current(row):
            return ['background-color: #4CAF50' if row['Current Model'] else '' for _ in row]
        
        styled_df = comparison_df.style.apply(highlight_current, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("Please load market data first to calculate VaR.")

# Backtesting Tab
with tab3:
    st.header("🧪 Backtesting & Validation")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_window = st.number_input("Backtesting Window (days)", 100, 1000, 252)
        
        with col2:
            st.subheader("📊 Backtesting Results")
            
            try:
                # Create VaR function for the selected model
                def var_func(ret, conf, horizon):
                    return calculate_var_by_model(
                        ret, var_model, conf, horizon,
                        cornish_fisher=cornish_fisher,
                        num_simulations=num_simulations if var_model == "Monte Carlo" else 10000,
                        garch_p=garch_p if var_model == "GARCH-Based" else 1,
                        garch_q=garch_q if var_model == "GARCH-Based" else 1
                    )
                
                backtest_results = st.session_state.backtesting.perform_backtesting(
                    portfolio_returns, confidence_level, backtest_window, var_func
                )
                
                if backtest_results:
                    st.metric("Kupiec Test p-value", f"{backtest_results['kupiec_pvalue']:.4f}")
                    st.metric("Actual Violations", f"{backtest_results['violations']}")
                    st.metric("Expected Violations", f"{backtest_results['expected_violations']:.1f}")
                    
                    # Basel Traffic Light
                    st.subheader("🚦 Basel Traffic Light System")
                    traffic_light = st.session_state.backtesting.basel_traffic_light(
                        backtest_results['violations'], backtest_results['expected_violations']
                    )
                    
                    if traffic_light == 'Green':
                        st.success("✅ Green Zone - Model performs well")
                    elif traffic_light == 'Yellow':
                        st.warning("⚠️ Yellow Zone - Model needs attention")
                    else:
                        st.error("🔴 Red Zone - Model requires immediate review")
                    
                    # Violation plot
                    st.subheader(f"📈 {var_model} VaR Violations Over Time")
                    fig = st.session_state.visualization.plot_var_violations(
                        portfolio_returns, backtest_results['var_estimates'], backtest_results['violations_dates']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data for backtesting or error occurred.")
            except Exception as e:
                st.error(f"Error during backtesting: {e}")
    else:
        st.info("Please load market data first to perform backtesting.")

# Scenario & Stress Tab
with tab4:
    st.header("⚡ Scenario & Stress Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Scenario Selection")
        scenario_type = st.selectbox(
            "Select Scenario",
            ["2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash", "Custom Scenario"]
        )
        
        # Custom scenario parameters
        if scenario_type == "Custom Scenario":
            vol_shock = st.slider("Volatility Shock (%)", -50, 200, 50)
            corr_shock = st.slider("Correlation Shock", -0.5, 0.5, 0.2)
            spot_shock = st.slider("Spot Price Shock (%)", -50, 50, -20)
    
    with col2:
        st.subheader(f"📈 Stress Test Results - {var_model}")
        
        if portfolio_returns is not None and not portfolio_returns.empty:
            try:
                if scenario_type == "Custom Scenario":
                    stress_results = st.session_state.stress_testing.run_custom_stress_test(
                        portfolio_returns, vol_shock, corr_shock, spot_shock, confidence_level
                    )
                else:
                    stress_results = st.session_state.stress_testing.run_stress_test(
                        portfolio_returns, scenario_type, confidence_level
                    )
                
                if stress_results and 'stressed_var' in stress_results:
                    st.metric("Stressed VaR", f"${stress_results['stressed_var']:,.2f}")
                    st.metric("VaR Increase", f"{stress_results['var_increase']:.1f}%")
                    st.metric("Worst Case Loss", f"${stress_results['worst_case']:,.2f}")
                else:
                    st.warning("Could not calculate stress test results.")
            except Exception as e:
                st.error(f"Error during stress testing: {e}")
        else:
            st.info("Please load market data first to perform stress testing.")
    
    # Scenario comparison
    if portfolio_returns is not None and not portfolio_returns.empty:
        st.subheader(f"📊 Scenario Comparison - {var_model}")
        
        scenarios = ["Normal", "2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash"]
        scenario_vars = []
        
        for scenario in scenarios:
            try:
                if scenario == "Normal":
                    var_val = calculate_var_by_model(
                        portfolio_returns, var_model, confidence_level, time_horizon,
                        cornish_fisher=cornish_fisher
                    )
                else:
                    stress_result = st.session_state.stress_testing.run_stress_test(
                        portfolio_returns, scenario, confidence_level
                    )
                    var_val = stress_result.get('stressed_var', 0) if stress_result else 0
                scenario_vars.append(var_val)
            except Exception as e:
                scenario_vars.append(0)
        
        if any(v > 0 for v in scenario_vars):
            fig = px.bar(
                x=scenarios,
                y=scenario_vars,
                title=f"VaR Across Different Scenarios - {var_model}",
                template="plotly_dark"
            )
            fig.update_traces(marker_color='#ff6b6b')
            st.plotly_chart(fig, use_container_width=True)

# Rolling Analysis Tab
with tab5:
    st.header("📈 Rolling Analysis")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            rolling_window = st.number_input("Rolling Window (days)", 30, 252, 60)
            
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Rolling VaR", "Rolling Volatility", "Rolling Sharpe", "Drawdown Analysis"]
            )
        
        # Perform rolling analysis
        if analysis_type == "Rolling VaR":
            st.subheader(f"📊 Rolling VaR Analysis - {var_model}")
            rolling_var = st.session_state.rolling_analysis.calculate_rolling_var(
                portfolio_returns, confidence_level, rolling_window
            )
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_var.index,
                y=rolling_var.values,
                mode='lines',
                name=f'Rolling {rolling_window}-Day VaR ({var_model})',
                line=dict(color='#4ecdc4', width=2)
            ))
            
            fig.update_layout(
                title=f"Rolling {rolling_window}-Day VaR - {var_model}",
                xaxis_title="Date",
                yaxis_title="VaR ($)",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Rolling Volatility":
            st.subheader("📊 Rolling Volatility Analysis")
            rolling_vol = portfolio_returns.rolling(rolling_window).std() * np.sqrt(252) * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name=f'Rolling {rolling_window}-Day Volatility',
                line=dict(color='#ff6b6b', width=2)
            ))
            
            fig.update_layout(
                title=f"Rolling {rolling_window}-Day Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Drawdown Analysis":
            st.subheader("📊 Drawdown Analysis")
            drawdown = st.session_state.rolling_analysis.calculate_drawdown(portfolio_returns)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                fill='tonexty',
                name='Drawdown',
                line=dict(color='#ff6b6b', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        if portfolio_type == "Multi-Asset" and returns is not None and not returns.empty:
            try:
                corr_matrix = returns.corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    template="plotly_dark",
                    color_continuous_scale="RdBu_r"
                )
                fig.update_layout(title="Asset Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {e}")
        else:
            st.info("Select 'Multi-Asset' portfolio type to view Correlation Heatmap.")
    else:
        st.info("Please load market data first to perform rolling analysis.")

# Options VaR Tab
with tab6:
    st.header("📊 Options Portfolio VaR")
    
    if portfolio_type == "Options Portfolio":
        # Default options parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            spot_price = st.number_input("Spot Price ($)", 50.0, 5000.0, 150.0)
            strike_price = st.number_input("Strike Price ($)", 50.0, 5000.0, 155.0)
            
        with col2:
            time_to_expiry = st.number_input("Time to Expiry (days)", 1, 365, 30) / 365
            risk_free_rate = st.slider("Risk-free Rate (%)", 0.0, 10.0, 2.0) / 100
            
        with col3:
            volatility_input = st.slider("Volatility (%)", 10, 100, 25) / 100
            option_type = st.selectbox("Option Type", ["Call", "Put"])
        
        # Options VaR calculation method
        options_var_method = st.selectbox(
            "Options VaR Method",
            ["Delta-Normal", "Delta-Gamma", "Full Revaluation Monte Carlo"]
        )
        
        # Calculate options VaR
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Options VaR Results")
            
            try:
                options_var_result = st.session_state.options_var.calculate_options_var(
                    spot_price, strike_price, time_to_expiry, risk_free_rate, 
                    volatility_input, option_type, options_var_method, confidence_level
                )
                
                st.metric("Options VaR", f"${options_var_result['var']:,.2f}")
                st.metric("Delta", f"{options_var_result['delta']:.4f}")
                st.metric("Gamma", f"{options_var_result['gamma']:.4f}")
                st.metric("Theta", f"{options_var_result['theta']:.4f}")
                st.metric("Vega", f"{options_var_result['vega']:.4f}")
            except Exception as e:
                st.error(f"Error calculating options VaR: {e}")
        
        with col2:
            st.subheader("📈 Greeks Sensitivity")
            
            # Plot Greeks
            spot_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 50)
            deltas = []
            gammas = []
            
            try:
                for s in spot_range:
                    greeks = st.session_state.options_var.calculate_greeks(
                        s, strike_price, time_to_expiry, risk_free_rate, volatility_input, option_type
                    )
                    deltas.append(greeks['delta'])
                    gammas.append(greeks['gamma'])
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Delta', 'Gamma'),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(x=spot_range, y=deltas, name='Delta', line=dict(color='#4ecdc4')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=spot_range, y=gammas, name='Gamma', line=dict(color='#ff6b6b')),
                    row=2, col=1
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    height=600,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting Greeks: {e}")
    else:
        st.info("Please select 'Options Portfolio' in the sidebar to access options VaR calculations.")

# Reports & Exports Tab
with tab7:
    st.header("📄 Reports & Exports")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Report Options")
            
            report_type = st.selectbox(
                "Report Type",
                ["Executive Summary", "Detailed Risk Report", "Backtesting Report", "Stress Testing Report"]
            )
            
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON"]
            )
            
            include_charts = st.checkbox("Include Charts", value=True)
            include_data = st.checkbox("Include Raw Data", value=False)
        
        with col2:
            st.subheader("📈 Quick Metrics Export")
            
            # Generate summary metrics with current model
            current_var = calculate_var_by_model(
                portfolio_returns, var_model, confidence_level, time_horizon,
                cornish_fisher=cornish_fisher
            )
            current_es = st.session_state.var_engines.calculate_expected_shortfall(portfolio_returns, confidence_level)
            
            summary_metrics = {
                f'VaR (95%) - {var_model}': current_var,
                'Expected Shortfall': current_es,
                'Volatility (%)': portfolio_returns.std() * np.sqrt(252) * 100,
                'Sharpe Ratio': (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252)),
                'Skewness': portfolio_returns.skew(),
                'Kurtosis': portfolio_returns.kurtosis(),
                'Model Used': var_model
            }
            
            metrics_df = pd.DataFrame(list(summary_metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(metrics_df, use_container_width=True)
        
        # Generate and download report
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                if export_format == "CSV":
                    csv_data = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV Report",
                        data=csv_data,
                        file_name=f"risk_report_{var_model.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                elif export_format == "JSON":
                    import json
                    json_serializable_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in summary_metrics.items()}
                    json_data = json.dumps(json_serializable_metrics, indent=2)
                    st.download_button(
                        label="Download JSON Report",
                        data=json_data,
                        file_name=f"risk_report_{var_model.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                st.success("✅ Report generated successfully!")
        
        # Data Export Section
        st.subheader("📊 Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Portfolio Returns"):
                csv_returns = portfolio_returns.to_csv()
                st.download_button(
                    label="Download Returns Data",
                    data=csv_returns,
                    file_name=f"portfolio_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Price Data"):
                if market_data is not None and not market_data.empty:
                    csv_prices = market_data.to_csv()
                    st.download_button(
                        label="Download Price Data",
                        data=csv_prices,
                        file_name=f"price_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No price data available to export.")
    else:
        st.info("Please load data first to generate reports.")

# Help Tab
with tab8:
    st.header("❓ Help & Documentation")
    
    st.markdown("""
    ## 🚀 Welcome to VaR & Risk Analytics Platform
    
    This comprehensive platform provides advanced Value at Risk (VaR) calculations and risk analytics for financial portfolios.
    """)
    
    # Getting Started
    with st.expander("🏁 Getting Started"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Select Data Source**: Choose from Live Market Data, CSV/XLSX Upload, or Manual Entry
        2. **Configure Portfolio**: Select portfolio type (Single Asset, Multi-Asset, Crypto, Options)
        3. **Choose VaR Model**: Pick from 5 different VaR calculation methods
        4. **Set Parameters**: Adjust confidence level, time horizon, and other risk parameters
        5. **Analyze Results**: View dashboard, run backtests, and perform stress tests
        
        ### Portfolio Types
        - **Single Asset**: Analyze individual stocks or assets
        - **Multi-Asset**: Create diversified portfolios with custom weights
        - **Crypto Portfolio**: Specialized for cryptocurrency analysis
        - **Options Portfolio**: Advanced derivatives risk modeling
        """)
    
    # Data Sources
    with st.expander("📊 Data Sources & Formats"):
        st.markdown("""
        ### Live Market Data
        - **Stocks**: Use ticker symbols (e.g., AAPL, GOOGL, MSFT)
        - **Crypto**: Add -USD suffix (e.g., BTC-USD, ETH-USD)
        - **Default Symbols**:
          - Single Asset: AAPL
          - Multi-Asset: AAPL,GOOGL,MSFT,TSLA
          - Crypto: BTC-USD
        
        ### CSV/XLSX Upload Format
        ```
        Date,Asset1,Asset2,Asset3
        2023-01-01,100.50,200.25,150.75
        2023-01-02,101.25,198.50,152.00
        2023-01-03,99.75,201.00,149.25
        ```
        - First column: Date (YYYY-MM-DD format)
        - Subsequent columns: Asset prices
        - Headers recommended but not required
        
        ### Manual Entry / Synthetic Data
        - **Generated Data**: Creates realistic stock price movements
        - **Parameters**: Initial price, annual return, volatility, number of days
        - **Format**: Date,Price (one per line)
        """)
    
    # VaR Models
    with st.expander("⚙️ VaR Models Explained"):
        st.markdown("""
        ### 1. Parametric (Delta-Normal)
        - **Best for**: Stable markets, normal return distributions
        - **Assumptions**: Returns follow normal distribution
        - **Speed**: Very fast
        - **Accuracy**: Good for most market conditions
        
        ### 2. Historical Simulation
        - **Best for**: Capturing actual market behavior
        - **Assumptions**: Past returns predict future risk
        - **Speed**: Fast
        - **Accuracy**: Excellent for stable patterns
        
        ### 3. Monte Carlo
        - **Best for**: Complex portfolios, scenario analysis
        - **Assumptions**: Parametric distribution with simulation
        - **Speed**: Moderate (depends on simulations)
        - **Accuracy**: Very high with sufficient simulations
        
        ### 4. GARCH-Based
        - **Best for**: Markets with volatility clustering
        - **Assumptions**: Time-varying volatility
        - **Speed**: Slow (model fitting required)
        - **Accuracy**: Excellent for volatile markets
        
        ### 5. Extreme Value Theory (EVT)
        - **Best for**: Tail risk, extreme market events
        - **Assumptions**: Focus on distribution tails
        - **Speed**: Moderate
        - **Accuracy**: Superior for extreme events
        """)
    
    # Parameters Guide
    with st.expander("📈 Parameter Settings"):
        st.markdown("""
        ### Risk Parameters
        - **Confidence Level**: 90-99% (95% is standard)
        - **Time Horizon**: 1-30 days (1 day is most common)
        - **Historical Window**: 30-1000 days (252 days = 1 year)
        
        ### Model-Specific Parameters
        - **Monte Carlo Simulations**: 1,000-100,000 (10,000 recommended)
        - **GARCH Parameters**: P=1, Q=1 (standard GARCH(1,1))
        - **Decay Factor**: 0.90-0.99 (0.94 is RiskMetrics standard)
        
        ### Advanced Settings
        - **Cornish-Fisher**: Adjusts for skewness and kurtosis
        - **Portfolio Weights**: Must sum to 1.0 for multi-asset portfolios
        """)
    
    # Interpretation Guide
    with st.expander("📊 Results Interpretation"):
        st.markdown("""
        ### VaR Interpretation
        - **VaR (95%, 1-day) = $10,000**: 95% confidence that losses won't exceed $10,000 in one day
        - **Expected Shortfall**: Average loss when VaR is exceeded
        - **Sharpe Ratio**: Risk-adjusted return (>1.0 is good, >2.0 is excellent)
        
        ### Backtesting Results
        - **Kupiec Test p-value > 0.05**: Model is statistically valid
        - **Basel Traffic Light**:
          - 🟢 Green: Model performs well
          - 🟡 Yellow: Model needs attention
          - 🔴 Red: Model requires review
        
        ### Stress Testing
        - **VaR Increase**: Percentage increase under stress scenarios
        - **Worst Case**: 1st percentile loss under stress
        """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        ### Common Issues
        
        **"Insufficient data for backtesting"**
        - Increase historical window or use more data
        - Minimum 252 days recommended for annual analysis
        
        **"GARCH model failed"**
        - Need at least 100 data points
        - Try reducing GARCH parameters or use different model
        
        **"No data found for symbols"**
        - Check ticker symbol spelling
        - Verify date range (weekends/holidays excluded)
        - For crypto, ensure -USD suffix
        
        **"Weights must sum to 1.0"**
        - Adjust portfolio weights in sidebar
        - Platform auto-normalizes but warns if sum ≠ 1.0
        
        ### Performance Tips
        - Use smaller simulation counts for faster Monte Carlo
        - Reduce historical window for quicker calculations
        - Cache is enabled - same parameters load faster
        """)
    
    # Contact & Support
    with st.expander("📞 Support & Resources"):
        st.markdown("""
        ### Additional Resources
        - **Academic Papers**: Basel Committee guidelines, RiskMetrics methodology
        - **Industry Standards**: Basel III, Solvency II frameworks
        - **Best Practices**: Risk management guidelines from major financial institutions
        
        ### Platform Features
        - **Real-time Updates**: All tabs update when parameters change
        - **Data Persistence**: Generated data persists across model changes
        - **Export Options**: CSV, JSON formats for further analysis
        - **Visualization**: Interactive charts with Plotly
        
        ### Technical Specifications
        - **Built with**: Streamlit, Python, Plotly
        - **Libraries**: NumPy, Pandas, SciPy, ARCH, yfinance
        - **Models**: Industry-standard implementations
        - **Validation**: Comprehensive backtesting and stress testing
        """)

# Footer
st.markdown("---")
st.markdown("🔬 **VaR & Risk Analytics Platform** | Built with Streamlit | © 2024")