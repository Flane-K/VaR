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
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

# Function to generate synthetic data
def generate_synthetic_data(num_days=500, initial_price=100, annual_return=0.08, annual_volatility=0.20):
    """Generate synthetic stock data with realistic characteristics"""
    np.random.seed(42)
    
    # Daily parameters
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)
    
    # Generate returns using geometric Brownian motion
    returns = np.random.normal(daily_return, daily_volatility, num_days)
    
    # Generate price series
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create date index
    start_date = datetime.now() - timedelta(days=num_days)
    dates = pd.date_range(start=start_date, periods=num_days+1, freq='D')
    
    # Create DataFrame
    df = pd.DataFrame({
        'Price': prices
    }, index=dates)
    
    return df

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
    
    # Initialize variables
    symbols_list = []
    crypto_symbols = []
    
    if data_source == "Live Market Data":
        if portfolio_type == "Single Asset":
            symbols = st.text_input("Enter symbol", "AAPL")
            symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        elif portfolio_type == "Multi-Asset":
            symbols = st.text_input("Enter symbols (comma-separated)", "AAPL,GOOGL,MSFT,TSLA")
            symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            
            # Crypto section for multi-asset
            st.markdown("**Crypto Assets (Optional)**")
            crypto_input = st.text_input("Enter crypto symbols (e.g., BTC-USD,ETH-USD)", "")
            if crypto_input.strip():
                crypto_symbols = [s.strip().upper() for s in crypto_input.split(",") if s.strip()]
                # Add crypto symbols to main symbols list
                symbols_list.extend(crypto_symbols)
        elif portfolio_type == "Crypto Portfolio":
            crypto_symbols_input = st.text_input("Enter crypto symbols", "BTC-USD")
            symbols_list = [s.strip().upper() for s in crypto_symbols_input.split(",") if s.strip()]
        elif portfolio_type == "Options Portfolio":
            st.info("Options data will be configured in the Options VaR tab")
            underlying_symbol = st.text_input("Underlying Asset", "AAPL")
            symbols_list = [underlying_symbol.strip().upper()] if underlying_symbol.strip() else []
            
    elif data_source == "CSV/XLSX Upload":
        uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=["csv", "xlsx"])
        st.markdown("""
        **File Format Requirements:**
        - First column: Date (YYYY-MM-DD format)
        - Subsequent columns: Asset prices
        - Header row with asset names
        - No missing values in price data
        """)
        
    elif data_source == "Manual Entry":
        st.subheader("Manual Data Entry")
        
        # Option to use default data or generate custom
        use_default = st.selectbox(
            "Data Generation",
            ["Use Default Data", "Generate Custom Data"]
        )
        
        if use_default == "Use Default Data":
            # Generate default synthetic data
            default_data = generate_synthetic_data()
            manual_data_input = "\n".join([f"{date.strftime('%Y-%m-%d')},{price:.2f}" 
                                         for date, price in zip(default_data.index, default_data['Price'])])
            st.text_area("Generated Data (Date, Price per line)", manual_data_input, height=150, disabled=True)
        else:
            # Custom data generation parameters
            st.markdown("**Custom Data Parameters**")
            num_days = st.number_input("Number of Days", 100, 1000, 500)
            initial_price = st.number_input("Initial Price ($)", 10.0, 1000.0, 100.0)
            annual_return = st.slider("Annual Return (%)", -20, 30, 8) / 100
            annual_volatility = st.slider("Annual Volatility (%)", 5, 50, 20) / 100
            
            if st.button("Generate Data"):
                custom_data = generate_synthetic_data(num_days, initial_price, annual_return, annual_volatility)
                manual_data_input = "\n".join([f"{date.strftime('%Y-%m-%d')},{price:.2f}" 
                                             for date, price in zip(custom_data.index, custom_data['Price'])])
            else:
                manual_data_input = "2023-01-01,100\n2023-01-02,101\n2023-01-03,102"
            
            st.text_area("Manual Data Entry (Date, Price per line)", manual_data_input, height=150, key="manual_input")
    
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
    
    # Initialize variables with defaults
    num_simulations = 10000
    garch_p = 1
    garch_q = 1
    
    if var_model == "Monte Carlo":
        num_simulations = st.number_input("Number of Simulations", 1000, 100000, 10000)
    
    if var_model == "GARCH-Based":
        garch_p = st.number_input("GARCH P", 1, 5, 1)
        garch_q = st.number_input("GARCH Q", 1, 5, 1)
    
    # Date Range - Set default to 2 years back
    st.markdown('<div class="sidebar-header">📅 Date Range</div>', unsafe_allow_html=True)
    end_date = st.date_input("End Date", datetime.now())
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=730))  # 2 years back
    
    # Advanced Settings
    st.markdown('<div class="sidebar-header">🔧 Advanced Settings</div>', unsafe_allow_html=True)
    decay_factor = st.slider("Decay Factor", 0.9, 0.99, 0.94)
    cornish_fisher = st.checkbox("Apply Cornish-Fisher Adjustment")
    
    # Portfolio Weights (for multi-asset)
    weights = {}
    if portfolio_type in ["Multi-Asset", "Crypto Portfolio"] and data_source == "Live Market Data" and symbols_list:
        st.markdown('<div class="sidebar-header">⚖️ Portfolio Weights</div>', unsafe_allow_html=True)
        for symbol in symbols_list:
            weights[symbol] = st.slider(f"{symbol} Weight", 0.0, 1.0, 1.0/len(symbols_list), key=f"weight_{symbol}")
        
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
    "❓ Help & Guide"
])

# Load data based on source
@st.cache_data(ttl=3600)
def load_market_data(symbols_list, start_date, end_date):
    if not symbols_list:
        return None
    try:
        data = yf.download(symbols_list, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.info(f"No data found for symbols: {', '.join(symbols_list)} in the specified date range.")
            return None

        # Handle both single and multiple symbols
        if len(symbols_list) == 1:
            if 'Adj Close' in data.columns:
                return data[['Adj Close']].rename(columns={'Adj Close': symbols_list[0]})
            elif 'Close' in data.columns:
                st.warning(f"'Adj Close' not found for {symbols_list[0]}. Using 'Close' prices instead.")
                return data[['Close']].rename(columns={'Close': symbols_list[0]})
            else:
                st.error(f"Neither 'Adj Close' nor 'Close' found for {symbols_list[0]}")
                return None
        else:
            try:
                if 'Adj Close' in data.columns.get_level_values(0):
                    return data['Adj Close']
                elif 'Close' in data.columns.get_level_values(0):
                    st.warning(f"'Adj Close' not found. Using 'Close' prices instead.")
                    return data['Close']
                else:
                    available_columns = data.columns.get_level_values(0).unique().tolist()
                    st.error(f"Neither 'Adj Close' nor 'Close' found. Available columns: {available_columns}")
                    return None
            except AttributeError:
                st.error("Unexpected data structure from yfinance")
                return None
                
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data from Yahoo Finance: {str(e)}")
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
        st.error(f"Error parsing manual data: {str(e)}. Please ensure format is 'YYYY-MM-DD,Price'.")
        return None

# Initialize variables
market_data = None
returns = None
portfolio_returns = None

# Data Loading Logic
if data_source == "Live Market Data":
    if symbols_list:
        market_data = load_market_data(symbols_list, start_date, end_date)
        if market_data is not None and not market_data.empty:
            returns = market_data.pct_change().dropna()
            
            if portfolio_type in ["Multi-Asset", "Crypto Portfolio"]:
                if all(col in returns.columns for col in weights.keys()):
                    aligned_weights = pd.Series(weights).reindex(returns.columns, fill_value=0).values
                    portfolio_returns = returns.dot(aligned_weights)
                else:
                    st.warning("Using equal weights for available data.")
                    portfolio_returns = returns.mean(axis=1)
            elif portfolio_type == "Single Asset":
                portfolio_returns = returns.iloc[:, 0] if len(returns.columns) > 0 else None
            else:
                portfolio_returns = None
        else:
            st.info("No market data fetched. Check symbols or date range.")
    else:
        st.info("Please enter symbols for Live Market Data.")

elif data_source == "CSV/XLSX Upload":
    if 'uploaded_file' in locals() and uploaded_file is not None:
        uploaded_df = load_uploaded_file_data(uploaded_file)
        if uploaded_df is not None and not uploaded_df.empty:
            market_data = uploaded_df
            returns = market_data.pct_change().dropna()
            
            if portfolio_type in ["Multi-Asset", "Crypto Portfolio"] and len(market_data.columns) > 1:
                portfolio_returns = returns.mean(axis=1)
            else:
                portfolio_returns = returns.iloc[:, 0] if len(returns.columns) > 0 else None
        else:
            st.info("Uploaded file processed, but no valid data found.")
    else:
        st.info("Please upload a CSV/XLSX file.")

elif data_source == "Manual Entry":
    if use_default == "Use Default Data":
        manual_df = generate_synthetic_data()
    else:
        manual_df = load_manual_data(st.session_state.get('manual_input', manual_data_input))
    
    if manual_df is not None and not manual_df.empty:
        market_data = manual_df
        returns = market_data.pct_change().dropna()
        portfolio_returns = returns.iloc[:, 0] if len(returns.columns) > 0 else None
    else:
        st.info("No valid manual data entered or parsed.")

# Dashboard Tab
with tab1:
    st.header("📊 Risk Dashboard")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_var = st.session_state.var_engines.calculate_parametric_var(
                portfolio_returns, confidence_level, time_horizon, cornish_fisher
            )
            st.metric("VaR (95%)", f"${current_var:,.2f}", delta=None)
        
        with col2:
            expected_shortfall = st.session_state.var_engines.calculate_expected_shortfall(
                portfolio_returns, confidence_level
            )
            st.metric("Expected Shortfall", f"${expected_shortfall:,.2f}", delta=None)
        
        with col3:
            volatility = portfolio_returns.std() * np.sqrt(252) * 100
            st.metric("Annual Volatility", f"{volatility:.2f}%", delta=None)
        
        with col4:
            risk_free_rate_annual = 0.02
            excess_returns = portfolio_returns.mean() * 252 - risk_free_rate_annual
            sharpe_ratio = excess_returns / (portfolio_returns.std() * np.sqrt(252))
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}", delta=None)
        
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
        
        # Returns Distribution and Rolling Volatility
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Returns Distribution")
            fig = px.histogram(
                x=portfolio_returns.values,
                nbins=50,
                title="Daily Returns Distribution",
                template="plotly_dark"
            )
            fig.update_traces(marker_color='#ff6b6b')
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
            
            var_results = {}
            calculated_var_value = None
            
            if var_model == "Parametric (Delta-Normal)":
                calculated_var_value = st.session_state.var_engines.calculate_parametric_var(
                    portfolio_returns, confidence_level, time_horizon, cornish_fisher
                )
                var_results['Parametric'] = calculated_var_value
            elif var_model == "Historical Simulation":
                calculated_var_value = st.session_state.var_engines.calculate_historical_var(
                    portfolio_returns, confidence_level, time_horizon
                )
                var_results['Historical'] = calculated_var_value
            elif var_model == "Monte Carlo":
                calculated_var_value = st.session_state.var_engines.calculate_monte_carlo_var(
                    portfolio_returns, confidence_level, time_horizon, num_simulations
                )
                var_results['Monte Carlo'] = calculated_var_value
            elif var_model == "GARCH-Based":
                calculated_var_value = st.session_state.var_engines.calculate_garch_var(
                    portfolio_returns, confidence_level, time_horizon, garch_p, garch_q
                )
                var_results['GARCH'] = calculated_var_value
            elif var_model == "Extreme Value Theory":
                calculated_var_value = st.session_state.var_engines.calculate_evt_var(
                    portfolio_returns, confidence_level
                )
                var_results['EVT'] = calculated_var_value

            for method, var_value in var_results.items():
                st.metric(f"{method} VaR", f"${var_value:,.2f}")
            
            es_value = st.session_state.var_engines.calculate_expected_shortfall(
                portfolio_returns, confidence_level
            )
            st.metric("Expected Shortfall", f"${es_value:,.2f}")
        
        with col2:
            if calculated_var_value is not None:
                st.subheader("📈 VaR Visualization")
                fig = st.session_state.visualization.plot_var_distribution(
                    portfolio_returns, confidence_level, calculated_var_value
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Model Comparison
        st.subheader("🔍 VaR Model Comparison")
        all_var_results = {}
        
        try:
            all_var_results['Parametric'] = st.session_state.var_engines.calculate_parametric_var(
                portfolio_returns, confidence_level, time_horizon, False
            )
            all_var_results['Historical'] = st.session_state.var_engines.calculate_historical_var(
                portfolio_returns, confidence_level, time_horizon
            )
            all_var_results['Monte Carlo'] = st.session_state.var_engines.calculate_monte_carlo_var(
                portfolio_returns, confidence_level, time_horizon, 10000
            )
            all_var_results['GARCH'] = st.session_state.var_engines.calculate_garch_var(
                portfolio_returns, confidence_level, time_horizon, garch_p, garch_q
            )
            all_var_results['EVT'] = st.session_state.var_engines.calculate_evt_var(
                portfolio_returns, confidence_level
            )
            
            comparison_df = pd.DataFrame(list(all_var_results.items()), columns=['Method', 'VaR'])
            st.dataframe(comparison_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Error in model comparison: {e}")
    else:
        st.info("Please load market data first to calculate VaR.")

# Backtesting Tab
with tab3:
    st.header("🧪 Backtesting & Validation")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_window = st.number_input("Backtesting Window (days)", 100, 1000, 252)
            var_method = st.selectbox("VaR Method for Backtesting", 
                                    ["Parametric", "Historical", "Monte Carlo"])
        
        with col2:
            st.subheader("📊 Backtesting Results")
            
            if len(portfolio_returns) < backtest_window + 50:
                st.warning(f"Insufficient data for backtesting. Need at least {backtest_window + 50} data points, but only have {len(portfolio_returns)}.")
            else:
                try:
                    if var_method == "Parametric":
                        var_func = lambda ret, conf, horizon: st.session_state.var_engines.calculate_parametric_var(ret, conf, horizon, False)
                    elif var_method == "Historical":
                        var_func = st.session_state.var_engines.calculate_historical_var
                    elif var_method == "Monte Carlo":
                        var_func = lambda ret, conf, horizon: st.session_state.var_engines.calculate_monte_carlo_var(ret, conf, horizon, 10000)

                    backtest_results = st.session_state.backtesting.perform_backtesting(
                        portfolio_returns, confidence_level, backtest_window, var_func
                    )
                    
                    if backtest_results and 'kupiec_pvalue' in backtest_results:
                        st.metric("Kupiec Test p-value", f"{backtest_results['kupiec_pvalue']:.4f}")
                        st.metric("Actual Violations", f"{backtest_results['violations']}")
                        st.metric("Expected Violations", f"{backtest_results['expected_violations']:.1f}")
                        
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
                        
                        st.subheader("📈 VaR Violations Over Time")
                        fig = st.session_state.visualization.plot_var_violations(
                            portfolio_returns, backtest_results['var_estimates'], backtest_results['violations_dates']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during backtesting: {e}")
    else:
        st.info("Please load market data first to perform backtesting.")

# Scenario & Stress Tab
with tab4:
    st.header("⚡ Scenario & Stress Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Historical Scenarios")
        scenario_type = st.selectbox(
            "Select Historical Scenario",
            ["2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash", "Custom Scenario"]
        )
        
        vol_shock = 0
        corr_shock = 0.0
        spot_shock = 0
        
        if scenario_type == "Custom Scenario":
            vol_shock = st.slider("Volatility Shock (%)", -50, 200, 0)
            corr_shock = st.slider("Correlation Shock", -0.5, 0.5, 0.0)
            spot_shock = st.slider("Spot Price Shock (%)", -50, 50, 0)
    
    with col2:
        st.subheader("📈 Stress Test Results")
        
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
                    st.error("Stress testing failed to return valid results.")
            except Exception as e:
                st.error(f"Error during stress testing: {e}")
        else:
            st.info("Please load market data first to perform stress testing.")
    
    # Scenario comparison
    if portfolio_returns is not None and not portfolio_returns.empty:
        st.subheader("📊 Scenario Comparison")
        
        scenarios = ["Normal", "2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash"]
        scenario_vars = []
        
        for scenario in scenarios:
            try:
                if scenario == "Normal":
                    var_val = st.session_state.var_engines.calculate_parametric_var(
                        portfolio_returns, confidence_level, time_horizon, cornish_fisher
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
                title="VaR Across Different Scenarios",
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
        
        if analysis_type == "Rolling VaR":
            st.subheader("📊 Rolling VaR Analysis")
            rolling_var = st.session_state.rolling_analysis.calculate_rolling_var(
                portfolio_returns, confidence_level, rolling_window
            )
            
            if not rolling_var.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rolling_var.index,
                    y=rolling_var.values,
                    mode='lines',
                    name=f'Rolling {rolling_window}-Day VaR',
                    line=dict(color='#4ecdc4', width=2)
                ))
                
                fig.update_layout(
                    title=f"Rolling {rolling_window}-Day VaR",
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
        
        # Correlation heatmap for multi-asset portfolios
        if portfolio_type in ["Multi-Asset", "Crypto Portfolio"] and returns is not None and len(returns.columns) > 1:
            st.subheader("🔥 Correlation Heatmap")
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
        st.info("Please load market data first to perform rolling analysis.")

# Options VaR Tab
with tab6:
    st.header("📊 Options Portfolio VaR")
    
    if portfolio_type == "Options Portfolio":
        # Default options parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            spot_price = st.number_input("Spot Price ($)", 50.0, 5000.0, 100.0)
            strike_price = st.number_input("Strike Price ($)", 50.0, 5000.0, 105.0)
            
        with col2:
            time_to_expiry = st.number_input("Time to Expiry (days)", 1, 365, 30) / 365
            risk_free_rate = st.slider("Risk-free Rate (%)", 0.0, 10.0, 2.0) / 100
            
        with col3:
            volatility_input = st.slider("Volatility (%)", 10, 100, 25) / 100
            option_type = st.selectbox("Option Type", ["Call", "Put"])
        
        options_var_method = st.selectbox(
            "Options VaR Method",
            ["Delta-Normal", "Delta-Gamma", "Full Revaluation Monte Carlo"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Options VaR Results")
            
            try:
                options_var_result = st.session_state.options_var.calculate_options_var(
                    spot_price, strike_price, time_to_expiry, risk_free_rate, 
                    volatility_input, option_type, options_var_method, confidence_level
                )
                
                if options_var_result:
                    st.metric("Options VaR", f"${options_var_result['var']:,.2f}")
                    st.metric("Delta", f"{options_var_result['delta']:.4f}")
                    st.metric("Gamma", f"{options_var_result['gamma']:.4f}")
                    st.metric("Theta", f"{options_var_result['theta']:.4f}")
                    st.metric("Vega", f"{options_var_result['vega']:.4f}")
            except Exception as e:
                st.error(f"Error calculating options VaR: {e}")
        
        with col2:
            st.subheader("📈 Greeks Sensitivity")
            
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
        
        with col2:
            st.subheader("📈 Quick Metrics Export")
            
            current_parametric_var = st.session_state.var_engines.calculate_parametric_var(portfolio_returns, 0.95, 1, cornish_fisher)
            current_es = st.session_state.var_engines.calculate_expected_shortfall(portfolio_returns, 0.95)

            summary_metrics = {
                'VaR (95%)': current_parametric_var,
                'Expected Shortfall': current_es,
                'Volatility (%)': portfolio_returns.std() * np.sqrt(252) * 100,
                'Sharpe Ratio': (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252)),
                'Skewness': portfolio_returns.skew(),
                'Kurtosis': portfolio_returns.kurtosis()
            }
            
            metrics_df = pd.DataFrame(list(summary_metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(metrics_df, use_container_width=True)
        
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                if export_format == "CSV":
                    csv_data = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV Report",
                        data=csv_data,
                        file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                elif export_format == "JSON":
                    import json
                    json_serializable_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in summary_metrics.items()}
                    json_data = json.dumps(json_serializable_metrics, indent=2)
                    st.download_button(
                        label="Download JSON Report",
                        data=json_data,
                        file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                st.success("✅ Report generated successfully!")
    else:
        st.info("Please load data first to generate reports.")

# Help & Guide Tab
with tab8:
    st.header("❓ Help & User Guide")
    
    # Platform Overview
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.subheader("🏠 Platform Overview")
    st.markdown("""
    The VaR & Risk Analytics Platform is a comprehensive tool for financial risk assessment and portfolio analysis. 
    It provides multiple VaR calculation methods, backtesting capabilities, stress testing, and detailed risk analytics.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Getting Started
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.subheader("🚀 Getting Started")
    st.markdown("""
    **Step 1: Choose Data Source**
    - **Live Market Data**: Enter stock symbols (e.g., AAPL, GOOGL) or crypto symbols (e.g., BTC-USD, ETH-USD)
    - **CSV/XLSX Upload**: Upload your own price data
    - **Manual Entry**: Use synthetic data or enter custom data points
    
    **Step 2: Select Portfolio Type**
    - **Single Asset**: Analyze one security
    - **Multi-Asset**: Analyze a portfolio of multiple securities
    - **Crypto Portfolio**: Focus on cryptocurrency analysis
    - **Options Portfolio**: Analyze options positions
    
    **Step 3: Configure Risk Parameters**
    - Set confidence level (90-99%)
    - Choose time horizon (1-30 days)
    - Select VaR calculation method
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Input Formats
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.subheader("📊 Data Input Formats")
    
    st.markdown("**CSV/XLSX File Format:**")
    st.code("""
Date,AAPL,GOOGL,MSFT
2023-01-01,150.25,95.30,245.50
2023-01-02,151.10,96.15,246.75
2023-01-03,149.80,94.85,244.20
    """)
    
    st.markdown("**Manual Entry Format:**")
    st.code("""
2023-01-01,100.00
2023-01-02,101.50
2023-01-03,99.75
    """)
    
    st.markdown("**Requirements:**")
    st.markdown("""
    - First column must be dates in YYYY-MM-DD format
    - Subsequent columns should contain price data
    - No missing values in price columns
    - At least 100 data points recommended for reliable analysis
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # VaR Methods Explained
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.subheader("📈 VaR Calculation Methods")
    
    st.markdown("**Parametric (Delta-Normal)**")
    st.markdown("- Assumes normal distribution of returns")
    st.markdown("- Fast calculation, suitable for linear portfolios")
    st.markdown("- May underestimate tail risks")
    
    st.markdown("**Historical Simulation**")
    st.markdown("- Uses actual historical return distribution")
    st.markdown("- No distributional assumptions")
    st.markdown("- Requires sufficient historical data")
    
    st.markdown("**Monte Carlo**")
    st.markdown("- Simulates future price paths")
    st.markdown("- Flexible for complex portfolios")
    st.markdown("- Computationally intensive")
    
    st.markdown("**GARCH-Based**")
    st.markdown("- Models time-varying volatility")
    st.markdown("- Good for volatile markets")
    st.markdown("- Requires parameter estimation")
    
    st.markdown("**Extreme Value Theory (EVT)**")
    st.markdown("- Focuses on tail events")
    st.markdown("- Better for extreme risk assessment")
    st.markdown("- Requires sufficient extreme observations")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Parameter Guidelines
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.subheader("⚙️ Parameter Guidelines")
    
    st.markdown("**Confidence Level:**")
    st.markdown("- 95%: Standard for daily risk management")
    st.markdown("- 99%: Regulatory requirements (Basel)")
    st.markdown("- 99.9%: Extreme risk assessment")
    
    st.markdown("**Time Horizon:**")
    st.markdown("- 1 day: Daily trading risk")
    st.markdown("- 10 days: Regulatory standard")
    st.markdown("- 30 days: Monthly risk assessment")
    
    st.markdown("**Historical Window:**")
    st.markdown("- 252 days: One year of trading data")
    st.markdown("- 500+ days: More stable estimates")
    st.markdown("- Shorter windows: More responsive to recent changes")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interpretation Guide
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.subheader("📋 Results Interpretation")
    
    st.markdown("**VaR Interpretation:**")
    st.markdown("- VaR of $10,000 at 95% confidence means:")
    st.markdown("- 95% chance losses will not exceed $10,000")
    st.markdown("- 5% chance losses will exceed $10,000")
    
    st.markdown("**Expected Shortfall (ES):**")
    st.markdown("- Average loss when VaR is exceeded")
    st.markdown("- Always higher than VaR")
    st.markdown("- Better measure of tail risk")
    
    st.markdown("**Backtesting Results:**")
    st.markdown("- Green Zone: Model is adequate")
    st.markdown("- Yellow Zone: Model needs attention")
    st.markdown("- Red Zone: Model requires immediate review")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Troubleshooting
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.subheader("🔧 Troubleshooting")
    
    st.markdown("**Common Issues:**")
    st.markdown("- **Insufficient Data**: Ensure at least 100+ data points")
    st.markdown("- **Missing Prices**: Check for gaps in price data")
    st.markdown("- **Symbol Errors**: Verify ticker symbols are correct")
    st.markdown("- **Date Format**: Use YYYY-MM-DD format")
    
    st.markdown("**Performance Tips:**")
    st.markdown("- Use shorter time series for faster calculations")
    st.markdown("- Reduce Monte Carlo simulations if needed")
    st.markdown("- Cache is enabled for market data (1 hour)")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("🔬 **VaR & Risk Analytics Platform** | Built with Streamlit | © 2024")