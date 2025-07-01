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

# Import custom modules (assuming these exist and work correctly)
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
        ["Single Asset", "Multi-Asset", "Options Portfolio"],
        key="portfolio_type"
    )
    
    symbols_list = [] # Initialize symbols_list
    if data_source == "Live Market Data":
        symbols = st.text_input("Enter symbols (comma-separated)", "AAPL,GOOGL,MSFT,TSLA")
        symbols_list = [s.strip().upper() for s in symbols.split(",")]
    elif data_source == "CSV/XLSX Upload":
        uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=["csv", "xlsx"])
    elif data_source == "Manual Entry":
        st.subheader("Manual Data Entry")
        manual_data_input = st.text_area("Enter historical prices (Date, Price per line)",
                                        "2023-01-01,100\n2023-01-02,101\n2023-01-03,102")
    
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
    
    if var_model == "Monte Carlo":
        num_simulations = st.number_input("Number of Simulations", 1000, 100000, 10000)
    
    if var_model == "GARCH-Based":
        garch_p = st.number_input("GARCH P", 1, 5, 1)
        garch_q = st.number_input("GARCH Q", 1, 5, 1)
    
    # Date Range
    st.markdown('<div class="sidebar-header">📅 Date Range</div>', unsafe_allow_html=True)
    end_date = st.date_input("End Date", datetime.now())
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    
    # Advanced Settings
    st.markdown('<div class="sidebar-header">🔧 Advanced Settings</div>', unsafe_allow_html=True)
    decay_factor = st.slider("Decay Factor", 0.9, 0.99, 0.94)
    cornish_fisher = st.checkbox("Apply Cornish-Fisher Adjustment")
    
    # Portfolio Weights (for multi-asset)
    weights = {} # Initialize weights
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
            weights = {s: 1.0/len(symbols_list) for s in symbols_list} # Default to equal weights if total is 0

# Main Content Area
st.title("📊 VaR & Risk Analytics Platform")

# Tab Navigation
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Dashboard", 
    "🧮 VaR Calculator", 
    "🧪 Backtest & Validate", 
    "⚡ Scenario & Stress", 
    "📈 Rolling Analysis", 
    "📊 Option VaR", 
    "📄 Reports & Exports"
])

# Load data based on source
@st.cache_data(ttl=3600) # Cache for 1 hour to reduce API calls
def load_market_data(symbols_list, start_date, end_date):
    if not symbols_list:
        return None
    try:
        data = yf.download(symbols_list, start=start_date, end=end_date)

        if data.empty:
            st.info(f"No data found for symbols: {', '.join(symbols_list)} in the specified date range.")
            return None

        # Attempt to get 'Adj Close' first
        try:
            if isinstance(data.columns, pd.MultiIndex):
                # For multiple symbols, data['Adj Close'] returns a DataFrame with symbols as columns
                return data['Adj Close']
            else:
                # For a single symbol, data['Adj Close'] returns a Series, convert to DataFrame
                return data[['Adj Close']]
        except KeyError:
            # If 'Adj Close' not found, try 'Close' as a fallback
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    st.warning(f"'Adj Close' not found for {', '.join(symbols_list)}. Using 'Close' prices instead.")
                    return data['Close']
                else:
                    st.warning(f"'Adj Close' not found for {', '.join(symbols_list)}. Using 'Close' prices instead.")
                    return data[['Close']] # Ensure it's a DataFrame
            except KeyError:
                # If neither is found
                available_columns = data.columns.get_level_values(0).unique().tolist() if isinstance(data.columns, pd.MultiIndex) else data.columns.tolist()
                st.error(f"Error: Neither 'Adj Close' nor 'Close' columns found in data for {', '.join(symbols_list)}. Available top-level columns: {available_columns}")
                return None
    except Exception as e:
        # Catch broader errors from yf.download or initial processing
        st.error(f"An unexpected error occurred while loading data from Yahoo Finance for {', '.join(symbols_list)}: {str(e)}")
        return None

# Function to load data from uploaded file
@st.cache_data
def load_uploaded_file_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        else: # Assuming .xlsx
            df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
        
        # Ensure the DataFrame has a numeric column for prices
        # Assuming the first numeric column after date is the price
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            return df[numeric_cols[0]]
        else:
            st.error("Uploaded file does not contain numeric price data.")
            return None
    except Exception as e:
        st.error(f"Error reading uploaded file: {str(e)}")
        return None

# Function to load data from manual entry
@st.cache_data
def load_manual_data(manual_data_input):
    try:
        data = [line.split(',') for line in manual_data_input.strip().split('\n') if line.strip()]
        df = pd.DataFrame(data, columns=['Date', 'Price'])
        df['Date'] = pd.to_datetime(df['Date'])
        df['Price'] = pd.to_numeric(df['Price'])
        df = df.set_index('Date')['Price'] # Return a Series for consistency
        return df
    except Exception as e:
        st.error(f"Error parsing manual data: {str(e)}. Please ensure format is 'YYYY-MM-DD,Price'.")
        return None

# Initialize variables to avoid NameError
market_data = None
returns = None
portfolio_returns = None

# --- Data Loading Logic ---
if data_source == "Live Market Data":
    if symbols_list:
        market_data = load_market_data(symbols_list, start_date, end_date)
        if market_data is not None and not market_data.empty:
            returns = market_data.pct_change().dropna()
            
            if portfolio_type == "Multi-Asset":
                # Ensure weights match the columns in returns
                if all(col in returns.columns for col in weights.keys()):
                    # Align weights to the returns DataFrame columns
                    aligned_weights = pd.Series(weights).reindex(returns.columns, fill_value=0).values
                    portfolio_returns = returns.dot(aligned_weights)
                else:
                    st.warning("Symbols in portfolio weights do not fully match fetched data columns. Using equal weights for available data.")
                    # Fallback to equal weights for available data
                    available_symbols = [s for s in symbols_list if s in returns.columns]
                    if available_symbols:
                        equal_weight = 1.0 / len(available_symbols)
                        portfolio_returns = returns[available_symbols].mean(axis=1) # Simple average for now
                    else:
                        st.warning("No matching symbols found in fetched data for portfolio construction.")
                        portfolio_returns = None # No portfolio returns can be calculated
            elif portfolio_type == "Single Asset":
                if len(returns.columns) > 0:
                    portfolio_returns = returns.iloc[:, 0] # Take the first asset's returns
                else:
                    st.warning("No data available for single asset portfolio.")
                    portfolio_returns = None
            else: # Options Portfolio, data handling will be different
                portfolio_returns = None # Not directly using market data returns for options VaR
        else:
            st.info("No market data fetched. Check symbols or date range.")
            market_data = None # Reset to None if fetching failed
    else:
        st.info("Please enter symbols for Live Market Data.")

elif data_source == "CSV/XLSX Upload":
    if 'uploaded_file' in locals() and uploaded_file is not None:
        uploaded_df = load_uploaded_file_data(uploaded_file)
        if uploaded_df is not None and not uploaded_df.empty:
            # Assuming uploaded_df is a Series of prices for a single asset or a DataFrame for multiple
            if isinstance(uploaded_df, pd.Series):
                market_data = pd.DataFrame(uploaded_df) # Convert to DataFrame for consistency
                portfolio_returns = uploaded_df.pct_change().dropna()
            elif isinstance(uploaded_df, pd.DataFrame):
                market_data = uploaded_df
                # For multi-column upload, you'd need logic to select columns and apply weights
                # For simplicity, assuming the first column is the main asset for returns calculation
                portfolio_returns = uploaded_df.iloc[:, 0].pct_change().dropna() 
            returns = market_data.pct_change().dropna() # Calculate returns for all columns
            
            if portfolio_type == "Multi-Asset" and isinstance(market_data, pd.DataFrame):
                st.warning("Multi-asset CSV/XLSX upload requires manual weight definition or a specific column selection logic.")
                st.info("Currently, only the first column of your uploaded data is used for portfolio returns calculation.")
                # You would need to extend this logic to allow users to select columns and define weights
                # For now, if multi-asset selected, we'll just use the first column's returns.
                # A more robust solution would involve letting the user specify which columns are assets and their weights.
                if not market_data.empty and len(market_data.columns) > 0:
                    portfolio_returns = market_data.iloc[:, 0].pct_change().dropna()
                else:
                    portfolio_returns = None
            elif portfolio_type == "Single Asset" and isinstance(market_data, pd.DataFrame):
                if not market_data.empty and len(market_data.columns) > 0:
                    portfolio_returns = market_data.iloc[:, 0].pct_change().dropna()
                else:
                    portfolio_returns = None
            else: # Options Portfolio, data handling will be different
                portfolio_returns = None
        else:
            st.info("Uploaded file processed, but no valid data found.")
    else:
        st.info("Please upload a CSV/XLSX file.")

elif data_source == "Manual Entry":
    if 'manual_data_input' in locals() and manual_data_input:
        manual_df_series = load_manual_data(manual_data_input)
        if manual_df_series is not None and not manual_df_series.empty:
            market_data = pd.DataFrame(manual_df_series) # Convert to DataFrame
            portfolio_returns = manual_df_series.pct_change().dropna()
            returns = market_data.pct_change().dropna() # For consistency, even if single column
        else:
            st.info("No valid manual data entered or parsed.")
    else:
        st.info("Please enter data manually.")

# Dashboard Tab
with tab1:
    st.header("📊 Risk Dashboard")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_var = st.session_state.var_engines.calculate_parametric_var(
                portfolio_returns, confidence_level, time_horizon, cornish_fisher # Pass cornish_fisher
            )
            st.metric("VaR (95%)", f"${current_var:,.2f}", delta=None)
        
        with col2:
            expected_shortfall = st.session_state.var_engines.calculate_expected_shortfall(
                portfolio_returns, confidence_level
            )
            st.metric("Expected Shortfall", f"${expected_shortfall:,.2f}", delta=None)
        
        with col3:
            # Ensure annualization is correct for daily returns
            volatility = portfolio_returns.std() * np.sqrt(252) * 100
            st.metric("Annual Volatility", f"{volatility:.2f}%", delta=None)
        
        with col4:
            # Ensure annualization is correct for daily returns
            risk_free_rate_annual = 0.02 # Assuming a default risk-free rate for Sharpe
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
        
        # Returns Distribution
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
        st.info("Please load data (Live Market Data, CSV/XLSX, or Manual Entry) to view the dashboard.")

# VaR Calculator Tab
with tab2:
    st.header("🧮 VaR Calculator")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 VaR Calculation Results")
            
            # Calculate VaR using selected method
            var_results = {}
            calculated_var_value = None # To hold the specific VaR value for visualization
            
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
                calculated_var_value = st.session_state.var_engines.calculate_evt_var( # Corrected method call
                    portfolio_returns, confidence_level
                )
                var_results['EVT'] = calculated_var_value

            # Display results
            for method, var_value in var_results.items():
                st.metric(f"{method} VaR", f"${var_value:,.2f}")
            
            # Expected Shortfall
            es_value = st.session_state.var_engines.calculate_expected_shortfall(
                portfolio_returns, confidence_level
            )
            st.metric("Expected Shortfall", f"${es_value:,.2f}")
        
        with col2:
            if calculated_var_value is not None:
                st.subheader("📈 VaR Visualization")
                
                # VaR visualization
                fig = st.session_state.visualization.plot_var_distribution(
                    portfolio_returns, confidence_level, calculated_var_value
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a VaR model and ensure data is loaded to see visualization.")
        
        # Model Comparison
        st.subheader("🔍 VaR Model Comparison")
        
        # Calculate VaR for all relevant models for comparison
        all_var_results = {}
        if not portfolio_returns.empty:
            try:
                all_var_results['Parametric'] = st.session_state.var_engines.calculate_parametric_var(
                    portfolio_returns, confidence_level, time_horizon, False # Default False for comparison
                )
            except Exception as e:
                all_var_results['Parametric'] = f"Error: {e}"
            try:
                all_var_results['Historical'] = st.session_state.var_engines.calculate_historical_var(
                    portfolio_returns, confidence_level, time_horizon
                )
            except Exception as e:
                all_var_results['Historical'] = f"Error: {e}"
            try:
                all_var_results['Monte Carlo'] = st.session_state.var_engines.calculate_monte_carlo_var(
                    portfolio_returns, confidence_level, time_horizon, 10000
                )
            except Exception as e:
                all_var_results['Monte Carlo'] = f"Error: {e}"
            try:
                all_var_results['GARCH'] = st.session_state.var_engines.calculate_garch_var( # Add GARCH to comparison
                    portfolio_returns, confidence_level, time_horizon, garch_p, garch_q
                )
            except Exception as e:
                all_var_results['GARCH'] = f"Error: {e}"
            try:
                all_var_results['EVT'] = st.session_state.var_engines.calculate_evt_var( # Add EVT to comparison
                    portfolio_returns, confidence_level
                )
            except Exception as e:
                all_var_results['EVT'] = f"Error: {e}"
        
            # Display comparison
            comparison_df = pd.DataFrame(list(all_var_results.items()), columns=['Method', 'VaR'])
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("No portfolio returns available for model comparison.")
    else:
        st.info("Please load market data first to calculate VaR.")

# Backtesting Tab
with tab3:
    st.header("🧪 Backtesting & Validation")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        
        # Backtesting parameters
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_window = st.number_input("Backtesting Window (days)", 100, 1000, 252)
            var_method = st.selectbox("VaR Method for Backtesting", 
                                    ["Parametric", "Historical", "Monte Carlo"])
        
        with col2:
            st.subheader("📊 Backtesting Results")
            
            # Perform backtesting
            try:
                # Ensure the backtesting function correctly handles the method names
                if var_method == "Parametric":
                    var_func = lambda ret, conf, horizon: st.session_state.var_engines.calculate_parametric_var(ret, conf, horizon, False) # Explicitly pass False for Cornish Fisher
                elif var_method == "Historical":
                    var_func = st.session_state.var_engines.calculate_historical_var
                elif var_method == "Monte Carlo":
                    var_func = st.session_state.var_engines.calculate_monte_carlo_var
                else:
                    st.error("Invalid VaR method selected for backtesting.")
                    var_func = None

                if var_func:
                    backtest_results = st.session_state.backtesting.perform_backtesting(
                        portfolio_returns, confidence_level, backtest_window, var_func
                    )
                    
                    # Display Kupiec test results
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
                    st.subheader("📈 VaR Violations Over Time")
                    fig = st.session_state.visualization.plot_var_violations(
                        portfolio_returns, backtest_results['var_estimates'], backtest_results['violations_dates']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error during backtesting: {e}. Please ensure sufficient data for the selected window.")
    else:
        st.info("Please load market data first to perform backtesting.")

# Scenario & Stress Tab
with tab4:
    st.header("⚡ Scenario & Stress Testing")
    
    # Scenario selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Historical Scenarios")
        scenario_type = st.selectbox(
            "Select Historical Scenario",
            ["2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash", "Custom Scenario"]
        )
        
        if scenario_type == "Custom Scenario":
            vol_shock = st.slider("Volatility Shock (%)", -50, 200, 0)
            corr_shock = st.slider("Correlation Shock", -0.5, 0.5, 0.0)
            spot_shock = st.slider("Spot Price Shock (%)", -50, 50, 0)
    
    with col2:
        st.subheader("📈 Stress Test Results")
        
        if portfolio_returns is not None and not portfolio_returns.empty:
            # Perform stress testing
            try:
                stress_results = st.session_state.stress_testing.run_stress_test(
                    portfolio_returns, scenario_type, confidence_level
                )
                
                st.metric("Stressed VaR", f"${stress_results['stressed_var']:,.2f}")
                st.metric("VaR Increase", f"{stress_results['var_increase']:.1f}%")
                st.metric("Worst Case Loss", f"${stress_results['worst_case']:,.2f}")
            except Exception as e:
                st.error(f"Error during stress testing: {e}")
        else:
            st.info("Please load market data first to perform stress testing.")
    
    # Scenario comparison
    if portfolio_returns is not None and not portfolio_returns.empty:
        st.subheader("📊 Scenario Comparison")
        
        scenarios = ["Normal", "2008 Crisis", "COVID-19", "Dot-com Crash"]
        scenario_vars = []
        
        for scenario in scenarios:
            try:
                if scenario == "Normal":
                    var_val = st.session_state.var_engines.calculate_parametric_var(
                        portfolio_returns, confidence_level, time_horizon, cornish_fisher # Use current Cornish Fisher setting
                    )
                else:
                    stress_result = st.session_state.stress_testing.run_stress_test(
                        portfolio_returns, scenario, confidence_level
                    )
                    var_val = stress_result['stressed_var']
                scenario_vars.append(var_val)
            except Exception as e:
                st.warning(f"Could not calculate VaR for {scenario} scenario: {e}")
                scenario_vars.append(0) # Append 0 or None for failed scenarios
        
        # Plot scenario comparison
        if any(v > 0 for v in scenario_vars): # Only plot if there's meaningful data
            fig = px.bar(
                x=scenarios,
                y=scenario_vars,
                title="VaR Across Different Scenarios",
                template="plotly_dark"
            )
            fig.update_traces(marker_color='#ff6b6b')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid scenario VaR data to plot.")

# Rolling Analysis Tab
with tab5:
    st.header("📈 Rolling Analysis")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        
        # Rolling analysis parameters
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
            st.subheader("📊 Rolling VaR Analysis")
            # Ensure calculate_rolling_var can take the cornish_fisher flag if needed, or default
            rolling_var = st.session_state.rolling_analysis.calculate_rolling_var(
                portfolio_returns, confidence_level, rolling_window
            )
            
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
                st.warning(f"Could not generate correlation heatmap: {e}. Ensure enough data for multi-asset correlation.")
        elif portfolio_type == "Multi-Asset":
             st.info("No returns data available for multi-asset correlation heatmap.")
        else:
            st.info("Select 'Multi-Asset' portfolio type to view Correlation Heatmap.")
    else:
        st.info("Please load market data first to perform rolling analysis.")

# Options VaR Tab
with tab6:
    st.header("📊 Options Portfolio VaR")
    
    if portfolio_type == "Options Portfolio":
        # Options parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            spot_price = st.number_input("Spot Price ($)", 50.0, 5000.0, 100.0)
            strike_price = st.number_input("Strike Price ($)", 50.0, 5000.0, 100.0)
            
        with col2:
            time_to_expiry = st.number_input("Time to Expiry (days)", 1, 365, 30) / 365
            risk_free_rate = st.slider("Risk-free Rate (%)", 0.0, 10.0, 2.0) / 100
            
        with col3:
            volatility_input = st.slider("Volatility (%)", 10, 100, 20) / 100
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
                st.error(f"Error calculating options VaR: {e}. Please check inputs.")
        
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
                st.error(f"Error plotting Greeks: {e}. Ensure inputs are valid.")
    else:
        st.info("Please select 'Options Portfolio' in the sidebar to access options VaR calculations.")

# Reports & Exports Tab
with tab7:
    st.header("📄 Reports & Exports")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        
        # Report generation options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Report Options")
            
            report_type = st.selectbox(
                "Report Type",
                ["Executive Summary", "Detailed Risk Report", "Backtesting Report", "Stress Testing Report"]
            )
            
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON"] # PDF and Excel generation requires more complex libraries and setup
            )
            
            include_charts = st.checkbox("Include Charts", value=True)
            include_data = st.checkbox("Include Raw Data", value=False)
        
        with col2:
            st.subheader("📈 Quick Metrics Export")
            
            # Generate summary metrics
            # Ensure all_var_results is defined, or calculate what's needed
            current_parametric_var = st.session_state.var_engines.calculate_parametric_var(portfolio_returns, 0.95, 1, cornish_fisher) # Pass cornish_fisher
            current_es = st.session_state.var_engines.calculate_expected_shortfall(portfolio_returns, 0.95)
            max_drawdown_val = st.session_state.rolling_analysis.calculate_drawdown(portfolio_returns).min() * 100 if not portfolio_returns.empty else 0

            summary_metrics = {
                'VaR (95%)': current_parametric_var,
                'Expected Shortfall': current_es,
                'Volatility (%)': portfolio_returns.std() * np.sqrt(252) * 100,
                'Sharpe Ratio': (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252)), # Assuming 2% risk-free
                'Skewness': portfolio_returns.skew(),
                'Kurtosis': portfolio_returns.kurtosis()
            }
            
            metrics_df = pd.DataFrame(list(summary_metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(metrics_df, use_container_width=True)
        
        # Generate and download report
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                # The actual report generation for PDF/Excel would be complex.
                # For this fix, only CSV and JSON are directly supported for download.
                
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
                    # Ensure values are JSON serializable (e.g., convert numpy types)
                    json_serializable_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in summary_metrics.items()}
                    json_data = json.dumps(json_serializable_metrics, indent=2)
                    st.download_button(
                        label="Download JSON Report",
                        data=json_data,
                        file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
                if market_data is not None and not market_data.empty: # Added .empty check
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

# Footer
st.markdown("---")
st.markdown("🔬 **VaR & Risk Analytics Platform** | Built with Streamlit | © 2024")
