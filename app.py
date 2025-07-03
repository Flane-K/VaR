import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import custom modules
from src.data_ingestion import DataIngestion
from src.var_engines import VaREngines
from src.backtesting import Backtesting
from src.stress_testing import StressTesting
from src.rolling_analysis import RollingAnalysis
from src.options_var import OptionsVaR
from src.visualization import Visualization
from src.utils import Utils

# Configure page
st.set_page_config(
    page_title="VaR & Risk Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data persistence
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'portfolio_returns' not in st.session_state:
    st.session_state.portfolio_returns = None
if 'var_results' not in st.session_state:
    st.session_state.var_results = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = "Parametric"

# Initialize classes
@st.cache_resource
def get_instances():
    return {
        'data_ingestion': DataIngestion(),
        'var_engines': VaREngines(),
        'backtesting': Backtesting(),
        'stress_testing': StressTesting(),
        'rolling_analysis': RollingAnalysis(),
        'options_var': OptionsVaR(),
        'visualization': Visualization(),
        'utils': Utils()
    }

instances = get_instances()

def safe_dataframe_display(df, title="Data"):
    """Safely display dataframes by ensuring proper data types"""
    try:
        if df is None or df.empty:
            st.warning(f"No {title.lower()} to display")
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
            elif np.issubdtype(display_df[col].dtype, np.number):
                # Ensure numeric columns are properly formatted
                if display_df[col].dtype in ['float64', 'float32']:
                    display_df[col] = display_df[col].round(6)
        
        # Display with proper formatting
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying {title}: {str(e)}")
        # Fallback to simple text display
        st.text(str(df))

def create_metrics_dataframe(metrics_dict, title="Metrics"):
    """Create a properly formatted dataframe from metrics dictionary"""
    try:
        if not metrics_dict:
            return pd.DataFrame()
        
        # Convert metrics to proper format
        formatted_metrics = []
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                if abs(value) > 1000:
                    formatted_value = f"${value:,.2f}" if 'var' in key.lower() or 'shortfall' in key.lower() else f"{value:,.4f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            formatted_metrics.append({
                'Metric': str(key).replace('_', ' ').title(),
                'Value': formatted_value
            })
        
        return pd.DataFrame(formatted_metrics)
    
    except Exception as e:
        st.error(f"Error creating metrics dataframe: {str(e)}")
        return pd.DataFrame()

def main():
    # Header
    st.title("📊 VaR & Risk Analytics Platform")
    st.markdown("*Professional Value at Risk and Risk Management Analytics*")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Source Selection
        st.subheader("📊 Data Source")
        data_source = st.selectbox(
            "Select Data Source",
            ["Live Market Data", "Upload File", "Manual Entry", "Synthetic Data"]
        )
        
        # Portfolio Type Selection
        st.subheader("💼 Portfolio Type")
        portfolio_type = st.selectbox(
            "Select Portfolio Type",
            ["Single Asset", "Multi-Asset", "Crypto Portfolio", "Options Portfolio"]
        )
        
        # Date Range (Default: 1 year)
        st.subheader("📅 Date Range")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)  # 1 year default
        
        date_start = st.date_input("Start Date", start_date)
        date_end = st.date_input("End Date", end_date)
        
        # VaR Model Selection
        st.subheader("🎯 VaR Model")
        var_model = st.selectbox(
            "Select VaR Model",
            ["Parametric", "Historical", "Monte Carlo", "GARCH", "EVT"]
        )
        
        # Check if model changed
        if var_model != st.session_state.current_model:
            st.session_state.current_model = var_model
            # Clear old results but keep data
            st.session_state.var_results = {}
        
        # Risk Parameters
        st.subheader("⚡ Risk Parameters")
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        time_horizon = st.slider("Time Horizon (days)", 1, 30, 1)
        
        # Model-specific parameters
        if var_model == "Monte Carlo":
            num_simulations = st.slider("Number of Simulations", 1000, 100000, 10000, 1000)
        elif var_model == "GARCH":
            garch_p = st.slider("GARCH P", 1, 3, 1)
            garch_q = st.slider("GARCH Q", 1, 3, 1)
        
        # Backtesting Parameters
        st.subheader("🧪 Backtesting")
        backtest_window = st.slider("Backtesting Window", 100, 1000, 252)
    
    # Data Loading Section
    st.header("📈 Data Loading & Portfolio Setup")
    
    # Portfolio configuration based on type and data source
    if data_source == "Live Market Data":
        if portfolio_type == "Single Asset":
            symbols = st.text_input("Enter Symbol", value="AAPL").upper().split(',')
            symbols = [s.strip() for s in symbols if s.strip()]
            weights = [1.0]
            
        elif portfolio_type == "Multi-Asset":
            default_symbols = "AAPL,GOOGL,MSFT,TSLA"
            symbols_input = st.text_input("Enter Symbols (comma-separated)", value=default_symbols)
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            
            # Add crypto field for multi-asset
            crypto_symbols_input = st.text_input("Enter Crypto Symbols (comma-separated, optional)", 
                                                placeholder="BTC-USD,ETH-USD")
            if crypto_symbols_input.strip():
                crypto_symbols = [s.strip().upper() for s in crypto_symbols_input.split(',') if s.strip()]
                symbols.extend(crypto_symbols)
            
            # Portfolio weights
            if len(symbols) > 1:
                st.write("Portfolio Weights:")
                weights = []
                for i, symbol in enumerate(symbols):
                    weight = st.number_input(f"Weight for {symbol}", 0.0, 1.0, 1.0/len(symbols), 0.01, key=f"weight_{i}")
                    weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
                    st.info(f"Normalized weights: {[f'{w:.3f}' for w in weights]}")
            else:
                weights = [1.0]
                
        elif portfolio_type == "Crypto Portfolio":
            default_crypto = "BTC-USD"
            symbols_input = st.text_input("Enter Crypto Symbols (comma-separated)", value=default_crypto)
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            
            if len(symbols) > 1:
                st.write("Portfolio Weights:")
                weights = []
                for i, symbol in enumerate(symbols):
                    weight = st.number_input(f"Weight for {symbol}", 0.0, 1.0, 1.0/len(symbols), 0.01, key=f"crypto_weight_{i}")
                    weights.append(weight)
                
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
            else:
                weights = [1.0]
                
        elif portfolio_type == "Options Portfolio":
            st.subheader("Options Configuration")
            
            # Default AAPL call option
            spot_price = st.number_input("Spot Price ($)", value=150.0, min_value=1.0)
            strike_price = st.number_input("Strike Price ($)", value=155.0, min_value=1.0)
            time_to_expiry = st.number_input("Time to Expiry (years)", value=0.25, min_value=0.01, max_value=5.0)
            risk_free_rate = st.number_input("Risk-free Rate", value=0.05, min_value=0.0, max_value=1.0)
            volatility = st.number_input("Volatility", value=0.25, min_value=0.01, max_value=2.0)
            option_type = st.selectbox("Option Type", ["Call", "Put"])
            quantity = st.number_input("Quantity", value=100, min_value=1)
            
            symbols = ["Options Portfolio"]
            weights = [1.0]
        
        # Load data button
        if st.button("Load Market Data", type="primary"):
            with st.spinner("Loading market data..."):
                # Calculate required data range for backtesting
                required_days = backtest_window + 100  # Extra buffer
                extended_start = date_end - timedelta(days=required_days)
                actual_start = min(date_start, extended_start)
                
                if portfolio_type != "Options Portfolio":
                    data = instances['data_ingestion'].load_live_data(symbols, actual_start, date_end)
                    
                    if data is not None:
                        st.session_state.portfolio_data = data
                        st.session_state.portfolio_returns = instances['data_ingestion'].get_portfolio_returns(weights)
                        st.session_state.data_loaded = True
                        st.success(f"✅ Successfully loaded data for {len(symbols)} assets")
                        
                        # Display data summary
                        summary = instances['data_ingestion'].get_data_summary()
                        if summary:
                            st.info(f"📊 Data: {summary['data_points']} points from {summary['date_range']}")
                    else:
                        st.error("❌ Failed to load market data")
                else:
                    # For options, create synthetic underlying data
                    synthetic_data = instances['data_ingestion'].generate_synthetic_data(
                        num_days=required_days,
                        initial_price=spot_price,
                        annual_return=0.08,
                        annual_volatility=volatility
                    )
                    
                    if synthetic_data is not None:
                        st.session_state.portfolio_data = synthetic_data
                        st.session_state.portfolio_returns = synthetic_data.pct_change().dropna().iloc[:, 0]
                        st.session_state.data_loaded = True
                        st.session_state.options_params = {
                            'spot_price': spot_price,
                            'strike_price': strike_price,
                            'time_to_expiry': time_to_expiry,
                            'risk_free_rate': risk_free_rate,
                            'volatility': volatility,
                            'option_type': option_type,
                            'quantity': quantity
                        }
                        st.success("✅ Options portfolio configured successfully")
    
    elif data_source == "Upload File":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = instances['data_ingestion'].load_csv_data(uploaded_file)
            if data is not None:
                st.session_state.portfolio_data = data
                st.session_state.portfolio_returns = data.pct_change().dropna().mean(axis=1)
                st.session_state.data_loaded = True
                st.success("✅ File uploaded successfully")
    
    elif data_source == "Synthetic Data":
        st.subheader("Synthetic Data Parameters")
        
        # Expandable section for custom parameters
        with st.expander("Custom Data Generation Parameters"):
            num_days = st.slider("Number of Days", 100, 2000, 500)
            initial_price = st.number_input("Initial Price ($)", value=100.0, min_value=1.0)
            annual_return = st.slider("Annual Return", -0.5, 0.5, 0.08, 0.01)
            annual_volatility = st.slider("Annual Volatility", 0.05, 1.0, 0.20, 0.01)
            random_seed = st.number_input("Random Seed", value=42, min_value=1)
        
        if st.button("Generate Synthetic Data", type="primary"):
            with st.spinner("Generating synthetic data..."):
                data = instances['data_ingestion'].generate_synthetic_data(
                    num_days=num_days,
                    initial_price=initial_price,
                    annual_return=annual_return,
                    annual_volatility=annual_volatility,
                    random_seed=random_seed
                )
                
                if data is not None:
                    st.session_state.portfolio_data = data
                    st.session_state.portfolio_returns = data.pct_change().dropna().iloc[:, 0]
                    st.session_state.data_loaded = True
                    st.success("✅ Synthetic data generated successfully")
    
    # Display loaded data
    if st.session_state.data_loaded and st.session_state.portfolio_data is not None:
        st.subheader("📊 Loaded Data Preview")
        
        # Filter data for display (last 1 year by default)
        display_start = date_end - timedelta(days=365)
        display_data = st.session_state.portfolio_data[
            (st.session_state.portfolio_data.index >= pd.Timestamp(display_start)) &
            (st.session_state.portfolio_data.index <= pd.Timestamp(date_end))
        ]
        
        safe_dataframe_display(display_data.tail(10), "Recent Data")
        
        # Basic statistics
        if len(display_data) > 0:
            stats_data = {
                'Count': len(display_data),
                'Start Date': display_data.index.min().strftime('%Y-%m-%d'),
                'End Date': display_data.index.max().strftime('%Y-%m-%d'),
                'Assets': len(display_data.columns)
            }
            
            stats_df = create_metrics_dataframe(stats_data, "Data Statistics")
            if not stats_df.empty:
                safe_dataframe_display(stats_df, "Data Statistics")
    
    # Main Analysis Tabs
    if st.session_state.data_loaded:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Dashboard", "🎯 VaR Calculator", "📈 Rolling Analysis", 
            "🧪 Backtesting", "⚡ Stress Testing", "📋 Options Analysis", "❓ Help"
        ])
        
        with tab1:
            st.header("📊 Risk Analytics Dashboard")
            
            # Dashboard time range controls
            st.subheader("📅 Dashboard Time Range")
            col1, col2 = st.columns(2)
            with col1:
                dash_start = st.date_input("Dashboard Start Date", 
                                         value=date_end - timedelta(days=365),
                                         key="dash_start")
            with col2:
                dash_end = st.date_input("Dashboard End Date", 
                                       value=date_end,
                                       key="dash_end")
            
            # Filter data for dashboard
            dash_data = st.session_state.portfolio_data[
                (st.session_state.portfolio_data.index >= pd.Timestamp(dash_start)) &
                (st.session_state.portfolio_data.index <= pd.Timestamp(dash_end))
            ]
            
            if len(dash_data) > 0:
                dash_returns = dash_data.pct_change().dropna()
                if len(dash_data.columns) > 1:
                    dash_portfolio_returns = dash_returns.mean(axis=1)
                else:
                    dash_portfolio_returns = dash_returns.iloc[:, 0]
                
                # Calculate VaR for current model
                if var_model == "Parametric":
                    current_var = instances['var_engines'].calculate_parametric_var(
                        dash_portfolio_returns, confidence_level, time_horizon
                    )
                elif var_model == "Historical":
                    current_var = instances['var_engines'].calculate_historical_var(
                        dash_portfolio_returns, confidence_level, time_horizon
                    )
                elif var_model == "Monte Carlo":
                    current_var = instances['var_engines'].calculate_monte_carlo_var(
                        dash_portfolio_returns, confidence_level, time_horizon, 
                        num_simulations if 'num_simulations' in locals() else 10000
                    )
                elif var_model == "GARCH":
                    current_var = instances['var_engines'].calculate_garch_var(
                        dash_portfolio_returns, confidence_level, time_horizon,
                        garch_p if 'garch_p' in locals() else 1,
                        garch_q if 'garch_q' in locals() else 1
                    )
                elif var_model == "EVT":
                    current_var = instances['var_engines'].calculate_evt_var(
                        dash_portfolio_returns, confidence_level
                    )
                
                # Calculate Expected Shortfall
                expected_shortfall = instances['var_engines'].calculate_expected_shortfall(
                    dash_portfolio_returns, confidence_level
                )
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label=f"VaR ({confidence_level*100:.0f}%) - {var_model}",
                        value=f"${current_var:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        label="Expected Shortfall",
                        value=f"${expected_shortfall:,.0f}"
                    )
                
                with col3:
                    annual_vol = dash_portfolio_returns.std() * np.sqrt(252) * 100
                    st.metric(
                        label="Annual Volatility",
                        value=f"{annual_vol:.1f}%"
                    )
                
                with col4:
                    annual_return = dash_portfolio_returns.mean() * 252 * 100
                    st.metric(
                        label="Annual Return",
                        value=f"{annual_return:.1f}%"
                    )
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Portfolio performance chart
                    cumulative_returns = (1 + dash_portfolio_returns).cumprod()
                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns.values,
                        mode='lines',
                        name='Cumulative Returns',
                        line=dict(color='#00ff88', width=2)
                    ))
                    fig_perf.update_layout(
                        title="Portfolio Performance",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Returns",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
                
                with col2:
                    # Returns distribution
                    fig_dist = instances['visualization'].plot_var_distribution(
                        dash_portfolio_returns, confidence_level, current_var
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Rolling VaR
                if len(dash_portfolio_returns) > 60:
                    rolling_var = instances['rolling_analysis'].calculate_rolling_var(
                        dash_portfolio_returns, confidence_level, 60
                    )
                    
                    if len(rolling_var) > 0:
                        fig_rolling = go.Figure()
                        fig_rolling.add_trace(go.Scatter(
                            x=rolling_var.index,
                            y=rolling_var.values,
                            mode='lines',
                            name=f'Rolling VaR ({var_model})',
                            line=dict(color='#ff6b6b', width=2)
                        ))
                        fig_rolling.update_layout(
                            title=f"Rolling VaR (60-day window) - {var_model}",
                            xaxis_title="Date",
                            yaxis_title="VaR ($)",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig_rolling, use_container_width=True)
        
        with tab2:
            st.header("🎯 VaR Calculator")
            
            # VaR Calculator time range controls
            st.subheader("📅 VaR Analysis Time Range")
            col1, col2 = st.columns(2)
            with col1:
                var_start = st.date_input("VaR Start Date", 
                                        value=date_end - timedelta(days=365),
                                        key="var_start")
            with col2:
                var_end = st.date_input("VaR End Date", 
                                      value=date_end,
                                      key="var_end")
            
            # Filter data for VaR analysis
            var_data = st.session_state.portfolio_data[
                (st.session_state.portfolio_data.index >= pd.Timestamp(var_start)) &
                (st.session_state.portfolio_data.index <= pd.Timestamp(var_end))
            ]
            
            if len(var_data) > 0:
                var_returns = var_data.pct_change().dropna()
                if len(var_data.columns) > 1:
                    var_portfolio_returns = var_returns.mean(axis=1)
                else:
                    var_portfolio_returns = var_returns.iloc[:, 0]
                
                if portfolio_type == "Options Portfolio" and 'options_params' in st.session_state:
                    # Options VaR Analysis
                    st.subheader("Options VaR Analysis")
                    
                    options_method = st.selectbox(
                        "Options VaR Method",
                        ["Delta-Normal", "Delta-Gamma", "Full Revaluation Monte Carlo"]
                    )
                    
                    params = st.session_state.options_params
                    options_var_result = instances['options_var'].calculate_options_var(
                        params['spot_price'],
                        params['strike_price'],
                        params['time_to_expiry'],
                        params['risk_free_rate'],
                        params['volatility'],
                        params['option_type'],
                        options_method,
                        confidence_level
                    )
                    
                    if options_var_result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Options VaR", f"${options_var_result.get('var', 0):,.2f}")
                            st.metric("Current Option Price", f"${options_var_result.get('current_price', 0):.2f}")
                            st.metric("Delta", f"{options_var_result.get('delta', 0):.4f}")
                        
                        with col2:
                            st.metric("Gamma", f"{options_var_result.get('gamma', 0):.6f}")
                            st.metric("Theta", f"${options_var_result.get('theta', 0):.2f}")
                            st.metric("Vega", f"${options_var_result.get('vega', 0):.2f}")
                
                else:
                    # Regular VaR Analysis
                    if var_model == "Parametric":
                        var_result = instances['var_engines'].calculate_parametric_var(
                            var_portfolio_returns, confidence_level, time_horizon
                        )
                    elif var_model == "Historical":
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
                    elif var_model == "EVT":
                        var_result = instances['var_engines'].calculate_evt_var(
                            var_portfolio_returns, confidence_level
                        )
                    
                    # Calculate Expected Shortfall
                    expected_shortfall = instances['var_engines'].calculate_expected_shortfall(
                        var_portfolio_returns, confidence_level
                    )
                    
                    # Store results
                    st.session_state.var_results = {
                        'var': var_result,
                        'expected_shortfall': expected_shortfall,
                        'model': var_model,
                        'confidence_level': confidence_level
                    }
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(f"VaR ({confidence_level*100:.0f}%) - {var_model}", f"${var_result:,.0f}")
                    
                    with col2:
                        st.metric("Expected Shortfall", f"${expected_shortfall:,.0f}")
                    
                    with col3:
                        ratio = expected_shortfall / var_result if var_result > 0 else 0
                        st.metric("ES/VaR Ratio", f"{ratio:.2f}")
                    
                    # VaR Distribution Plot
                    fig_dist = instances['visualization'].plot_var_distribution(
                        var_portfolio_returns, confidence_level, var_result
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Portfolio Statistics
                    st.subheader("Portfolio Statistics")
                    stats = instances['utils'].calculate_portfolio_statistics(var_portfolio_returns)
                    
                    if stats:
                        stats_df = create_metrics_dataframe(stats, "Portfolio Statistics")
                        safe_dataframe_display(stats_df, "Portfolio Statistics")
        
        with tab3:
            st.header("📈 Rolling Analysis")
            
            # Rolling Analysis time range controls
            st.subheader("📅 Rolling Analysis Time Range")
            col1, col2 = st.columns(2)
            with col1:
                rolling_start = st.date_input("Rolling Start Date", 
                                            value=date_end - timedelta(days=365),
                                            key="rolling_start")
            with col2:
                rolling_end = st.date_input("Rolling End Date", 
                                          value=date_end,
                                          key="rolling_end")
            
            # Filter data for rolling analysis
            rolling_data = st.session_state.portfolio_data[
                (st.session_state.portfolio_data.index >= pd.Timestamp(rolling_start)) &
                (st.session_state.portfolio_data.index <= pd.Timestamp(rolling_end))
            ]
            
            if len(rolling_data) > 0:
                rolling_returns = rolling_data.pct_change().dropna()
                if len(rolling_data.columns) > 1:
                    rolling_portfolio_returns = rolling_returns.mean(axis=1)
                else:
                    rolling_portfolio_returns = rolling_returns.iloc[:, 0]
                
                window_size = st.slider("Rolling Window Size", 30, 252, 60)
                
                if len(rolling_portfolio_returns) > window_size:
                    # Calculate rolling metrics
                    rolling_var = instances['rolling_analysis'].calculate_rolling_var(
                        rolling_portfolio_returns, confidence_level, window_size
                    )
                    rolling_vol = instances['rolling_analysis'].calculate_rolling_volatility(
                        rolling_portfolio_returns, window_size
                    )
                    rolling_sharpe = instances['rolling_analysis'].calculate_rolling_sharpe(
                        rolling_portfolio_returns, window_size
                    )
                    
                    # Display charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if len(rolling_var) > 0:
                            fig_rolling_var = instances['visualization'].plot_rolling_metrics(
                                rolling_var, f"Rolling VaR ({var_model})"
                            )
                            st.plotly_chart(fig_rolling_var, use_container_width=True)
                    
                    with col2:
                        if len(rolling_vol) > 0:
                            fig_rolling_vol = instances['visualization'].plot_rolling_metrics(
                                rolling_vol, "Rolling Volatility"
                            )
                            st.plotly_chart(fig_rolling_vol, use_container_width=True)
                    
                    # Rolling Sharpe Ratio
                    if len(rolling_sharpe) > 0:
                        fig_rolling_sharpe = instances['visualization'].plot_rolling_metrics(
                            rolling_sharpe, "Rolling Sharpe Ratio"
                        )
                        st.plotly_chart(fig_rolling_sharpe, use_container_width=True)
                    
                    # Drawdown Analysis
                    drawdown_result = instances['rolling_analysis'].calculate_maximum_drawdown(rolling_portfolio_returns)
                    if drawdown_result and 'drawdown_series' in drawdown_result:
                        st.subheader("Drawdown Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Maximum Drawdown", f"{drawdown_result['max_drawdown']*100:.2f}%")
                        with col2:
                            if drawdown_result['max_drawdown_date']:
                                st.metric("Max DD Date", drawdown_result['max_drawdown_date'].strftime('%Y-%m-%d'))
                        
                        fig_dd = instances['visualization'].plot_drawdown(drawdown_result['drawdown_series'])
                        st.plotly_chart(fig_dd, use_container_width=True)
        
        with tab4:
            st.header("🧪 Backtesting & Validation")
            
            # Backtesting time range controls
            st.subheader("📅 Backtesting Time Range")
            col1, col2 = st.columns(2)
            with col1:
                backtest_start = st.date_input("Backtest Start Date", 
                                             value=date_end - timedelta(days=500),
                                             key="backtest_start")
            with col2:
                backtest_end = st.date_input("Backtest End Date", 
                                           value=date_end,
                                           key="backtest_end")
            
            # Filter data for backtesting
            backtest_data = st.session_state.portfolio_data[
                (st.session_state.portfolio_data.index >= pd.Timestamp(backtest_start)) &
                (st.session_state.portfolio_data.index <= pd.Timestamp(backtest_end))
            ]
            
            if len(backtest_data) > backtest_window + 50:
                backtest_returns = backtest_data.pct_change().dropna()
                if len(backtest_data.columns) > 1:
                    backtest_portfolio_returns = backtest_returns.mean(axis=1)
                else:
                    backtest_portfolio_returns = backtest_returns.iloc[:, 0]
                
                if st.button("Run Backtesting", type="primary"):
                    with st.spinner("Running backtesting..."):
                        # Create VaR method function based on selected model
                        def var_method_func(returns, conf_level, horizon):
                            if var_model == "Parametric":
                                return instances['var_engines'].calculate_parametric_var(returns, conf_level, horizon)
                            elif var_model == "Historical":
                                return instances['var_engines'].calculate_historical_var(returns, conf_level, horizon)
                            elif var_model == "Monte Carlo":
                                return instances['var_engines'].calculate_monte_carlo_var(
                                    returns, conf_level, horizon, 
                                    num_simulations if 'num_simulations' in locals() else 10000
                                )
                            elif var_model == "GARCH":
                                return instances['var_engines'].calculate_garch_var(
                                    returns, conf_level, horizon,
                                    garch_p if 'garch_p' in locals() else 1,
                                    garch_q if 'garch_q' in locals() else 1
                                )
                            elif var_model == "EVT":
                                return instances['var_engines'].calculate_evt_var(returns, conf_level)
                        
                        backtest_results = instances['backtesting'].perform_backtesting(
                            backtest_portfolio_returns, confidence_level, backtest_window, var_method_func
                        )
                        
                        if backtest_results:
                            # Display results
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Violations", backtest_results.get('violations', 0))
                            
                            with col2:
                                expected_viol = backtest_results.get('expected_violations', 0)
                                st.metric("Expected Violations", f"{expected_viol:.1f}")
                            
                            with col3:
                                violation_rate = backtest_results.get('violation_rate', 0) * 100
                                st.metric("Violation Rate", f"{violation_rate:.2f}%")
                            
                            with col4:
                                kupiec_p = backtest_results.get('kupiec_pvalue', 0)
                                st.metric("Kupiec p-value", f"{kupiec_p:.4f}")
                            
                            # Basel Traffic Light
                            traffic_light = instances['backtesting'].basel_traffic_light(
                                backtest_results.get('violations', 0),
                                backtest_results.get('expected_violations', 0)
                            )
                            
                            if traffic_light == "Green":
                                st.success(f"🟢 Basel Traffic Light: {traffic_light} - Model performs well")
                            elif traffic_light == "Yellow":
                                st.warning(f"🟡 Basel Traffic Light: {traffic_light} - Model needs attention")
                            else:
                                st.error(f"🔴 Basel Traffic Light: {traffic_light} - Model needs review")
                            
                            # Violations plot
                            if 'var_estimates' in backtest_results and 'violations_dates' in backtest_results:
                                fig_violations = instances['visualization'].plot_var_violations(
                                    backtest_portfolio_returns,
                                    backtest_results['var_estimates'],
                                    backtest_results['violations_dates']
                                )
                                st.plotly_chart(fig_violations, use_container_width=True)
                            
                            # Test results table
                            test_results = {
                                'Kupiec Test p-value': backtest_results.get('kupiec_pvalue', 0),
                                'Independence Test p-value': backtest_results.get('independence_pvalue', 0),
                                'Conditional Coverage p-value': backtest_results.get('cc_pvalue', 0)
                            }
                            
                            test_df = create_metrics_dataframe(test_results, "Statistical Tests")
                            if not test_df.empty:
                                st.subheader("Statistical Test Results")
                                safe_dataframe_display(test_df, "Statistical Tests")
                                
                                st.info("💡 **Interpretation**: p-values > 0.05 indicate the model passes the test")
            else:
                st.warning(f"Insufficient data for backtesting. Need at least {backtest_window + 50} data points.")
        
        with tab5:
            st.header("⚡ Stress Testing & Scenario Analysis")
            
            # Stress Testing time range controls
            st.subheader("📅 Stress Testing Time Range")
            col1, col2 = st.columns(2)
            with col1:
                stress_start = st.date_input("Stress Test Start Date", 
                                           value=date_end - timedelta(days=365),
                                           key="stress_start")
            with col2:
                stress_end = st.date_input("Stress Test End Date", 
                                         value=date_end,
                                         key="stress_end")
            
            # Filter data for stress testing
            stress_data = st.session_state.portfolio_data[
                (st.session_state.portfolio_data.index >= pd.Timestamp(stress_start)) &
                (st.session_state.portfolio_data.index <= pd.Timestamp(stress_end))
            ]
            
            if len(stress_data) > 0:
                stress_returns = stress_data.pct_change().dropna()
                if len(stress_data.columns) > 1:
                    stress_portfolio_returns = stress_returns.mean(axis=1)
                else:
                    stress_portfolio_returns = stress_returns.iloc[:, 0]
                
                # Scenario Selection
                scenario_type = st.selectbox(
                    "Select Stress Scenario",
                    ["2008 Financial Crisis", "COVID-19 Pandemic", "Dot-com Crash", "Custom Scenario"]
                )
                
                if scenario_type == "Custom Scenario":
                    st.subheader("Custom Scenario Parameters")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        vol_shock = st.slider("Volatility Shock (%)", 0, 500, 100)
                    with col2:
                        corr_shock = st.slider("Correlation Shock", 0.0, 1.0, 0.3, 0.1)
                    with col3:
                        market_shock = st.slider("Market Shock (%)", -50, 50, -20)
                    
                    if st.button("Run Custom Stress Test", type="primary"):
                        with st.spinner("Running custom stress test..."):
                            stress_results = instances['stress_testing'].run_custom_stress_test(
                                stress_portfolio_returns, vol_shock, corr_shock, market_shock,
                                confidence_level, time_horizon
                            )
                            
                            if stress_results:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    baseline_var = stress_results.get('baseline_var', 0)
                                    st.metric("Baseline VaR", f"${baseline_var:,.0f}")
                                
                                with col2:
                                    stressed_var = stress_results.get('stressed_var', 0)
                                    st.metric("Stressed VaR", f"${stressed_var:,.0f}")
                                
                                with col3:
                                    var_increase = stress_results.get('var_increase', 0)
                                    st.metric("VaR Increase", f"{var_increase:.1f}%")
                
                else:
                    if st.button("Run Historical Stress Test", type="primary"):
                        with st.spinner(f"Running {scenario_type} stress test..."):
                            stress_results = instances['stress_testing'].run_stress_test(
                                stress_portfolio_returns, scenario_type, confidence_level, time_horizon
                            )
                            
                            if stress_results:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    baseline_var = stress_results.get('baseline_var', 0)
                                    st.metric("Baseline VaR", f"${baseline_var:,.0f}")
                                
                                with col2:
                                    stressed_var = stress_results.get('stressed_var', 0)
                                    st.metric("Stressed VaR", f"${stressed_var:,.0f}")
                                
                                with col3:
                                    var_increase = stress_results.get('var_increase', 0)
                                    st.metric("VaR Increase", f"{var_increase:.1f}%")
                                
                                # Worst case scenario
                                worst_case = stress_results.get('worst_case', 0)
                                st.metric("Worst Case Loss (1st percentile)", f"${worst_case:,.0f}")
                
                # Sensitivity Analysis
                st.subheader("Sensitivity Analysis")
                if st.button("Run Sensitivity Analysis"):
                    with st.spinner("Running sensitivity analysis..."):
                        sensitivity_data = instances['stress_testing'].sensitivity_analysis(
                            stress_portfolio_returns, confidence_level
                        )
                        
                        if not sensitivity_data.empty:
                            fig_sensitivity = instances['visualization'].plot_sensitivity_analysis(sensitivity_data)
                            st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        with tab6:
            st.header("📋 Options Analysis")
            
            if portfolio_type == "Options Portfolio" and 'options_params' in st.session_state:
                params = st.session_state.options_params
                
                # Options Analysis time range controls
                st.subheader("📅 Options Analysis Time Range")
                col1, col2 = st.columns(2)
                with col1:
                    options_start = st.date_input("Options Start Date", 
                                                value=date_end - timedelta(days=365),
                                                key="options_start")
                with col2:
                    options_end = st.date_input("Options End Date", 
                                              value=date_end,
                                              key="options_end")
                
                # Current option details
                st.subheader("Option Details")
                details_data = {
                    'Underlying': 'AAPL',
                    'Option Type': params['option_type'],
                    'Spot Price': f"${params['spot_price']:.2f}",
                    'Strike Price': f"${params['strike_price']:.2f}",
                    'Time to Expiry': f"{params['time_to_expiry']:.3f} years",
                    'Risk-free Rate': f"{params['risk_free_rate']*100:.1f}%",
                    'Volatility': f"{params['volatility']*100:.1f}%",
                    'Quantity': params['quantity']
                }
                
                details_df = create_metrics_dataframe(details_data, "Option Details")
                safe_dataframe_display(details_df, "Option Details")
                
                # Greeks calculation
                greeks = instances['options_var'].calculate_greeks(
                    params['spot_price'],
                    params['strike_price'],
                    params['time_to_expiry'],
                    params['risk_free_rate'],
                    params['volatility'],
                    params['option_type']
                )
                
                if greeks:
                    st.subheader("Option Greeks")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Delta", f"{greeks['delta']:.4f}")
                    with col2:
                        st.metric("Gamma", f"{greeks['gamma']:.6f}")
                    with col3:
                        st.metric("Theta", f"${greeks['theta']:.2f}")
                    with col4:
                        st.metric("Vega", f"${greeks['vega']:.2f}")
                    with col5:
                        st.metric("Rho", f"${greeks['rho']:.2f}")
                
                # VaR Analysis for different methods
                st.subheader("Options VaR Analysis")
                
                methods = ["Delta-Normal", "Delta-Gamma", "Full Revaluation Monte Carlo"]
                var_results = {}
                
                for method in methods:
                    var_result = instances['options_var'].calculate_options_var(
                        params['spot_price'],
                        params['strike_price'],
                        params['time_to_expiry'],
                        params['risk_free_rate'],
                        params['volatility'],
                        params['option_type'],
                        method,
                        confidence_level
                    )
                    
                    if var_result:
                        var_results[method] = var_result['var']
                
                if var_results:
                    # Display VaR comparison
                    var_comparison_df = pd.DataFrame(list(var_results.items()), columns=['Method', 'VaR ($)'])
                    var_comparison_df['VaR ($)'] = var_comparison_df['VaR ($)'].apply(lambda x: f"${x:,.2f}")
                    safe_dataframe_display(var_comparison_df, "VaR Comparison")
                    
                    # VaR comparison chart
                    fig_var_comp = instances['visualization'].plot_var_comparison(var_results)
                    st.plotly_chart(fig_var_comp, use_container_width=True)
                
                # Payoff diagram
                st.subheader("Option Payoff Diagram")
                spot_range = np.linspace(
                    params['spot_price'] * 0.8,
                    params['spot_price'] * 1.2,
                    50
                )
                
                fig_payoff = instances['visualization'].plot_options_payoff(
                    spot_range,
                    params['strike_price'],
                    params['option_type'],
                    params['time_to_expiry'],
                    params['risk_free_rate'],
                    params['volatility']
                )
                st.plotly_chart(fig_payoff, use_container_width=True)
            
            else:
                st.info("Options analysis is available when 'Options Portfolio' is selected.")
                st.markdown("""
                **To analyze options:**
                1. Select 'Options Portfolio' in the sidebar
                2. Configure option parameters
                3. Load the data
                4. Return to this tab for detailed options analysis
                """)
        
        with tab7:
            st.header("❓ Help & Documentation")
            
            st.markdown("""
            ## 🚀 Welcome to VaR & Risk Analytics Platform
            
            This comprehensive platform provides professional-grade Value at Risk (VaR) and risk management analytics.
            
            ### 📊 **Getting Started**
            
            1. **Select Data Source** in the sidebar:
               - **Live Market Data**: Real-time data from Yahoo Finance
               - **Upload File**: CSV/Excel files with your data
               - **Manual Entry**: Direct data input
               - **Synthetic Data**: Generated data for testing
            
            2. **Choose Portfolio Type**:
               - **Single Asset**: Analyze one stock/crypto
               - **Multi-Asset**: Portfolio of multiple assets
               - **Crypto Portfolio**: Cryptocurrency analysis
               - **Options Portfolio**: Derivatives risk analysis
            
            3. **Configure Parameters**:
               - VaR Model (Parametric, Historical, Monte Carlo, GARCH, EVT)
               - Confidence Level (90-99%)
               - Time Horizon (1-30 days)
               - Backtesting Window (100-1000 days)
            
            ### 💼 **Portfolio Types & Defaults**
            
            #### Single Asset
            - **Default**: AAPL
            - **Format**: Standard ticker symbols (AAPL, GOOGL, MSFT)
            
            #### Multi-Asset
            - **Default**: AAPL, GOOGL, MSFT, TSLA
            - **Crypto Support**: Add crypto symbols (BTC-USD, ETH-USD)
            - **Weights**: Automatically normalized to sum to 1.0
            
            #### Crypto Portfolio
            - **Default**: BTC-USD
            - **Format**: Symbol + "-USD" (ETH-USD, ADA-USD, SOL-USD)
            - **Popular Cryptos**: BTC-USD, ETH-USD, BNB-USD, ADA-USD, SOL-USD
            
            #### Options Portfolio
            - **Default**: AAPL call option
            - **Parameters**:
              - Spot Price: $150
              - Strike Price: $155
              - Time to Expiry: 0.25 years (3 months)
              - Risk-free Rate: 5%
              - Volatility: 25%
              - Quantity: 100 contracts
            
            ### 📈 **Live Options Data Format**
            
            For live options data, use this format:
            ```
            [SYMBOL][YYMMDD][C/P][STRIKE_PRICE]
            ```
            
            **Examples**:
            - `AAPL230616C00150000` - AAPL call expiring June 16, 2023, strike $150
            - `TSLA231215P00200000` - TSLA put expiring Dec 15, 2023, strike $200
            - `GOOGL240119C00120000` - GOOGL call expiring Jan 19, 2024, strike $120
            
            **Popular Options**:
            - **AAPL**: Strikes around $140-$160, 1-6 month expiries
            - **TSLA**: Strikes around $180-$220, high volatility (~40%)
            - **GOOGL**: Strikes around $110-$130, moderate volatility (~30%)
            - **MSFT**: Strikes around $280-$320, stable volatility (~25%)
            
            ### 📁 **File Upload Formats**
            
            #### CSV Format
            ```csv
            Date,Asset1,Asset2,Asset3
            2023-01-01,100.50,200.25,150.75
            2023-01-02,101.25,198.50,152.00
            2023-01-03,99.75,201.00,149.25
            ```
            
            #### Excel Format
            - Same structure as CSV
            - First column: Date (YYYY-MM-DD)
            - Subsequent columns: Asset prices
            - Headers recommended but optional
            
            **Requirements**:
            - Minimum 30 data points
            - Date format: YYYY-MM-DD
            - Numeric price data
            - No missing values in price columns
            
            ### 🎯 **VaR Models Explained**
            
            #### 1. Parametric (Delta-Normal)
            - **Best for**: Normal market conditions
            - **Assumptions**: Returns follow normal distribution
            - **Speed**: Very fast
            - **Accuracy**: Good for linear portfolios
            
            #### 2. Historical Simulation
            - **Best for**: Capturing actual market behavior
            - **Assumptions**: Past patterns repeat
            - **Speed**: Fast
            - **Accuracy**: Good for non-normal distributions
            
            #### 3. Monte Carlo
            - **Best for**: Complex portfolios
            - **Assumptions**: Specified return distribution
            - **Speed**: Slower (depends on simulations)
            - **Accuracy**: Very high with sufficient simulations
            
            #### 4. GARCH
            - **Best for**: Volatility clustering markets
            - **Assumptions**: Time-varying volatility
            - **Speed**: Moderate
            - **Accuracy**: Excellent for volatile markets
            
            #### 5. Extreme Value Theory (EVT)
            - **Best for**: Tail risk analysis
            - **Assumptions**: Extreme events follow GPD
            - **Speed**: Moderate
            - **Accuracy**: Superior for extreme scenarios
            
            ### 📊 **Options VaR Methods**
            
            #### Delta-Normal
            - **Approach**: Linear approximation using delta
            - **Speed**: Very fast
            - **Accuracy**: Good for small price moves
            - **Best for**: Quick estimates, at-the-money options
            
            #### Delta-Gamma
            - **Approach**: Second-order Taylor expansion
            - **Speed**: Fast
            - **Accuracy**: Better for larger price moves
            - **Best for**: Options with significant gamma exposure
            
            #### Full Revaluation Monte Carlo
            - **Approach**: Complete option repricing
            - **Speed**: Slower
            - **Accuracy**: Highest accuracy
            - **Best for**: Complex options, precise risk measurement
            
            ### 📈 **Individual Graph Time Controls**
            
            Each chart has independent time range controls:
            - **Default**: 1-year lookback period
            - **Customizable**: Start and end date selectors
            - **Automatic Filtering**: Data subset for selected period
            - **Consistent Analysis**: Same time range across related metrics
            
            **Available in**:
            - Dashboard performance charts
            - VaR distribution analysis
            - Rolling metrics visualization
            - Backtesting violation plots
            - Stress testing scenarios
            - Options payoff diagrams
            
            ### 🔧 **Parameter Guidelines**
            
            #### Confidence Levels
            - **90%**: Conservative, frequent violations expected
            - **95%**: Standard industry practice
            - **99%**: Regulatory requirement, rare violations
            
            #### Time Horizons
            - **1 day**: Daily VaR, most common
            - **10 days**: Basel regulatory standard
            - **30 days**: Monthly risk assessment
            
            #### Historical Windows
            - **60 days**: Short-term, recent market conditions
            - **252 days**: One trading year, standard
            - **500+ days**: Long-term, stable estimates
            
            #### Backtesting Windows
            - **252 days**: Minimum for reliable testing
            - **500 days**: Good balance of data and relevance
            - **1000 days**: Comprehensive long-term validation
            
            ### 🚨 **Troubleshooting**
            
            #### Common Issues
            
            **"Insufficient data for backtesting"**
            - Increase date range or reduce backtesting window
            - Ensure at least 300+ data points available
            
            **"Symbol not found"**
            - Verify ticker format (AAPL, not Apple)
            - Add -USD for crypto (BTC-USD, not BTC)
            - Check symbol exists on Yahoo Finance
            
            **"GARCH model failed"**
            - Requires minimum 100 observations
            - Increase historical window
            - Try different GARCH parameters (p,q)
            
            **"Weights don't sum to 1"**
            - Weights automatically normalized
            - Check for negative weights
            - Ensure all weights are numeric
            
            #### Performance Tips
            
            **Speed Optimization**:
            - Use fewer Monte Carlo simulations for testing
            - Reduce historical windows for faster calculations
            - Parametric VaR is fastest for quick analysis
            
            **Accuracy Improvement**:
            - Increase Monte Carlo simulations (10K+)
            - Use longer historical windows (500+ days)
            - GARCH for volatile markets, EVT for tail risk
            
            ### 📚 **Interpretation Guide**
            
            #### VaR Results
            - **VaR(95%, 1-day) = $10,000**: 95% confidence daily losses won't exceed $10,000
            - **Expected Shortfall**: Average loss when VaR is breached
            - **ES/VaR Ratio**: Higher ratios indicate more tail risk
            
            #### Backtesting
            - **Kupiec p-value > 0.05**: Model passes statistical validation
            - **Violation Rate ≈ (1 - Confidence Level)**: Expected violation frequency
            - **Basel Green**: Model performs well
            - **Basel Yellow/Red**: Model needs attention/review
            
            #### Options Greeks
            - **Delta**: Price sensitivity to underlying movement
            - **Gamma**: Delta sensitivity (convexity)
            - **Theta**: Time decay (daily P&L impact)
            - **Vega**: Volatility sensitivity
            - **Rho**: Interest rate sensitivity
            
            ### 🎓 **Best Practices**
            
            1. **Start Simple**: Begin with Parametric VaR, then explore advanced models
            2. **Validate Models**: Always run backtesting before relying on results
            3. **Multiple Models**: Compare results across different VaR methods
            4. **Regular Updates**: Refresh data and recalibrate models regularly
            5. **Stress Testing**: Complement VaR with scenario analysis
            6. **Documentation**: Keep records of model choices and parameters
            
            ### 📞 **Support**
            
            For additional help:
            - Check parameter settings in sidebar
            - Verify data quality and format
            - Try different VaR models for comparison
            - Use synthetic data for testing and learning
            
            ---
            
            **Built with ❤️ for Risk Management Professionals**
            """)
    
    else:
        st.info("👆 Please configure your portfolio in the sidebar and load data to begin analysis.")
        
        # Quick start guide
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        1. **Select Data Source** (Live Market Data recommended for beginners)
        2. **Choose Portfolio Type** (Single Asset is simplest to start)
        3. **Configure Date Range** (Default 1-year period works well)
        4. **Select VaR Model** (Parametric is fastest for initial analysis)
        5. **Click "Load Market Data"** to begin
        
        ### 💡 **Recommended First Steps**:
        - Try **Single Asset** with **AAPL** using **Parametric VaR**
        - Explore the **Dashboard** tab for overview
        - Run **Backtesting** to validate the model
        - Experiment with **Stress Testing** scenarios
        """)

if __name__ == "__main__":
    main()