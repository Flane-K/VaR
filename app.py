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
        'utils': Utils()
    }
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Source Selection
        st.subheader("📊 Data Source")
        data_source = st.selectbox(
            "Select Data Source",
            ["Live Market Data", "Upload File", "Manual Entry", "Synthetic Data"],
            key="data_source_select"
        )
        
        # Portfolio Type Selection
        st.subheader("💼 Portfolio Type")
        portfolio_type = st.selectbox(
            "Select Portfolio Type",
            ["Single Asset", "Multi-Asset", "Crypto Portfolio", "Options Portfolio"],
            key="portfolio_type_select"
        )
        
        # VaR Model Selection
        st.subheader("🎯 VaR Model")
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
        
        # Data Source Specific Configuration
        if data_source == "Live Market Data":
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
                
            elif portfolio_type == "Options Portfolio":
                st.write("Options Configuration:")
                underlying = st.text_input("Underlying Symbol", "AAPL", key="options_underlying_input")
                spot_price = st.number_input("Current Spot Price ($)", 100.0, 1000.0, 150.0, 1.0, key="spot_price_input")
                strike_price = st.number_input("Strike Price ($)", 50.0, 1000.0, 155.0, 1.0, key="strike_price_input")
                time_to_expiry = st.number_input("Time to Expiry (years)", 0.01, 2.0, 0.25, 0.01, key="time_expiry_input")
                risk_free_rate = st.number_input("Risk-free Rate", 0.0, 0.1, 0.05, 0.001, key="risk_free_input")
                volatility = st.number_input("Volatility", 0.1, 1.0, 0.25, 0.01, key="volatility_input")
                option_type = st.selectbox("Option Type", ["Call", "Put"], key="option_type_select")
                quantity = st.number_input("Quantity", 1, 1000, 100, 1, key="quantity_input")
                
                symbols = [underlying]
                weights = [1.0]
        
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
                    if data_source == "Live Market Data":
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
            
            # Calculate VaR based on selected model
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
            "🧪 Backtesting", "⚡ Stress Testing", "📊 Options Analysis", "❓ Help"
        ])
        
        with tab1:  # Dashboard
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
            filtered_returns = filtered_data.pct_change().dropna()
            
            if len(filtered_returns.shape) > 1:
                filtered_portfolio_returns = filtered_returns.dot(st.session_state.weights)
            else:
                filtered_portfolio_returns = filtered_returns
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Portfolio Performance")
                if not filtered_portfolio_returns.empty:
                    cumulative_returns = (1 + filtered_portfolio_returns).cumprod()
                    
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
                if not filtered_portfolio_returns.empty:
                    fig_dist = instances['visualization'].plot_var_distribution(
                        filtered_portfolio_returns, confidence_level, var_result
                    )
                    st.plotly_chart(fig_dist, use_container_width=True, key="dashboard_distribution_chart")
            
            # Portfolio Statistics
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
            var_filtered_returns = var_filtered_data.pct_change().dropna()
            
            if len(var_filtered_returns.shape) > 1:
                var_filtered_portfolio_returns = var_filtered_returns.dot(st.session_state.weights)
            else:
                var_filtered_portfolio_returns = var_filtered_returns
            
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
                    comparison_results['Parametric'] = instances['var_engines'].calculate_parametric_var(
                        var_filtered_portfolio_returns, confidence_level, time_horizon
                    )
                    comparison_results['Historical'] = instances['var_engines'].calculate_historical_var(
                        var_filtered_portfolio_returns, confidence_level, time_horizon
                    )
                    comparison_results['Monte Carlo'] = instances['var_engines'].calculate_monte_carlo_var(
                        var_filtered_portfolio_returns, confidence_level, time_horizon, 5000
                    )
                    
                    comparison_df = create_metrics_dataframe(comparison_results, "Model Comparison")
                    safe_dataframe_display(comparison_df, "Model Comparison", "var_comparison")
                    
                except Exception as e:
                    st.warning(f"Error in model comparison: {str(e)}")
            
            # VaR Distribution Plot
            st.subheader("Returns Distribution with VaR Threshold")
            if not var_filtered_portfolio_returns.empty:
                fig_var_dist = instances['visualization'].plot_var_distribution(
                    var_filtered_portfolio_returns, confidence_level, var_result
                )
                st.plotly_chart(fig_var_dist, use_container_width=True, key="var_calculator_distribution")
        
        with tab3:  # Data Overview
            st.header("📋 Data Overview")
            
            # Data Summary
            st.subheader("📊 Data Summary")
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
                
                safe_dataframe_display(display_data.tail(20), "Recent Price Data", "price_data")
            
            # Returns Data
            st.subheader("📊 Returns Data")
            if st.session_state.current_returns is not None:
                display_returns_mask = (st.session_state.current_returns.index >= pd.to_datetime(data_display_start)) & \
                                      (st.session_state.current_returns.index <= pd.to_datetime(data_display_end))
                display_returns = st.session_state.current_returns.loc[display_returns_mask]
                
                safe_dataframe_display(display_returns.tail(20), "Recent Returns Data", "returns_data")
            
            # Portfolio Weights
            if hasattr(st.session_state, 'weights') and hasattr(st.session_state, 'symbols'):
                st.subheader("💼 Portfolio Composition")
                portfolio_composition = pd.DataFrame({
                    'Asset': st.session_state.symbols,
                    'Weight': [f"{w:.4f}" for w in st.session_state.weights],
                    'Weight (%)': [f"{w*100:.2f}%" for w in st.session_state.weights]
                })
                safe_dataframe_display(portfolio_composition, "Portfolio Composition", "portfolio_weights")
            
            # Data Quality Report
            st.subheader("🔍 Data Quality")
            quality_report = instances['utils'].validate_data_quality(st.session_state.current_data)
            if quality_report:
                quality_metrics = {
                    'Data Valid': "✅ Yes" if quality_report['valid'] else "❌ No",
                    'Warnings': len(quality_report.get('warnings', [])),
                    'Errors': len(quality_report.get('errors', [])),
                    'Recommendations': len(quality_report.get('recommendations', []))
                }
                quality_df = create_metrics_dataframe(quality_metrics, "Data Quality")
                safe_dataframe_display(quality_df, "Data Quality", "data_quality")
                
                # Show warnings and recommendations
                if quality_report.get('warnings'):
                    st.warning("⚠️ Data Quality Warnings:")
                    for warning in quality_report['warnings']:
                        st.write(f"• {warning}")
                
                if quality_report.get('recommendations'):
                    st.info("💡 Recommendations:")
                    for rec in quality_report['recommendations']:
                        st.write(f"• {rec}")
        
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
            backtest_filtered_returns = backtest_filtered_data.pct_change().dropna()
            
            if len(backtest_filtered_returns.shape) > 1:
                backtest_portfolio_returns = backtest_filtered_returns.dot(st.session_state.weights)
            else:
                backtest_portfolio_returns = backtest_filtered_returns
            
            if st.button("🔄 Run Backtesting", key="run_backtesting_button"):
                if len(backtest_portfolio_returns) >= backtesting_window + 50:
                    with st.spinner("Running backtesting..."):
                        # Create a simple VaR function for backtesting
                        def var_function(returns, conf_level, horizon):
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
        
        with tab7:  # Options Analysis
            st.header("📊 Options Analysis")
            
            if portfolio_type == "Options Portfolio":
                # Options VaR calculation
                st.subheader("Options VaR Analysis")
                
                options_method = st.selectbox(
                    "Select Options VaR Method",
                    ["Delta-Normal", "Delta-Gamma", "Full Revaluation Monte Carlo"],
                    key="options_method_select"
                )
                
                if st.button("🔄 Calculate Options VaR", key="calculate_options_var_button"):
                    with st.spinner("Calculating Options VaR..."):
                        options_results = instances['options_var'].calculate_options_var(
                            spot_price, strike_price, time_to_expiry, risk_free_rate,
                            volatility, option_type.lower(), options_method, confidence_level
                        )
                        
                        if options_results:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Options VaR Results")
                                options_metrics = {
                                    'Options VaR': f"${options_results.get('var', 0):,.2f}",
                                    'Current Option Price': f"${options_results.get('current_price', 0):.2f}",
                                    'Method': options_results.get('method', 'N/A'),
                                    'Delta': f"{options_results.get('delta', 0):.4f}",
                                    'Gamma': f"{options_results.get('gamma', 0):.6f}",
                                    'Theta': f"${options_results.get('theta', 0):.2f}",
                                    'Vega': f"${options_results.get('vega', 0):.2f}"
                                }
                                options_df = create_metrics_dataframe(options_metrics, "Options VaR Results")
                                safe_dataframe_display(options_df, "Options VaR Results", "options_results")
                            
                            with col2:
                                st.subheader("Option Payoff Diagram")
                                S_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
                                fig_payoff = instances['visualization'].plot_options_payoff(
                                    S_range, strike_price, option_type.lower(), time_to_expiry, risk_free_rate, volatility
                                )
                                st.plotly_chart(fig_payoff, use_container_width=True, key="options_payoff_chart")
            else:
                st.info("Options analysis is available when 'Options Portfolio' is selected in the sidebar.")
                
                # Show example options configuration
                st.subheader("Example Options Configuration")
                example_options = {
                    'Underlying': 'AAPL',
                    'Spot Price': '$150.00',
                    'Strike Price': '$155.00',
                    'Option Type': 'Call',
                    'Time to Expiry': '0.25 years (3 months)',
                    'Risk-free Rate': '5%',
                    'Volatility': '25%',
                    'Quantity': '100 contracts'
                }
                example_df = create_metrics_dataframe(example_options, "Example Options")
                safe_dataframe_display(example_df, "Example Options Configuration", "example_options")
        
        with tab8:  # Help
            st.header("❓ Help & Documentation")
            
            st.markdown("""
            ## 🚀 Welcome to the VaR & Risk Analytics Platform
            
            This comprehensive platform provides sophisticated financial risk modeling capabilities for portfolio management and risk assessment.
            
            ### 📊 Getting Started
            
            1. **Select Data Source**: Choose from Live Market Data, Upload File, Manual Entry, or Synthetic Data
            2. **Configure Portfolio**: Define portfolio type and asset allocation
            3. **Choose VaR Model**: Select from 5 sophisticated VaR calculation methods
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
            - **Default**: AAPL call option (Strike: $155, Expiry: 3 months)
            - **Live Market Format**: Use underlying symbol (e.g., AAPL) and configure parameters
            
            ### 📈 VaR Models
            
            1. **Parametric (Delta-Normal)**: Classical normal distribution approach
            2. **Historical Simulation**: Non-parametric historical method
            3. **Monte Carlo**: Simulation-based approach (1K-100K simulations)
            4. **GARCH**: Advanced volatility modeling for time-varying risk
            5. **Extreme Value Theory (EVT)**: Tail risk modeling for extreme events
            
            ### 📁 File Upload Formats
            
            #### CSV Format
            ```
            Date,Asset1,Asset2,Asset3
            2023-01-01,100.50,200.25,150.75
            2023-01-02,101.25,198.50,152.00
            2023-01-03,99.75,201.00,149.25
            ```
            
            #### Excel Format
            - Same structure as CSV
            - First column: Date (YYYY-MM-DD format)
            - Subsequent columns: Asset prices
            - Headers recommended for clarity
            
            ### 🎯 Key Features
            
            #### Individual Graph Time Controls
            - Each tab has independent time range selectors
            - Default 1-year period with customizable start/end dates
            - Automatic data filtering for selected periods
            
            #### Data Persistence
            - Generated data survives model changes
            - Only calculations update when switching VaR models
            - Session state maintains data across interactions
            
            #### Real-time Updates
            - All tabs update dynamically when parameters change
            - Model switching preserves loaded data
            - Consistent results across different methodologies
            
            ### 🔧 Configuration Tips
            
            #### Risk Parameters
            - **Confidence Levels**: 90%, 95%, 99% (95% is standard)
            - **Time Horizons**: 1-30 days (1-day most common)
            - **Historical Windows**: 30-1000 days (252 days = 1 trading year)
            
            #### Model-Specific Settings
            - **Monte Carlo**: 10K simulations recommended for balance of speed/accuracy
            - **GARCH**: (1,1) specification is industry standard
            - **Backtesting**: 252-day window provides 1 year of validation
            
            ### 🚨 Troubleshooting
            
            #### Common Issues
            - **"Insufficient data"**: Increase historical window or data period
            - **"GARCH model failed"**: Requires minimum 100 observations
            - **"Symbol not found"**: Verify ticker format (add -USD for crypto)
            - **"Weights don't sum to 1"**: Portfolio weights are automatically normalized
            
            #### Cryptocurrency Issues
            - **API Failures**: System automatically generates realistic synthetic crypto data
            - **Symbol Format**: Ensure crypto symbols end with -USD (e.g., BTC-USD)
            - **Data Quality**: Fallback provides realistic price movements for analysis
            
            ### 📊 Output Interpretation
            
            #### VaR Results
            - **VaR(95%, 1-day) = $10,000**: 95% confidence that daily losses won't exceed $10,000
            - **Expected Shortfall**: Average loss when VaR threshold is breached
            - **Model Comparison**: Side-by-side results from different methodologies
            
            #### Backtesting Metrics
            - **Kupiec p-value > 0.05**: Model passes statistical validation
            - **Violation Rate**: Should approximate (1 - confidence level)
            - **Basel Traffic Light**: Green (good), Yellow (attention), Red (review required)
            
            ### 🎓 Academic Background
            
            The platform implements industry-standard methodologies based on:
            - **Basel Committee**: International regulatory framework for VaR
            - **RiskMetrics**: J.P. Morgan's technical document (1996)
            - **Extreme Value Theory**: Embrechts, Klüppelberg, and Mikosch
            - **GARCH Models**: Engle (1982), Bollerslev (1986)
            - **Options Pricing**: Black-Scholes-Merton model
            
            ### 💡 Best Practices
            
            1. **Start Simple**: Begin with single asset and parametric VaR
            2. **Validate Models**: Always run backtesting to verify model performance
            3. **Compare Methods**: Use multiple VaR models for robust risk assessment
            4. **Stress Test**: Regular stress testing reveals portfolio vulnerabilities
            5. **Monitor Regularly**: Risk metrics should be updated frequently
            
            ### 📞 Support
            
            For additional support or questions about specific features, refer to the academic literature or consult with risk management professionals.
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
