# VaR & Risk Analytics Platform

A comprehensive Value at Risk (VaR) and risk analytics platform built with Streamlit and Python. This application provides sophisticated financial risk modeling capabilities for portfolio management and risk assessment with support for stocks, cryptocurrencies, and derivatives.

## 🚀 Features

### 🔄 Data Ingestion
- **Live Market Data**: Integration with Yahoo Finance for real-time market data
- **Cryptocurrency Support**: Native support for crypto assets (BTC-USD, ETH-USD, etc.)
- **File Upload**: Support for CSV/XLSX file uploads with flexible formatting
- **Synthetic Data Generation**: Create realistic market data for testing and analysis
- **Manual Entry**: Direct data input for custom portfolios
- **Portfolio Types**: Single asset, multi-asset, crypto portfolio, and options portfolios

### 📊 Advanced VaR Engines
- **Parametric (Delta-Normal)**: Classical normal distribution approach with Cornish-Fisher adjustments
- **Historical Simulation**: Non-parametric historical method capturing actual market behavior
- **Monte Carlo**: Simulation-based approach with customizable parameters (1K-100K simulations)
- **GARCH-Based**: Advanced volatility modeling (GARCH(1,1), EGARCH) for time-varying volatility
- **Extreme Value Theory (EVT)**: Tail risk modeling for extreme market events

### 📈 Comprehensive Risk Metrics
- **Expected Shortfall/CVaR**: Conditional Value at Risk calculations
- **Cornish-Fisher Adjustments**: Skewness and kurtosis corrections for non-normal distributions
- **Risk Contributions**: Marginal VaR and component ES analysis
- **Rolling Analysis**: Time-varying risk metrics with customizable windows
- **Drawdown Analysis**: Maximum drawdown and recovery period calculations

### 🧪 Robust Backtesting & Validation
- **Kupiec's POF Test**: Unconditional coverage testing for model validation
- **Christoffersen's Tests**: Independence and conditional coverage testing
- **Basel Traffic Light System**: Regulatory compliance assessment (Green/Yellow/Red zones)
- **Hit Ratio Analysis**: Violation tracking over time with statistical significance
- **Model Comparison**: Side-by-side performance evaluation of different VaR models

### ⚡ Advanced Stress Testing & Scenario Analysis
- **Historical Scenarios**: Pre-built scenarios (2008 Crisis, COVID-19, Dot-com Crash)
- **Custom Shock Generator**: User-defined parameter shocks for volatility, correlation, and market moves
- **Correlation Stress**: Dynamic correlation matrix adjustments
- **Sensitivity Analysis**: Multi-dimensional stress testing across parameter ranges
- **Tail Risk Analysis**: Extreme event modeling with multiple confidence levels

### 📊 Sophisticated Options Portfolio VaR
- **Delta-Normal Approach**: Linear approximation method for quick estimates
- **Delta-Gamma Method**: Second-order Taylor expansion for improved accuracy
- **Full Revaluation Monte Carlo**: Complete option repricing for maximum precision
- **Greeks Analysis**: Comprehensive sensitivity metrics (Delta, Gamma, Theta, Vega, Rho)
- **Multi-Option Portfolios**: Complex derivatives portfolio risk assessment

### 📱 Professional Interactive Dashboard
- **Dark Mode Interface**: Professional financial application styling optimized for extended use
- **Real-time Updates**: All tabs update dynamically when parameters change
- **Interactive Charts**: Professional Plotly visualizations with zoom, pan, and export capabilities
- **Comprehensive Controls**: Extensive parameter customization with intelligent defaults
- **Data Persistence**: Generated data persists across model changes
- **Export Capabilities**: CSV, JSON outputs with detailed metadata

### 🎯 Dynamic Model Integration
- **Real-time Model Switching**: All calculations update instantly when VaR model changes
- **Consistent Results**: Same data fed to different models for accurate comparison
- **Model-Specific Insights**: Contextual information and recommendations for each method
- **Performance Optimization**: Intelligent caching and computation management

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd var-risk-analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## 📊 Usage Guide

### Basic Setup
1. **Data Source Selection**: Choose from Live Market Data, File Upload, or Manual Entry
2. **Portfolio Configuration**: Define portfolio type and asset allocation
3. **VaR Model Selection**: Choose from 5 sophisticated VaR calculation methods
4. **Risk Parameters**: Set confidence levels (90-99%), time horizons (1-30 days), and historical windows

### Data Sources

#### Live Market Data
- **Stocks**: Use standard ticker symbols (AAPL, GOOGL, MSFT, TSLA)
- **Cryptocurrencies**: Add -USD suffix (BTC-USD, ETH-USD, ADA-USD)
- **Mixed Portfolios**: Combine stocks and crypto in multi-asset portfolios
- **Default Configurations**:
  - Single Asset: AAPL
  - Multi-Asset: AAPL, GOOGL, MSFT, TSLA
  - Crypto Portfolio: BTC-USD

#### File Upload Format
```csv
Date,Asset1,Asset2,Asset3
2023-01-01,100.50,200.25,150.75
2023-01-02,101.25,198.50,152.00
2023-01-03,99.75,201.00,149.25
```
- First column: Date (YYYY-MM-DD format)
- Subsequent columns: Asset prices (any number of assets)
- Headers recommended for clarity

#### Synthetic Data Generation
- **Realistic Market Simulation**: Geometric Brownian Motion with customizable parameters
- **Configurable Parameters**: Initial price, annual return, volatility, time period
- **Sufficient Data**: Generates enough data points for all analysis types
- **Reproducible**: Seed-based generation for consistent results

### Advanced Features

#### VaR Model Selection
- **Parametric**: Best for stable markets with normal return distributions
- **Historical**: Captures actual market behavior without distributional assumptions
- **Monte Carlo**: Flexible simulation approach with user-defined complexity
- **GARCH**: Handles volatility clustering and time-varying risk
- **EVT**: Specialized for tail risk and extreme market events

#### Portfolio Management
- **Weight Optimization**: Automatic normalization and validation
- **Multi-Asset Support**: Up to unlimited assets with custom allocations
- **Crypto Integration**: Seamless mixing of traditional and digital assets
- **Rebalancing**: Dynamic weight adjustments with real-time updates

#### Risk Analysis
- **Rolling Windows**: 30-1000 day analysis periods
- **Multiple Confidence Levels**: 90-99% with 95% as standard
- **Time Horizons**: 1-30 day risk periods
- **Advanced Metrics**: Skewness, kurtosis, and higher-moment adjustments

### Professional Reporting
- **Executive Summaries**: High-level risk overview for management
- **Technical Reports**: Detailed analysis for risk professionals
- **Regulatory Compliance**: Basel-compliant backtesting and validation
- **Custom Exports**: Flexible data export in multiple formats

## 🏗️ Technical Architecture

### Core Modules
- `app.py`: Main Streamlit application with dynamic UI
- `data_ingestion.py`: Multi-source data loading with crypto support
- `var_engines.py`: Five sophisticated VaR calculation methodologies
- `backtesting.py`: Comprehensive model validation and testing
- `stress_testing.py`: Scenario analysis and stress testing framework
- `rolling_analysis.py`: Time-series risk analytics and metrics
- `options_var.py`: Advanced derivatives risk modeling
- `visualization.py`: Professional interactive plotting and charts
- `utils.py`: Utility functions and portfolio analytics

### Key Dependencies
- **Streamlit**: Modern web application framework
- **Pandas/NumPy**: High-performance data manipulation
- **SciPy**: Advanced statistical functions and optimization
- **Plotly**: Interactive, publication-quality visualizations
- **YFinance**: Real-time market data API
- **ARCH**: Professional GARCH modeling for volatility
- **OpenPyXL**: Excel file handling and export

### Performance Features
- **Intelligent Caching**: Reduces API calls and computation time
- **Lazy Loading**: Data loaded only when needed
- **Optimized Calculations**: Vectorized operations for speed
- **Memory Management**: Efficient handling of large datasets

## 📈 Model Specifications

### VaR Methodologies
1. **Parametric VaR**: σ√t × Φ⁻¹(α) with optional Cornish-Fisher adjustments
2. **Historical VaR**: Empirical quantile estimation from historical returns
3. **Monte Carlo VaR**: Simulation-based with user-defined iterations
4. **GARCH VaR**: Conditional volatility modeling with GARCH(p,q)
5. **EVT VaR**: Generalized Pareto Distribution for tail estimation

### Validation Framework
- **Kupiec Test**: Likelihood ratio test for unconditional coverage
- **Christoffersen Test**: Independence and conditional coverage testing
- **Basel Framework**: Traffic light system for regulatory compliance
- **Hit Ratio Analysis**: Time-series violation pattern analysis

### Stress Testing
- **Historical Scenarios**: Calibrated to major market events
- **Custom Shocks**: User-defined volatility, correlation, and market moves
- **Sensitivity Analysis**: Multi-dimensional parameter stress testing
- **Tail Risk**: Extreme value analysis for crisis scenarios

## 🔧 Configuration Options

### Risk Parameters
- **Confidence Levels**: 90%, 95%, 99% (industry standard options)
- **Time Horizons**: 1-30 days (1-day most common for daily VaR)
- **Historical Windows**: 30-1000 days (252 days = 1 trading year)
- **Decay Factors**: 0.90-0.99 (0.94 is RiskMetrics standard)

### Model-Specific Settings
- **Monte Carlo**: 1K-100K simulations (10K recommended balance)
- **GARCH**: P and Q parameters (1,1 is standard specification)
- **EVT**: Threshold percentiles for extreme value fitting
- **Options**: Greeks calculation with multiple sensitivity measures

## 📊 Output Interpretation

### VaR Results
- **VaR(95%, 1-day) = $10,000**: 95% confidence that daily losses won't exceed $10,000
- **Expected Shortfall**: Average loss when VaR threshold is breached
- **Model Comparison**: Side-by-side results from different methodologies

### Backtesting Metrics
- **Kupiec p-value > 0.05**: Model passes statistical validation
- **Violation Rate**: Should approximate (1 - confidence level)
- **Basel Traffic Light**: Green (good), Yellow (attention), Red (review required)

### Stress Testing
- **VaR Increase**: Percentage change under stress scenarios
- **Worst Case**: 1st percentile loss under extreme conditions
- **Scenario Impact**: Comparative analysis across different stress events

## 🚨 Troubleshooting

### Common Issues
- **"Insufficient data"**: Increase historical window or data period
- **"GARCH model failed"**: Requires minimum 100 observations
- **"Symbol not found"**: Verify ticker format (add -USD for crypto)
- **"Weights don't sum to 1"**: Portfolio weights automatically normalized

### Performance Optimization
- **Reduce simulations**: Use fewer Monte Carlo iterations for speed
- **Smaller windows**: Decrease historical window for faster calculations
- **Cache utilization**: Same parameters load from cache automatically

## 📚 Academic References

### Methodology Sources
- **Basel Committee**: International regulatory framework for VaR
- **RiskMetrics**: J.P. Morgan's technical document (1996)
- **Extreme Value Theory**: Embrechts, Klüppelberg, and Mikosch
- **GARCH Models**: Engle (1982), Bollerslev (1986)

### Industry Standards
- **Basel III**: Current international regulatory framework
- **Solvency II**: European insurance regulation
- **FRTB**: Fundamental Review of Trading Book

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Feature requests and bug reports

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔬 **VaR & Risk Analytics Platform** | Built with Streamlit | © 2024

---

*Professional risk management tools for modern financial markets*