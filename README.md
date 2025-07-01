# VaR & Risk Analytics Platform

A comprehensive Value at Risk (VaR) and risk analytics platform built with Streamlit and Python. This application provides sophisticated financial risk modeling capabilities for portfolio management and risk assessment.

## Features

### 🔄 Data Ingestion
- **Live Market Data**: Integration with Yahoo Finance for real-time market data
- **File Upload**: Support for CSV/XLSX file uploads
- **Manual Entry**: Direct data input for custom portfolios
- **Portfolio Types**: Single asset, multi-asset, and options portfolios

### 📊 Risk Engines (VaR Calculation)
- **Parametric (Delta-Normal)**: Classical normal distribution approach
- **Historical Simulation**: Non-parametric historical method
- **Monte Carlo**: Simulation-based approach with customizable parameters
- **GARCH-Based**: Advanced volatility modeling (GARCH(1,1), EGARCH)
- **Extreme Value Theory (EVT)**: Tail risk modeling for extreme events

### 📈 Advanced Risk Metrics
- **Expected Shortfall/CVaR**: Conditional Value at Risk calculations
- **Cornish-Fisher Adjustments**: Skewness and kurtosis corrections
- **Risk Contributions**: Marginal VaR and component ES analysis
- **Rolling Analysis**: Time-varying risk metrics

### 🧪 Backtesting & Validation
- **Kupiec's POF Test**: Unconditional coverage testing
- **Christoffersen's Tests**: Independence and conditional coverage
- **Basel Traffic Light System**: Regulatory compliance assessment
- **Hit Ratio Analysis**: Violation tracking over time

### ⚡ Stress Testing & Scenario Analysis
- **Historical Scenarios**: 2008 Crisis, COVID-19, Dot-com Crash
- **Custom Shock Generator**: User-defined parameter shocks
- **Correlation Stress**: Dynamic correlation matrix adjustments
- **Sensitivity Analysis**: Multi-dimensional stress testing

### 📊 Options Portfolio VaR
- **Delta-Normal Approach**: Linear approximation method
- **Delta-Gamma Method**: Second-order Taylor expansion
- **Full Revaluation Monte Carlo**: Complete option repricing
- **Greeks Analysis**: Comprehensive sensitivity metrics

### 📱 Interactive Dashboard
- **Dark Mode Interface**: Professional financial application styling
- **Real-time Charts**: Interactive Plotly visualizations
- **Comprehensive Controls**: Extensive parameter customization
- **Export Capabilities**: PDF, Excel, CSV, and JSON outputs

## Installation

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

## Usage

### Basic Setup
1. **Data Source**: Select your preferred data source (Live Market Data, File Upload, or Manual Entry)
2. **Portfolio Configuration**: Define your portfolio type and asset allocation
3. **Risk Parameters**: Set confidence levels, time horizons, and model parameters
4. **Model Selection**: Choose your preferred VaR calculation method

### Advanced Features
- **Rolling Analysis**: Configure rolling windows for dynamic risk assessment
- **Stress Testing**: Apply historical or custom scenarios to test portfolio resilience
- **Options Analysis**: Analyze complex derivatives portfolios with Greeks
- **Backtesting**: Validate model performance with historical data

### Reporting
- **Executive Summaries**: High-level risk overview for management
- **Detailed Reports**: Comprehensive technical analysis
- **Custom Exports**: Flexible data export in multiple formats
- **Visualization**: Professional charts and graphs for presentations

## Technical Architecture

### Core Modules
- `data_ingestion.py`: Data loading and preprocessing
- `var_engines.py`: VaR calculation methodologies
- `backtesting.py`: Model validation and testing
- `stress_testing.py`: Scenario analysis and stress testing
- `rolling_analysis.py`: Time-series risk analytics
- `options_var.py`: Derivatives risk modeling
- `visualization.py`: Interactive plotting and charts
- `utils.py`: Utility functions and helpers

### Dependencies
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation and numerical computing
- **SciPy**: Statistical functions and optimization
- **Plotly**: Interactive visualizations
- **YFinance**: Market data API
- **ARCH**: GARCH modeling for volatility
- **OpenPyXL**: Excel file handling

