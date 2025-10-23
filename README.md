👉 Main library: [asxshorts](https://github.com/ay-mich/asxshorts)

# ASX Shorts Examples

This repository contains comprehensive examples and tutorials for the [`asxshorts`](https://pypi.org/project/asxshorts/) Python package - a simple and efficient library for fetching Australian Securities Exchange (ASX) short selling data.

## 📊 What is ASX Short Selling Data?

Short selling data from the Australian Securities Exchange (ASX) provides insights into:

- **Short Interest**: Percentage of shares sold short for each stock
- **Market Sentiment**: Bearish sentiment indicators
- **Trading Patterns**: Daily changes in short positions
- **Risk Analysis**: Potential squeeze scenarios and volatility indicators

This data is published daily by ASIC (Australian Securities and Investments Commission) and covers all ASX-listed securities.

## 🚀 Quick Start

### Installation

```bash
pip install asxshorts
```

### Basic Usage

```python
from asxshorts import ShortsClient
from datetime import date, timedelta

# Create a client
client = ShortsClient()

# Fetch data for yesterday
ten_days_ago = date.today() - timedelta(days=10)
result = client.fetch_day(ten_days_ago)

print(f"Found {result.record_count} records")
for record in result.records[:5]:  # Show first 5
    print(f"{record.asx_code}: {record.percent_short}% short")
```

## 📁 Repository Structure

```
├── examples/
│   ├── 01_basic_usage.py          # Getting started with the basics
│   ├── 02_data_analysis.py        # Data analysis and visualization
│   ├── 03_pandas_integration.py   # Working with pandas DataFrames
│   ├── 04_polars_integration.py   # Working with polars DataFrames
│   └── 05_advanced_features.py    # Advanced features and production patterns
├── requirements.txt               # Dependencies for examples
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## 📚 Examples Overview

### Python Scripts

| Script                     | Description                                  | Key Features                                             |
| -------------------------- | -------------------------------------------- | -------------------------------------------------------- |
| `01_basic_usage.py`        | Essential operations and basic API usage     | Client setup, single day fetch, date ranges, caching     |
| `02_data_analysis.py`      | Data analysis and visualization examples     | Matplotlib/seaborn charts, trend analysis, statistics    |
| `03_pandas_integration.py` | Integration with pandas ecosystem            | DataFrames, filtering, aggregations, Excel export        |
| `04_polars_integration.py` | High-performance data processing with polars | Fast operations, lazy evaluation, performance comparison |
| `05_advanced_features.py`  | Advanced features and production patterns    | Cache management, error handling, validation, monitoring |

## 🔧 Package Features Covered

- **Core Functionality**

  - Fetching single-day data
  - Date range queries
  - Data validation and parsing

- **Data Integration**

  - Pandas DataFrame conversion
  - Polars DataFrame conversion
  - Custom data adapters

- **Performance & Caching**

  - Automatic caching system
  - Cache management utilities
  - Performance optimization tips

- **Error Handling**
  - Network error recovery
  - Data validation errors
  - Rate limiting handling

## 🏃‍♂️ Running the Examples

### Prerequisites

```bash
# Clone this repository
git clone <your-repo-url>
cd asxshorts-examples

# Install dependencies
pip install -r requirements.txt
```

### Running Python Scripts

```bash
# Run any example script
python examples/01_basic_usage.py

# Or run all examples
for script in examples/*.py; do
    echo "Running $script..."
    python "$script"
    echo "---"
done
```

## 📄 License

This examples repository is provided under the MIT License. See `LICENSE` file for details.

The `asxshorts` package itself may have different licensing terms.

## 🔗 Links

- [asxshorts on PyPI](https://pypi.org/project/asxshorts/)
- [ASIC Short Selling Data](https://download.asic.gov.au/short-selling/)
- [ASX Official Website](https://www.asx.com.au/)

---

**Note**: This package fetches publicly available data from ASIC. Always ensure compliance with data usage terms and conditions.
