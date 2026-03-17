# Customer Feedback Analysis Project

## Project Structure

```
task4-customer-feedback/
├── design/
│   ├── images/                  # Screenshots, exported diagrams
│   ├── drawio/                  # .drawio pipeline/architecture diagrams
│   └── plantuml/                # Future PlantUML files
│
├── data/
│   ├── raw/                     # Original, untouched dataset
│   ├── processed/               # Cleaned & preprocessed data
│   └── samples/                 # Small samples for quick testing
│
├── notebooks/
│   ├── 01_eda/                  # Exploratory Data Analysis
│   ├── 02_clustering/           # k-means & HAC experiments
│   ├── 03_text_representation/  # TF-IDF vs Embeddings
│   ├── 04_pipeline/             # Full analytics pipeline
│   └── 05_evaluation/           # Metrics, error analysis
│
├── src/
│   ├── data/                    # Data loading & preprocessing
│   ├── features/                # TF-IDF, embedding extractors
│   ├── clustering/              # k-means, HAC implementations
│   ├── pipeline/                # Main analytics pipeline
│   ├── evaluation/              # Scoring & metrics
│   └── utils/                   # Shared helpers
│
├── reports/
│   ├── figures/                 # Saved plots for the report
│   └── tables/                  # Result tables
│
├── tests/                       # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Overview

This project analyzes customer feedback using various text analysis and clustering techniques.

## Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Descriptions

- **design/**: Contains visual assets and diagrams
- **data/**: All data files, organized by processing stage
- **notebooks/**: Jupyter notebooks for each stage of analysis
- **src/**: Python source code organized by functionality
- **reports/**: Generated figures and tables for reporting
- **tests/**: Unit tests for code validation