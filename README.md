# AutoMart-AI

> An end-to-end retail analytics platform combining ML models, intelligent agents, and an interactive dashboard to give supermarket managers actionable insights on customers, inventory, profit, and product recommendations.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Agents](#-agents)
- [Dashboard](#-dashboard)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Flow](#-data-flow)
- [Configuration](#-configuration)
- [Dependencies](#-dependencies)

---

##  Overview

Supermarket AI is built around three core pillars:
https://automart-ai-mpworqt3pcf76dsgxam9bl.streamlit.app/

| Pillar | Description |
|--------|-------------|
| **ML Models** | Train on historical sales data to find patterns, customer segments, and sales forecasts |
| **Intelligent Agents** | Wrap models with business logic to produce human-readable, actionable insights |
| **Interactive Dashboard** | A Streamlit web app that presents all outputs visually for non-technical managers |

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data | `pandas`, CSV files | Load, clean, and engineer features |
| ML Models | `scikit-learn`, `mlxtend`, `PyTorch` | Clustering, association rules, forecasting |
| Agents | Python + LangChain (optional) | Business logic + AI insights |
| Dashboard | Streamlit | Interactive web UI |
| AI Insights | LangChain + Google Gemini | LLM-powered marketing suggestions |

---

##  Architecture

```
CSV Files
  supermarket_dummy_data.csv ──→ utils/preprocessing.py ──→ Enriched DataFrame
  transaction_dataset.csv    ──→ models/apriori_model.py ──→ Association Rules

Enriched DataFrame
  ├──→ models/kmeans_model.py    ──→ Customer Clusters
  │      └──→ agents/customer_agent.py   ──→ High/Low Value + Insights
  │
  ├──→ models/lstm_pytorch.py    ──→ Next-Day Sales Prediction
  │
  ├──→ agents/inventory_agent.py ──→ Demand / Stock / Festival Insights
  │
  └──→ agents/profit_agent.py    ──→ Profit / Loss Analysis

Association Rules
  └──→ agents/recommendation_agent.py ──→ Product Bundle Suggestions

All Agent Outputs
  ├──→ app/dashboard.py   (Streamlit UI)
  └──→ main.py            (CLI output)
```

---

##  Project Structure

```
supermarket-ai/
│
├── data/
│   ├── supermarket_dummy_data.csv   # Main sales dataset
│   └── transaction_dataset.csv      # Transaction records for basket analysis
│
├── utils/
│   └── preprocessing.py             # Data loading & feature engineering pipeline
│
├── models/
│   ├── kmeans_model.py              # Customer segmentation (K-Means)
│   ├── apriori_model.py             # Product recommendations (Apriori)
│   └── lstm_pytorch.py              # Sales forecasting (LSTM)
│
├── agents/
│   ├── customer_agent.py            # Customer value segmentation + insights
│   ├── inventory_agent.py           # Demand, stock, and festival tracking
│   ├── profit_agent.py              # Product and store-level profit analysis
│   └── recommendation_agent.py      # Cross-sell and bundling opportunities
│
├── app/
│   └── dashboard.py                 # Streamlit web dashboard
│
├── main.py                          # CLI entry point
└── requirements.txt                 # Python dependencies
```

---

##  Models

### K-Means Customer Segmentation (`models/kmeans_model.py`)

Groups customers into **Low / Medium / High Value** segments based on purchase behaviour.

**Pipeline:** Aggregate by Customer ID → Normalize with `StandardScaler` → Apply `KMeans(n_clusters=3)` → Rank by profit

| Function | Purpose |
|----------|---------|
| `prepare_customer_data(df)` | Aggregates raw rows into one row per customer |
| `scale_features(customer_df)` | Normalises Quantity_Sold and Profit |
| `apply_kmeans(customer_df)` | Runs KMeans and appends Cluster column |
| `segment_customers(df)` | Full pipeline: prepare → scale → cluster |
| `label_clusters(df)` | Adds human-readable Segment column |
| `cluster_summary(df)` | Mean quantity, mean profit, and count per cluster |

---

### Apriori Association Rules (`models/apriori_model.py`)

Answers the question: *"If a customer buys Product A, how likely are they to buy Product B?"*

| Metric | Meaning |
|--------|---------|
| **Support** | How often an itemset appears across all transactions |
| **Confidence** | How often B is bought given A is bought |
| **Lift** | How much more likely B is when A is present vs. by chance (Lift > 1 = genuine association) |

**Pipeline:** Load `transaction_dataset.csv` → Build basket matrix → Run Apriori (`min_support=0.01`) → Generate rules (`min_lift=1`) → Sort by lift

---

### LSTM Sales Forecasting (`models/lstm_pytorch.py`)

Predicts **next-day total sales quantity** using a recurrent neural network trained on historical daily sales.

**Model Architecture:**
```
Input: [batch, 7 days, 1 feature]
  → LSTM(input_size=1, hidden_size=50, batch_first=True)
  → Linear(50 → 1)
Output: Predicted next-day quantity (single float)
```

**Pipeline:** Aggregate daily sales → Create 7-day sliding window sequences → Train for 10 epochs (MSELoss + Adam) → Predict from last 7 days

---

##  Agents

Each agent exposes a single `*_agent_summary()` master function used by both the dashboard and CLI. Models are pure ML; agents add the business interpretation on top.

### Customer Agent (`agents/customer_agent.py`)
Segments customers and generates marketing strategies. Uses **Google Gemini** via LangChain if `GEMINI_API_KEY` is set, otherwise falls back to rule-based logic:
- Profit > 500 → Offer loyalty rewards
- Profit < 200 → Provide discounts
- Otherwise → Target with promotions

### Inventory Agent (`agents/inventory_agent.py`)
Tracks product demand, flags low-stock items, calculates restock quantities, and surfaces festival-driven demand spikes.

**Restock Formula:**
```
If Quantity_Sold < 100:  restock = 150 - Quantity_Sold
Else:                    restock = 0   (no action needed)
```

### Profit Agent (`agents/profit_agent.py`)
Analyses profitability at the product level and store level. Identifies top performers and products running at a loss.

### Recommendation Agent (`agents/recommendation_agent.py`)
Uses the Apriori model to surface product bundling and cross-sell opportunities. Reads from `transaction_dataset.csv`.

---

##  Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

**Page 1 — Sales & Inventory**
- Sales trend line chart
- One-click LSTM next-day sales prediction
- Top profitable products bar chart
- Low-stock alerts and restock suggestions
- Festival demand highlights

**Page 2 — Customers & Recommendations**
- Interactive customer segmentation table
- High-value and low-value customer breakdowns
- AI-powered marketing insights
- Top product association rules

---

##  Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/supermarket-ai.git
cd supermarket-ai
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. (Optional) Set your Google Gemini API key for AI-powered insights**
```bash
export GEMINI_API_KEY="your-api-key-here"   # macOS/Linux
set GEMINI_API_KEY=your-api-key-here        # Windows
```

---

##  Usage

### Run the Dashboard (recommended)
```bash
streamlit run app/dashboard.py
```

### Run the Full Pipeline via CLI
```bash
python main.py
```

The CLI runner executes every pipeline step in order and prints results to the terminal. Each step is wrapped in `try/except` so a single failure won't crash the entire run.

| Step | What Runs | Output |
|------|-----------|--------|
| 1 | `preprocess_data()` | Loads and enriches the CSV |
| 2 | `run_lstm(df)` | Trains LSTM, prints predicted next-day sales |
| 3 | `label_clusters(df)` | Prints customer segments and cluster stats |
| 4 | `get_rules()` | Prints top Apriori association rules |
| 5 | `inventory_agent_summary(df)` | Prints demand, low stock, and alerts |
| 6 | `customer_agent_summary(df)` | Prints high-value customers and insights |
| 7 | `profit_agent_summary(df)` | Prints profit totals and loss warnings |
| 8 | `recommendation_agent_summary()` | Prints top product bundle rules |

---

##  Data Flow

The preprocessing module is called **once** at startup — both `main.py` and `dashboard.py` pass the same enriched DataFrame to every downstream component, avoiding duplicate file reads.

| Component | Depends On | Feeds Into |
|-----------|-----------|-----------|
| `preprocessing.py` | Raw CSV files | All models and agents |
| `kmeans_model.py` | `preprocessing.py` output | `customer_agent.py`, `dashboard.py` |
| `apriori_model.py` | `transaction_dataset.csv` | `recommendation_agent.py`, `dashboard.py` |
| `lstm_pytorch.py` | `preprocessing.py` output | `dashboard.py`, `main.py` |
| `customer_agent.py` | `kmeans_model.py` | `dashboard.py`, `main.py` |
| `inventory_agent.py` | `preprocessing.py` output | `dashboard.py`, `main.py` |
| `profit_agent.py` | `preprocessing.py` output | `dashboard.py`, `main.py` |
| `recommendation_agent.py` | `apriori_model.py` | `dashboard.py`, `main.py` |

---

##  Configuration

The system works out of the box without any configuration. The only optional setting is:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key for LLM-powered marketing insights in the Customer Agent | Falls back to rule-based insights |

---

##  Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.56.0 | Web dashboard framework |
| `pandas` | 3.0.2 | Data loading, manipulation, and aggregation |
| `numpy` | 2.4.4 | Numerical operations and array handling |
| `torch` | 2.11.0 | LSTM model training and inference |
| `scikit-learn` | 1.8.0 | KMeans clustering and StandardScaler |
| `mlxtend` | 0.24.0 | Apriori algorithm and association rules |
| `langchain-core` | 1.3.2 | Agent framework for LLM-powered insights |
| `matplotlib` | 3.10.9 | Charting support |
| `scipy` | 1.17.1 | Scientific computing (used by mlxtend) |

---

##  Key Design Decisions

- **Single preprocessing call** — `preprocess_data()` runs once; the enriched DataFrame is passed to every component.
- **Agent pattern** — Each agent exposes one `*_summary()` master function, keeping the dashboard and CLI interfaces clean.
- **Graceful AI fallback** — The Customer Agent tries Google Gemini first but always falls back to rule-based insights, so the system works without any API key.
- **Models vs. Agents separation** — Models are pure ML (no business logic). Agents add the business interpretation on top of model outputs.
- **Error isolation** — Every step in `main.py` is wrapped in `try/except` so a single model failure doesn't crash the entire pipeline.

---


*Built with ❤️ using Python, PyTorch, and Streamlit*
