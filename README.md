# Fragmentation Risk Detector

A high-frequency trading surveillance system that detects liquidity fragmentation and spoofing patterns across 20,000+ order book snapshots, with slippage risk simulation and microsecond-level anomaly detection.

---

## 🚀 Features

- **Liquidity Fragmentation Radar:** Flags volume dispersion and spread anomalies across order books.
- **Spoofing Detection:** Identifies hidden bid-ask depth manipulation patterns at microsecond resolution.
- **Slippage Simulator:** Quantifies execution risks under uneven liquidity (30% slippage risk identification).
- **Real-Time Alerts:** Generates timing flags to optimize trade execution and reduce fill risk.
- **Microstructure Analysis:** Processes 20,000+ order book snapshots with millisecond timestamps.

---

## 🏗️ Architecture Overview

| Component          | Functionality                              |
|--------------------|--------------------------------------------|
| Data Ingestion     | Order book snapshot parser (CSV/WebSocket) |
| Risk Engine        | Volume dispersion analyzer + spread model  |
| Spoofing Detector  | Bid-ask depth shift pattern recognition    |
| Simulation Module  | Trade execution path simulator             |
| Alert System       | Slippage risk flags and timing suggestions |

---

## 📦 Installation

1. **Clone the Repository**
   ```
   git clone <repo-url>
   cd fragmentation-risk-detector
   ```

2. **Set Up Virtual Environment**
   ```
   python -m venv env
   source env/bin/activate
   ```

3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Configure Market Data Paths**
   ```
   cp config_sample.yaml config.yaml
   # Edit config.yaml with your order book data paths
   ```

---

## 🖥️ Usage

1. **Run Analysis Pipeline**
   ```
   python main.py --data_path order_books/ --output risks/
   ```

2. **Launch Visualization Dashboard**
   ```
   streamlit run dashboard.py
   ```

3. **Simulate Trade Execution**
   ```
   python simulate.py --qty 5000 --side buy --strategy VWAP
   ```

---

## ⚙️ How It Works

1. **Data Pipeline**
   - Ingests order book snapshots (CSV/real-time feed)
   - Normalizes timestamps to microsecond precision

2. **Risk Detection**
   - Calculates volume dispersion index (VDI)
   - Detects spoofing via depth imbalance patterns
   - Monitors spread volatility clusters

3. **Simulation**
   - Models trade execution paths
   - Quantifies price impact and slippage
   - Generates optimal timing flags

---

## 📊 Key Metrics

| Metric               | Target            |
|----------------------|-------------------|
| Order Books Processed| 20,000+           |
| Slippage Risk Ident. | 30%+ scenarios    |
| Detection Latency    | <50μs per snapshot|
| Alert Precision      | 92% (backtested)  |

---

## 🛠️ Customization

**Adjust Detection Thresholds**
```
# config.yaml
risk_params:
  vdi_threshold: 0.65
  max_spread_volatility: 1.8%
  min_spoofing_duration: 150μs
```

**Modify Simulation Parameters**
```
# simulate.py
SIMULATION_PARAMS = {
    'qty_tiers': [1000][5000][10000],
    'slippage_model': 'square_root',
    'time_horizon': '15min'
}
```

---

## 📚 References

- [Market Microstructure Theory](https://www.wiley.com/en-us/Market+Microstructure+Theory-p-9780631205321)
- [Spoofing Detection Techniques](https://arxiv.org/abs/1708.03580)
- [Liquidity Fragmentation Metrics](https://www.sciencedirect.com/science/article/pii/S0378426620302204)

---

## 📄 License

Apache 2.0 License. See `LICENSE` for details.

---

## 🙏 Acknowledgments

Market data patterns inspired by NASDAQ ITCH feed specifications and FIX protocol documentatis 
