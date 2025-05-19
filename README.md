# Anomaly Detection in Financial Transactions

This project identifies anomalies in financial transactions using AI-driven techniques including Isolation Forest and Autoencoders.

## Features
- Preprocesses transaction data
- Trains and uses Isolation Forest
- Trains and uses Autoencoder
- Detects potential fraudulent or erroneous transactions

## Setup
```bash
pip install -r requirements.txt
```

## Run Steps
1. Preprocess data:
```bash
python scripts/preprocess_data.py
```

2. Train models:
```bash
python scripts/train_isolation_forest.py
python scripts/train_autoencoder.py
```

3. Detect anomalies:
```bash
python scripts/detect_anomalies.py
```
