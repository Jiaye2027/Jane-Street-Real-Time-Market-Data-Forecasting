# Jane Street Financial Forecasting: Model Exploration  
**Kaggle Competition** | [Competition Link](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)

This repository documents my iterative approach to solving Jane Street’s high-frequency financial forecasting challenge. The project involved experimenting with **LSTMs**, **custom neural networks**, and **fine-tuning public models**, with a focus on addressing challenges like massive datasets, anonymized features, and computational constraints.  

---

## Overview  
Financial market forecasting involves modeling noisy, non-stationary data with complex temporal dependencies. Key challenges included:  
- **Scalability**: Processing 20GB+ datasets efficiently on local hardware.  
- **Anonymized features**: Limited interpretability of input variables.  
- **Computational limits**: Training complex models within practical timeframes.  

In this project, I:  
- Leveraged **Polars** for fast, out-of-memory data processing.
- Explored limitations of recurrent architectures (LSTMs) in high-noise environments. 
- Designed a custom neural network to handle fat-tailed distributions and non-stationarity.  
- Fine-tune public neural networks by adding features and architectures.  

---

## Approach & Challenges  

### Data Processing with Polars  
**Goal**: Efficiently process 20GB+ datasets.  
**Implementation**:  
- Replaced Pandas with **Polars** for out-of-core computation and parallelized feature engineering.  
- Engineered anonymized features using rolling volatility, lagged returns, and statistical aggregation.  

**Key Challenges**:  
- Anonymized feature names limited domain-driven insights, requiring exploratory analysis.  

---

### LSTM for Time-Series Forecasting  
**Goal**: Capture temporal patterns in market data.  
**Implementation**:  
- Built an LSTM model using Keras.  
- Trained on a windowed sequence of historical trades.  

**Key Challenges**:  
- Model struggled to generalize due to high noise and non-stationarity in financial data.  
- Theoretical insight: LSTMs may underperform in environments with low signal-to-noise ratios.

---

### Custom Neural Network with Weighted Loss  
**Goal**: Model non-stationary financial data with fat-tailed distributions.  
**Implementation**:  
- Built a high-capacity feedforward neural network (TensorFlow/Keras) to capture complex market patterns.  
- Integrated **weighted Huber loss** to reduce sensitivity to outliers and sudden market shifts.  
- Engineered features: **sin/cos transforms of `time_id`** to encode cyclical time patterns.  

**Key Challenges**:  
- Training exceeded 20+ hours on local hardware, forcing early termination and reevaluation of scalability.  
- Trade-off: Balancing model complexity with computational feasibility.  

---

### Public Neural Network Fine-Tuning  
**Goal**: Optimize existing architectures for anonymized data.  
**Implementation**:  
- Fine-tuned a public neural network by:  
  - Adding time-series features: sin/cos transforms of `time_id` for cyclical encoding.  
  - Adding product name embedding.
  - Adding time series features like ewa, std, momentum, etc.
  - Adding features interactions layer.
  - Adding residual block.
  - Using weighted huber loss.

**Key Challenges**:  
- Limited performance gains despite feature additions, likely due to noisy anonymized inputs.  
- Final submission prioritized simplicity and inference speed over marginal accuracy improvements.  

---

## Future Work  
- Explore **online training** to keep the model up to date.  
- Consider **feature selection** to reduce noise in anonymized datasets.  
- Experiment with **quantile regression** to better model fat-tailed distributions.  

---

