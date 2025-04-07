**Anomaly Detection techniques** 

---

### 📊 Anomaly Detection Techniques Overview

| **Method** | **What it is** | **How it works** | **Best suited datasets** | **Type** | **Advantages** | **Disadvantages** | **Use Cases** |
|------------|----------------|------------------|---------------------------|----------|----------------|-------------------|----------------|
| **Z-Score / Statistical Methods** | Based on statistical deviation from mean | Calculates how far a data point is from mean in standard deviations | Normally distributed numeric data | Unsupervised | Easy to implement, interpretable | Assumes normal distribution | Fraud detection, sensor data |
| **IQR (Interquartile Range)** | Uses median and quartiles to detect outliers | Flags points that fall below Q1 - 1.5×IQR or above Q3 + 1.5×IQR | Small-medium datasets, non-parametric | Unsupervised | Robust to outliers, no normality assumption | Not suitable for high-dimensional data | Quality control, financial data |
| **Isolation Forest** | Tree-based model for anomaly detection | Randomly isolates observations; anomalies are isolated faster | High-dimensional, large datasets | Unsupervised | Fast, handles high-dimensional data | Not good with categorical features | Network intrusion detection, manufacturing anomalies |
| **One-Class SVM** | SVM variant that learns boundary of normal class | Fits a hypersphere around normal data and detects points outside it | Small to medium datasets with one class | Unsupervised / Semi-supervised | Effective with clear margins | Sensitive to parameter tuning | Industrial equipment monitoring |
| **Autoencoder** | Neural network that learns compressed representation | Reconstructs input and measures reconstruction error | Image, time-series, high-dimensional data | Unsupervised | Captures complex patterns | Requires a lot of training data | Credit card fraud, image anomaly |
| **PCA (Principal Component Analysis)** | Dimensionality reduction technique | Projects data into lower dimensions and uses reconstruction error | High-dimensional numeric data | Unsupervised | Simple, fast | Assumes linear relationships | Feature reduction, anomaly in system logs |
| **DBSCAN** | Clustering method detecting dense regions | Points in low-density regions are anomalies | Spatial or density-based datasets | Unsupervised | Detects arbitrary shape clusters | Hard to set parameters like ε | Geospatial anomaly, user behavior |
| **LOF (Local Outlier Factor)** | Density-based anomaly detection | Compares local density of a point with its neighbors | Spatial, numerical, small-medium data | Unsupervised | Good with local density variations | Sensitive to parameter setting | Network security, transaction outliers |
| **KNN-based** | Measures distance to nearest neighbors | Large distance indicates anomaly | Numerical or metric space data | Unsupervised | Easy to understand | Computationally expensive | Health monitoring, rare event detection |
| **Time-Series Forecasting (ARIMA, LSTM)** | Model learns pattern over time | Predicts next value and compares with actual | Sequential data | Supervised / Unsupervised | Captures temporal dynamics | Needs well-prepared time series | Energy usage, stock trends |
| **RNN / LSTM Autoencoders** | Sequence-aware neural autoencoders | Learns sequential patterns and reconstructs data | Time-series data | Unsupervised | Handles sequence dependence | Training complexity | IoT anomaly, power grid monitoring |
| **Gaussian Mixture Models (GMM)** | Probabilistic model of mixture distributions | Calculates likelihood of each point | Numeric data following mixtures | Unsupervised | Probabilistic interpretation | Assumes Gaussian | Financial fraud, customer behavior |
| **Rule-based Systems** | Hand-coded thresholds or logic | Flag deviations from expected rule | Structured business data | Supervised / Manual | Human interpretable | Not adaptable | Compliance checks, monitoring systems |
| **RAGAS (for LLMs)** | RAG evaluation metrics for LLM outputs | Measures factuality, relevance of generated answers | Text-based response evaluation | Supervised | Focused on LLM validation | Not general-purpose | LLM QA validation, hallucination detection |

