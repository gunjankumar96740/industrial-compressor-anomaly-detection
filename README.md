# Industrial Compressor Condition Monitoring & Anomaly Detection

An end-to-end **unsupervised machine learning system** for detecting abnormal operating conditions in industrial compressor sensor data.

The project processes **1.5M+ sensor observations across 15 sensor features**, learns patterns representing normal compressor behavior, and identifies potentially abnormal operating conditions using **Isolation Forest** and **PCA-based Local Outlier Factor (LOF)**.

The trained Isolation Forest model is deployed through an interactive **Streamlit application** for real-time anomaly prediction and anomaly-score visualization.

---

## 🚀 Project Overview

Industrial equipment continuously generates large volumes of sensor data. Detecting unusual operating patterns early can help identify potential equipment problems and support condition-based monitoring.

This project builds an unsupervised anomaly detection pipeline that:

- Processes large-scale industrial sensor data
- Cleans and standardizes sensor features
- Reduces dimensionality using **Principal Component Analysis (PCA)**
- Detects anomalies using **Isolation Forest**
- Compares results with **Local Outlier Factor (LOF)**
- Provides anomaly scores for individual observations
- Deploys the trained model through **Streamlit**

---

## 🎯 Objectives

- Identify abnormal compressor operating patterns without requiring labeled failure data.
- Build a scalable anomaly detection pipeline for high-volume sensor data.
- Compare multiple unsupervised anomaly detection techniques.
- Reduce feature dimensionality while retaining **95% of the variance**.
- Provide an interactive interface for sensor-level anomaly detection.

---

## 📊 Dataset

The dataset contains:

- **1.5M+ sensor observations**
- **15 sensor features**
- Industrial compressor operating measurements

The data is used to learn the normal operating behavior of the compressor and identify observations that deviate significantly from learned patterns.

> **Note:** The project focuses on unsupervised anomaly detection rather than supervised failure classification.

---

## 🔄 Machine Learning Pipeline

```text
Raw Sensor Data
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
Feature Standardization
       ↓
       PCA
       ↓
95% Variance Retention
       ↓
 ┌───────────────┐
 │               │
 ▼               ▼
Isolation      Local Outlier
Forest (IF)    Factor (LOF)
 │               │
 └───────┬───────┘
         ↓
Anomaly Detection
         ↓
Model Evaluation & Comparison
         ↓
Streamlit Deployment
```

---

## 🧠 Algorithms Used

### 1. Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm that isolates unusual observations by randomly partitioning the feature space.

It is particularly useful for this project because it can efficiently handle large datasets and identify observations that are significantly different from normal operating patterns.

### 2. PCA — Principal Component Analysis

PCA is used for dimensionality reduction.

The project retains **95% of the variance** while reducing the dimensional representation of the sensor data.

Benefits:

- Reduces dimensionality
- Removes redundant information
- Makes anomaly detection more scalable
- Helps identify the dominant variation in sensor measurements

### 3. Local Outlier Factor

PCA-based **Local Outlier Factor (LOF)** is implemented as a comparison model.

LOF identifies observations whose local density differs significantly from that of their neighboring observations.

---

## 🛠️ Tech Stack

### Programming
- Python

### Data Processing
- Pandas
- NumPy
- Scikit-learn

### Machine Learning
- Isolation Forest
- Local Outlier Factor
- PCA
- Feature Standardization

### Visualization
- Matplotlib
- Seaborn

### Deployment
- Streamlit

### Development
- Jupyter Notebook
- Git
- GitHub

---

## 📁 Project Structure

```text
industrial-compressor-anomaly-detection/
│
├── data/
│   └── README.md
│
├── models/
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── anomaly_detection.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

> The exact files may vary depending on the final repository structure.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/gunjankumar96740/industrial-compressor-anomaly-detection.git
```

Move into the project directory:

```bash
cd industrial-compressor-anomaly-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application


The application provides:

- Sensor-level input
- Anomaly classification
- Anomaly score
- Interactive visualization

---

## 📈 Key Results

- Processed **1.5M+ industrial sensor observations**
- Worked with **15 sensor features**
- Applied feature standardization before modeling
- Used PCA while retaining **95% variance**
- Implemented **Isolation Forest** for anomaly detection
- Compared Isolation Forest with **PCA-based LOF**
- Deployed the trained Isolation Forest model using **Streamlit**

---

## 💡 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

- Unsupervised Machine Learning
- Anomaly Detection
- Feature Scaling
- Dimensionality Reduction
- PCA
- Isolation Forest
- Local Outlier Factor
- Large-scale Data Processing
- Model Deployment
- Streamlit
- ML Pipeline Development

---

## 🔮 Future Improvements

Potential extensions include:

- Adding real-time sensor streaming
- Monitoring anomaly trends over time
- Adding automated alerts for persistent anomalies
- Experimenting with additional anomaly detection algorithms
- Building a monitoring dashboard for multiple machines
- Integrating the model with an industrial IoT pipeline

---

## 👨‍💻 Author

**Gunjan Kumar**

B.Tech — Smart Manufacturing  
PDPM IIITDM Jabalpur

### Connect

- LinkedIn: [Gunjan Kumar](https://www.linkedin.com/in/gunjan-kumar-2bbb1b286/)
- GitHub: [gunjankumar96740](https://github.com/gunjankumar96740)

---

## ⭐ Project Highlights

**1.5M+ sensor observations | 15 features | PCA | Isolation Forest | LOF | Streamlit**

An end-to-end machine learning project demonstrating how unsupervised learning can be applied to industrial sensor data for condition monitoring and anomaly detection.
