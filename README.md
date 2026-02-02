# -Customer-Segmentation-Platform-Machine-Learning
Enterprise-grade customer segmentation system using Python, Scikit-learn, and Streamlit, built with an end-to-end unsupervised machine learning pipeline.

# Customer Segmentation Platform – Machine Learning

An enterprise-grade **Customer Segmentation Platform** built using **Python, Scikit-learn, and Streamlit**, leveraging **unsupervised machine learning** to categorize customers based on behavioral and demographic attributes.

This project simulates a real-world analytics solution commonly deployed in **banking, financial services, and retail enterprises** to support data-driven decision-making.

---

## Executive Summary

Customer segmentation is a foundational capability for organizations seeking to enhance customer engagement, optimize marketing strategies, and improve operational efficiency.

This solution applies **K-Means clustering** on normalized customer data to identify distinct behavioral segments. The model is productionized and exposed through an interactive web interface, demonstrating the complete lifecycle of a machine learning solution—from data analysis to deployment.

---

## Business Objectives

- Identify distinct customer behavioral groups
- Enable targeted marketing and personalization strategies
- Support strategic decision-making through data-driven insights
- Demonstrate scalable ML model deployment architecture

---

## Solution Architecture

Data Source (CSV)
↓
Data Analysis & Feature Engineering (Jupyter)
↓
Feature Scaling (StandardScaler)
↓
Unsupervised Learning (K-Means)
↓
Model Serialization (Joblib)
↓
Interactive Web Application (Streamlit)


---

## Key Technologies

| Layer | Technology |
|-----|-----------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Feature Scaling | StandardScaler |
| Model Storage | Joblib |
| Application Layer | Streamlit |
| Version Control | Git, GitHub |

---

## Feature Set

The model segments customers using the following standardized features:

| Feature | Description |
|------|------------|
| Age | Customer age |
| Income | Annual income |
| TotalSpending | Aggregate purchase value |
| NumWebPurchases | Online transaction count |
| NumStorePurchases | In-store transaction count |
| NumWebVisitsMonth | Monthly digital engagement |
| Recency | Days since most recent transaction |

---

## Model Development Details

- **Algorithm:** K-Means Clustering
- **Learning Type:** Unsupervised
- **Scaling Technique:** Z-score normalization using StandardScaler
- **Cluster Selection:** Elbow Method
- **Model Persistence:** Joblib serialization
- **Inference Mode:** Batch & Single-record prediction

The model assigns customers to the nearest cluster centroid based on normalized feature distances.

---

## Application Capabilities

- Intuitive, enterprise-style user interface
- Real-time customer segment prediction
- Input validation and feature consistency checks
- Production-ready inference pipeline

---

## Repository Structure

customer-segmentation-ml/

├── customer_segmentation.py # Streamlit application (inference layer)

├── Analysis_Model.ipynb # EDA, feature engineering, model training

├── customer_segmentation.csv # Source dataset

├── scaler.pkl # Serialized feature scaler

├── kmeans_model.pkl # Trained clustering model

├── requirements.txt # Dependency management

└── README.md # Technical documentation



---

## Output
<img width="1843" height="806" alt="output" src="https://github.com/user-attachments/assets/f67728c1-2173-4b47-8ea7-3627597c8e7e" />

## Setup & Execution

### Environment Setup

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
Dependency Installation
pip install -r requirements.txt
Application Execution
streamlit run customer_segmentation.py
The application will be accessible at:

http://localhost:8501
Quality & Engineering Considerations
Reproducible ML pipeline

Separation of training and inference

Scalable architecture for future data sources

Clear feature alignment between training and prediction

Maintainable and extensible codebase

Enterprise Use Cases
Banking customer segmentation

Credit card user behavior analysis

Personalized financial product recommendations

Customer lifecycle management

Risk and engagement profiling

Limitations & Assumptions
Dataset is static and CSV-based

Clusters are unlabeled and require domain interpretation

Model retraining is manual

## Future Enhancements
Cluster labeling using business rules

Model monitoring and drift detection

Database-backed data ingestion

REST API exposure

Cloud-native deployment (AWS / Azure)

Advanced visualization dashboards

## Author
Vignesh Raj

Machine Learning & Data Analytics Enthusiast

Focused on building production-ready, business-aligned ML solutions.
