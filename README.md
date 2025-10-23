# Customer Churn Prediction for Telecom

## Overview
This project predicts customer churn for a telecom company using the Telco Customer Churn dataset. By identifying customers at risk of leaving, the model enables targeted retention strategies, potentially saving millions in revenue. The project demonstrates end-to-end data science skills: data cleaning, exploratory data analysis (EDA), feature engineering, machine learning, model evaluation, and deployment via a Streamlit app.

## Business Problem
Customer churn is a critical issue in the telecom industry, costing companies billions annually. This project builds a predictive model to identify high-risk customers based on demographics and service usage, enabling proactive interventions like discounts or personalized offers.

## Dataset
- **Source:** Kaggle Telco Customer Churn dataset
- **Size:** 7,043 rows, 21 columns
- **Features:** tenure, MonthlyCharges, Contract, InternetService, PaymentMethod, etc.
- **Target:** Churn (binary: Yes/No)

## Project Structure
```
churn_prediction_project/
├── data/
│   └── telco_customer_churn.csv
├── notebook/
│   └── churn_prediction.ipynb
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
├── figures/
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── churn_by_contract.png
│   └── feature_importance.png
├── app/
│   └── streamlit_app.py
└── README.md
```

## Methodology

### Data Cleaning
- Handled missing values in TotalCharges (filled with median)
- Dropped irrelevant customerID and duplicates
- Converted data types for consistency

### Exploratory Data Analysis (EDA)
- Visualized churn distribution, correlations, and key patterns
- Identified higher churn in month-to-month contracts
- Generated various plots including churn distribution and correlation heatmap

### Feature Engineering
- Created AvgMonthlyCharge (TotalCharges / tenure)
- Applied one-hot encoding to categorical features
- Scaled numerical features with StandardScaler

### Modeling
Trained and evaluated three machine learning models:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Used GridSearchCV for hyperparameter tuning on XGBoost and evaluated models using accuracy, F1-score, and ROC-AUC metrics.

### Deployment
Built an interactive Streamlit app that:
- Predicts churn probability for new customers
- Displays feature importance for model interpretability
- Uses joblib for model persistence

## Results

### Model Performance
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 80% | 0.845 |
| Random Forest | 79% | 0.826 |
| XGBoost | 80% | 0.836 |

**Best XGBoost Parameters:**
- learning_rate: 0.1
- max_depth: 5
- n_estimators: 100

### Key Insights
- Tenure and contract type are top predictors of churn
- Month-to-month customers are 3x more likely to churn than those on longer contracts
- Model achieves 80% accuracy and 61% precision for churn prediction

## Business Impact
The model identifies high-risk customers with 80% accuracy, enabling targeted retention strategies. By implementing proactive interventions (discounts, loyalty programs), the company could:
- Reduce churn by approximately 20%
- Save significant revenue based on average customer lifetime value
- Improve customer retention and satisfaction

## Installation & Setup

### Prerequisites
- Python 3.7+
- Required packages listed in requirements.txt

### Installation Steps
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd churn_prediction_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset**
   - Place `telco_customer_churn.csv` in the `/data` folder

### Usage

**Run Jupyter Notebook:**
```bash
jupyter notebook notebook/churn_prediction.ipynb
```

**Run Streamlit App:**
```bash
streamlit run app/streamlit_app.py
```

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- streamlit
- joblib

## Future Improvements
- Add SHAP explanations for deeper model interpretability
- Implement SMOTE to address class imbalance and improve F1-score
- Conduct cost-benefit analysis to quantify retention strategy savings
- Explore deep learning approaches for improved performance
- Add real-time data integration capabilities
