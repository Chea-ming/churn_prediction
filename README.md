```markdown
# Customer Churn Prediction for Telecom

## Overview
This project predicts customer churn for a telecom company using the Telco Customer Churn dataset. By identifying customers at risk of leaving, the model enables targeted retention strategies, potentially saving millions in revenue. The project demonstrates end-to-end data science skills: data cleaning, exploratory data analysis (EDA), feature engineering, machine learning, model evaluation, and deployment via a Streamlit app.

## Business Problem
Customer churn is a critical issue in the telecom industry, costing companies billions annually. This project builds a predictive model to identify high-risk customers based on demographics and service usage, enabling proactive interventions like discounts or personalized offers.

## Dataset
- **Source:** Kaggle Telco Customer Churn dataset (7,043 rows, 21 columns).
- **Features:** Includes tenure, MonthlyCharges, Contract, InternetService, PaymentMethod, etc.
- **Target:** Churn (binary: Yes/No).

## Methodology

### Data Cleaning
- Handled missing values in TotalCharges (filled with median).
- Dropped irrelevant customerID and duplicates.
- Converted data types for consistency.

### Exploratory Data Analysis (EDA)
- Visualized churn distribution, correlations, and key patterns (e.g., higher churn in month-to-month contracts).
- Generated plots: churn distribution, correlation heatmap, churn by contract type.

### Feature Engineering
- Created AvgMonthlyCharge (TotalCharges / tenure).
- Applied one-hot encoding to categorical features and scaled numerical features with `StandardScaler`.

### Modeling
- Trained three models: Logistic Regression (baseline), Random Forest, and XGBoost.
- Used `GridSearchCV` for hyperparameter tuning on XGBoost.
- Evaluated with accuracy, F1-score, and ROC-AUC due to class imbalance.

### Deployment
- Built an interactive Streamlit app to predict churn probability for new customers.
- Displays feature importance for model interpretability.
- Saved model and scaler using `joblib`.

## Results
- **Logistic Regression:** 80% accuracy, 0.845 ROC-AUC.
- **Random Forest:** 79% accuracy, 0.826 ROC-AUC.
- **XGBoost:** 80% accuracy, 0.836 ROC-AUC (best parameters: learning_rate=0.1, max_depth=5, n_estimators=100).
- **Key Insight:** Tenure and contract type are top predictors of churn. Month-to-month customers are 3x more likely to churn than those on longer contracts.

## Business Impact
The model identifies high-risk customers with 80% accuracy and 61% precision for churners. By targeting these customers with retention strategies (e.g., discounts, loyalty programs), the company could reduce churn by ~20%, potentially saving significant revenue based on average customer lifetime value.

## Repository Structure
```
/churn_prediction_project
├── /data
│   └── telco_customer_churn.csv
├── /notebook
│   └── churn_prediction.ipynb
├── /models
│   ├── churn_model.pkl
│   └── scaler.pkl
├── /figures
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── churn_by_contract.png
│   └── feature_importance.png
├── /app
│   └── streamlit_app.py
└── README.md
```

## Setup Instructions
1. **Clone Repository:** 
   ```bash
   git clone <your-repo-url>
   cd churn_prediction_project
   ```

2. **Install Dependencies:** 
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost streamlit joblib
   ```

3. **Download Dataset:**
   Place `telco_customer_churn.csv` in the `/data` folder.

4. **Run Notebook:** 
   Open `notebook/churn_prediction.ipynb` in Jupyter Notebook and execute cells.

5. **Run Streamlit App:** 
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Future Improvements
- Add SHAP explanations for deeper model interpretability.
- Implement SMOTE to address class imbalance and improve F1-score.
- Conduct cost-benefit analysis to quantify retention strategy savings.
# churn_prediction
