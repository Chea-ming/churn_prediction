
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load('../model/churn_model.pkl')
scaler = joblib.load('../model/scaler.pkl')

# Streamlit app
st.title('Telco Customer Churn Prediction')
st.write('Enter customer details to predict churn probability')

# Input fields
tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, value=50.0)
contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Prepare input data
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [tenure * monthly_charges],
    'AvgMonthlyCharge': [monthly_charges],
    'Contract_One year': [1 if contract == 'One year' else 0],
    'Contract_Two year': [1 if contract == 'Two year' else 0],
    'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
    'InternetService_No': [1 if internet_service == 'No' else 0],
    'PaymentMethod_Bank transfer (automatic)': [1 if payment_method == 'Bank transfer (automatic)' else 0],
    'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card (automatic)' else 0],
    'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
    'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0]
})
# Add remaining dummy columns to match training data
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0
# Scale numerical features
input_data[['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge']] = scaler.transform(
    input_data[['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge']]
)
# Predict
if st.button('Predict'):
    # Get feature names from model
    model_features = model.get_booster().feature_names

    # Reorder columns of input_data to match training order
    input_data = input_data[model_features]

    # Then predict
    prob = model.predict_proba(input_data)[:, 1][0]
    st.write(f'Churn Probability: {prob:.2%}')
    if prob > 0.5:
        st.warning('High risk of churn! Consider offering retention incentives.')
    else:
        st.success('Low risk of churn.')
    # Display feature importance
    st.subheader('Model Feature Importance')
    feat_importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    fig, ax = plt.subplots(figsize=(8, 4))
    feat_importance.nlargest(10).plot(kind='barh', ax=ax)
    ax.set_title('Top 10 Feature Importance')
    st.pyplot(fig)
