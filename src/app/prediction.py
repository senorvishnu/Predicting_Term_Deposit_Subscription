import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.preprocessing.data_processor import DataProcessor

def show_prediction_page():
    """
    Display the prediction page of the application.
    """
    st.title("Term Deposit Subscription Prediction")
    
    st.markdown("""
    Use this page to predict whether a client will subscribe to a term deposit based on their information.
    Fill in the form below and click the 'Predict' button to get a prediction.
    """)
    
    # Check if model exists
    model_path = os.path.join(os.getcwd(), 'src', 'models', 'best_model.pkl')
    preprocessor_path = os.path.join(os.getcwd(), 'src', 'models', 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        st.warning("Model or preprocessor not found. Please train the model first.")
        
        # Provide option to train model
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Import necessary modules
                from src.models.model_trainer import ModelTrainer
                
                # Load and preprocess data
                data_path = os.path.join(os.getcwd(), 'src', 'data', 'bank-full.csv')
                # If the file doesn't exist in the data directory, try the root directory
                if not os.path.exists(data_path):
                    data_path = os.path.join(os.getcwd(), 'bank-full.csv')
                data_processor = DataProcessor(data_path=data_path)
                X_train, X_test, y_train, y_test = data_processor.preprocess_data()
                
                # Train models
                model_trainer = ModelTrainer()
                model_trainer.train_logistic_regression(X_train, y_train)
                model_trainer.train_decision_tree(X_train, y_train)
                model_trainer.train_random_forest(X_train, y_train)
                model_trainer.train_xgboost(X_train, y_train)
                
                # Evaluate models
                results = model_trainer.evaluate_all_models(X_test, y_test)
                
                # Save best model and preprocessor
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model_trainer.save_model(file_path=model_path)
                data_processor.save_preprocessor(preprocessor_path)
                
                st.success("Model trained and saved successfully!")
                st.rerun()
        
        return
    
    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Create form for user input
    with st.form("prediction_form"):
        st.markdown("### Client Information")
        
        # Bank client data
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            
            job_options = [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician",
                "unemployed", "unknown"
            ]
            job = st.selectbox("Job", options=job_options, index=4)  # Default to management
            
            marital_options = ["divorced", "married", "single"]
            marital = st.selectbox("Marital Status", options=marital_options, index=1)  # Default to married
            
            education_options = ["primary", "secondary", "tertiary", "unknown"]
            education = st.selectbox("Education", options=education_options, index=2)  # Default to tertiary
        
        with col2:
            default_options = ["no", "yes"]
            default = st.selectbox("Has Credit in Default?", options=default_options, index=0)  # Default to no
            
            balance = st.number_input("Average Yearly Balance (euros)", min_value=-10000, max_value=100000, value=1000)
            
            housing_options = ["no", "yes"]
            housing = st.selectbox("Has Housing Loan?", options=housing_options, index=1)  # Default to yes
            
            loan_options = ["no", "yes"]
            loan = st.selectbox("Has Personal Loan?", options=loan_options, index=0)  # Default to no
        
        st.markdown("### Last Contact Information")
        
        col3, col4 = st.columns(2)
        
        with col3:
            contact_options = ["cellular", "telephone", "unknown"]
            contact = st.selectbox("Contact Communication Type", options=contact_options, index=0)  # Default to cellular
            
            day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15)
        
        with col4:
            month_options = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            month = st.selectbox("Last Contact Month", options=month_options, index=4)  # Default to may
            
            duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=300)
        
        st.markdown("### Other Information")
        
        col5, col6 = st.columns(2)
        
        with col5:
            campaign = st.number_input("Number of Contacts in Current Campaign", min_value=1, max_value=100, value=2)
            
            pdays = st.number_input("Days Since Last Contact from Previous Campaign (-1 if not contacted)", min_value=-1, max_value=1000, value=-1)
        
        with col6:
            previous = st.number_input("Number of Contacts Before Current Campaign", min_value=0, max_value=100, value=0)
            
            poutcome_options = ["failure", "other", "success", "unknown"]
            poutcome = st.selectbox("Outcome of Previous Campaign", options=poutcome_options, index=3)  # Default to unknown
        
        # Submit button
        submitted = st.form_submit_button("Predict")
    
    # Make prediction when form is submitted
    if submitted:
        # Create input data
        input_data = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Create data processor and preprocess input
        data_processor = DataProcessor()
        data_processor.preprocessor = preprocessor
        processed_input = data_processor.preprocess_input(input_df)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_input)[0, 1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Display prediction
        st.markdown("### Prediction Result")
        
        # Create columns for prediction display
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success("The client is likely to subscribe to a term deposit.")
            else:
                st.error("The client is unlikely to subscribe to a term deposit.")
        
        with col2:
            st.metric("Subscription Probability", f"{prediction_proba:.2%}")
        
        # Display gauge chart for probability
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Subscription Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        st.plotly_chart(fig)
        
        # Display feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.markdown("### Feature Importance")
            
            # Get feature names from preprocessor
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_:
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(columns))
                else:
                    feature_names.extend(columns)
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': feature_names[:len(model.feature_importances_)],
                'Importance': model.feature_importances_
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Display top 10 features
            st.bar_chart(feature_importance.set_index('Feature').head(10))
        
        # Display interpretation
        st.markdown("### Interpretation")
        
        if prediction == 1:
            st.markdown("""
            The model predicts that this client is likely to subscribe to a term deposit. 
            Key factors that may have influenced this prediction include:
            
            - **Duration**: Longer call durations are associated with higher subscription rates.
            - **Previous Outcome**: Clients with successful previous campaigns are more likely to subscribe.
            - **Age and Education**: Certain age groups and education levels have higher subscription rates.
            - **Balance**: Clients with higher account balances tend to have higher subscription rates.
            
            Consider prioritizing this client for your marketing campaign.
            """)
        else:
            st.markdown("""
            The model predicts that this client is unlikely to subscribe to a term deposit. 
            Factors that may have influenced this prediction include:
            
            - **Duration**: Shorter call durations are associated with lower subscription rates.
            - **Previous Outcome**: Clients with unsuccessful or no previous campaigns are less likely to subscribe.
            - **Age and Education**: Certain age groups and education levels have lower subscription rates.
            - **Balance**: Clients with lower account balances tend to have lower subscription rates.
            
            Consider allocating your marketing resources to other clients with higher subscription probabilities.
            """)