import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from src.models.model_trainer import ModelTrainer

def show_model_info_page():
    """
    Display the model information page of the application.
    """
    st.title("Model Information")
    
    st.markdown("""
    This page provides information about the machine learning model used for predicting term deposit subscriptions.
    It includes model performance metrics, feature importance, and other relevant information.
    """)
    
    # Check if model exists
    model_path = os.path.join(os.getcwd(), 'src', 'models', 'best_model.pkl')
    results_path = os.path.join(os.getcwd(), 'src', 'models', 'model_results.pkl')
    
    if not os.path.exists(model_path):
        st.warning("Model not found. Please train the model first.")
        return
    
    # Load model
    model = joblib.load(model_path)
    
    # Display model type
    st.markdown("## Model Type")
    st.write(f"**Model Type:** {type(model).__name__}")
    
    # Display model parameters
    st.markdown("## Model Parameters")
    params = model.get_params()
    params_df = pd.DataFrame({
        'Parameter': list(params.keys()),
        'Value': list(params.values())
    })
    st.dataframe(params_df)
    
    # Display model performance metrics if results exist
    if os.path.exists(results_path):
        results = joblib.load(results_path)
        
        st.markdown("## Model Performance Metrics")
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Value': [
                results['accuracy'],
                results['precision'],
                results['recall'],
                results['f1_score'],
                results['roc_auc']
            ]
        })
        
        # Display metrics
        st.dataframe(metrics_df.set_index('Metric'))
        
        # Display confusion matrix
        st.markdown("### Confusion Matrix")
        conf_matrix = results['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        
        # Display classification report
        st.markdown("### Classification Report")
        class_report = results['classification_report']
        
        # Convert classification report to DataFrame
        report_df = pd.DataFrame({
            'Class': list(class_report.keys()),
            'Precision': [class_report[k]['precision'] if isinstance(class_report[k], dict) else None for k in class_report.keys()],
            'Recall': [class_report[k]['recall'] if isinstance(class_report[k], dict) else None for k in class_report.keys()],
            'F1-Score': [class_report[k]['f1-score'] if isinstance(class_report[k], dict) else None for k in class_report.keys()],
            'Support': [class_report[k]['support'] if isinstance(class_report[k], dict) else None for k in class_report.keys()]
        })
        
        # Filter out None values
        report_df = report_df.dropna()
        
        # Display classification report
        st.dataframe(report_df.set_index('Class'))
    else:
        # If results don't exist, display a message
        st.info("Model performance metrics not available. Please evaluate the model first.")
        
        # Provide option to evaluate model
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                # Import necessary modules
                from src.preprocessing.data_processor import DataProcessor
                
                # Load and preprocess data
                data_path = os.path.join(os.getcwd(), 'src', 'data', 'bank-full.csv')
                # If the file doesn't exist in the data directory, try the root directory
                if not os.path.exists(data_path):
                    data_path = os.path.join(os.getcwd(), 'bank-full.csv')
                data_processor = DataProcessor(data_path=data_path)
                X_train, X_test, y_train, y_test = data_processor.preprocess_data()
                
                # Create model trainer
                model_trainer = ModelTrainer()
                
                # Evaluate model
                metrics = model_trainer.evaluate_model(model, X_test, y_test)
                
                # Save results
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                joblib.dump(metrics, results_path)
                
                st.success("Model evaluated successfully!")
                st.rerun()
    
    # Display feature importance if available
    st.markdown("## Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        # Get feature names from preprocessor
        preprocessor_path = os.path.join(os.getcwd(), 'src', 'models', 'preprocessor.pkl')
        
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            
            # Get feature names
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
            
            # Display feature importance
            st.bar_chart(feature_importance.set_index('Feature'))
            
            # Display top 10 features as a table
            st.markdown("### Top 10 Most Important Features")
            st.dataframe(feature_importance.head(10).reset_index(drop=True))
        else:
            st.warning("Preprocessor not found. Feature names cannot be displayed.")
    elif hasattr(model, 'coef_'):
        # For linear models
        preprocessor_path = os.path.join(os.getcwd(), 'src', 'models', 'preprocessor.pkl')
        
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            
            # Get feature names
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_:
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(columns))
                else:
                    feature_names.extend(columns)
            
            # Create feature importance DataFrame
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            feature_importance = pd.DataFrame({
                'Feature': feature_names[:len(coef)],
                'Coefficient': coef
            })
            
            # Sort by absolute coefficient value
            feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
            feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
            
            # Display feature importance
            st.bar_chart(feature_importance.set_index('Feature')['Coefficient'])
            
            # Display top 10 features as a table
            st.markdown("### Top 10 Most Influential Features")
            st.dataframe(feature_importance.drop('Abs_Coefficient', axis=1).head(10).reset_index(drop=True))
        else:
            st.warning("Preprocessor not found. Feature names cannot be displayed.")
    else:
        st.info("Feature importance not available for this model type.")
    
    # Model interpretation
    st.markdown("## Model Interpretation")
    
    st.markdown("""
    ### Key Factors Influencing Subscription
    
    Based on the model and feature importance, the following factors are most influential in predicting whether a client will subscribe to a term deposit:
    
    1. **Duration of Last Contact**: Longer call durations are strongly associated with higher subscription rates. However, this is a leaky variable as the duration is only known after the call is made.
    
    2. **Previous Campaign Outcome**: Clients with a successful previous marketing campaign outcome have significantly higher subscription rates.
    
    3. **Month of Contact**: Certain months (particularly March, September, October, and December) show higher subscription rates.
    
    4. **Age**: Middle-aged and older clients tend to have higher subscription rates.
    
    5. **Balance**: Clients with higher account balances are more likely to subscribe.
    
    6. **Education**: Clients with tertiary education have higher subscription rates.
    
    7. **Job Type**: Certain job types like management, retired, and student have higher subscription rates.
    
    8. **Contact Type**: Clients contacted via cellular have higher subscription rates than those contacted via telephone.
    
    ### Business Recommendations
    
    Based on the model insights, here are some recommendations for the bank's marketing campaigns:
    
    1. **Target Segmentation**: Focus on clients with higher predicted subscription probabilities, particularly those with successful previous campaign outcomes, higher balances, and specific job types.
    
    2. **Timing Optimization**: Schedule campaigns during months with historically higher subscription rates (March, September, October, and December).
    
    3. **Contact Method**: Prefer cellular contact over telephone when possible.
    
    4. **Call Quality**: While call duration is a strong predictor, it's a result rather than a cause. Focus on the quality of the conversation rather than just the length.
    
    5. **Personalization**: Tailor the marketing approach based on client demographics and financial status.
    
    6. **Continuous Improvement**: Regularly update the model with new campaign data to improve its predictive power.
    """)