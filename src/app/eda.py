import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.preprocessing.data_processor import DataProcessor
from src.visualization.visualizer import Visualizer

def show_eda_page():
    """
    Display the exploratory data analysis page of the application.
    """
    st.title("Exploratory Data Analysis")
    
    st.markdown("""
    This page contains visualizations and insights from the bank marketing dataset.
    The exploratory data analysis helps understand the patterns and relationships in the data.
    """)
    
    # Load data
    data_path = os.path.join(os.getcwd(), 'src', 'data', 'bank-full.csv')
    # If the file doesn't exist in the data directory, try the root directory
    if not os.path.exists(data_path):
        data_path = os.path.join(os.getcwd(), 'bank-full.csv')
    
    if os.path.exists(data_path):
        # Load and process data
        data_processor = DataProcessor(data_path=data_path)
        df = data_processor.df
        
        # Create visualizer
        visualizer = Visualizer(df)
        
        # Display basic information about the dataset
        st.markdown("## Dataset Overview")
        st.write(f"**Number of rows:** {df.shape[0]}")
        st.write(f"**Number of columns:** {df.shape[1]}")
        
        # Display first few rows of the dataset
        st.markdown("### Sample Data")
        st.dataframe(df.head())
        
        # Display data types and missing values
        st.markdown("### Data Types and Missing Values")
        dtypes_df = pd.DataFrame({
            'Data Type': df.dtypes,
            'Missing Values': df.isnull().sum(),
            'Missing (%)': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(dtypes_df)
        
        # Target variable distribution
        st.markdown("## Target Variable Analysis")
        visualizer.plot_target_distribution()
        
        # Categorical features analysis
        st.markdown("## Categorical Features Analysis")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'y' in categorical_cols:
            categorical_cols.remove('y')
        
        # Allow user to select categorical features to analyze
        selected_cat_cols = st.multiselect(
            "Select categorical features to analyze:",
            options=categorical_cols,
            default=categorical_cols[:3]  # Default to first 3 categorical columns
        )
        
        if selected_cat_cols:
            visualizer.plot_categorical_features(columns=selected_cat_cols)
        
        # Numerical features analysis
        st.markdown("## Numerical Features Analysis")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Allow user to select numerical features to analyze
        selected_num_cols = st.multiselect(
            "Select numerical features to analyze:",
            options=numerical_cols,
            default=numerical_cols[:3]  # Default to first 3 numerical columns
        )
        
        if selected_num_cols:
            visualizer.plot_numerical_features(columns=selected_num_cols)
        
        # Correlation analysis
        st.markdown("## Correlation Analysis")
        visualizer.plot_correlation_matrix()
        
        # Key insights
        st.markdown("## Key Insights")
        st.markdown("""
        Based on the exploratory data analysis, here are some key insights:
        
        1. **Target Distribution**: The dataset is imbalanced, with a majority of clients not subscribing to term deposits.
        
        2. **Age Distribution**: The age distribution shows that middle-aged clients are more likely to subscribe to term deposits.
        
        3. **Job Type**: Certain job types like management and retired have higher subscription rates.
        
        4. **Education Level**: Clients with tertiary education have higher subscription rates.
        
        5. **Contact Type**: Clients contacted via cellular have higher subscription rates than those contacted via telephone.
        
        6. **Previous Outcome**: Clients with a successful previous marketing campaign outcome have significantly higher subscription rates.
        
        7. **Duration**: Longer call durations are associated with higher subscription rates, but this is a leaky variable that should be used carefully in modeling.
        
        8. **Balance**: Clients with higher account balances tend to have higher subscription rates.
        
        These insights can help the bank target the right clients for their marketing campaigns and improve the efficiency of their telemarketing efforts.
        """)
    else:
        st.error(f"Data file not found at {data_path}. Please make sure the file exists.")