import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Import custom modules
from src.preprocessing.data_processor import DataProcessor
from src.visualization.visualizer import Visualizer
from src.models.model_trainer import ModelTrainer
from src.app.home import show_home_page
from src.app.eda import show_eda_page
from src.app.prediction import show_prediction_page
from src.app.model_info import show_model_info_page

# Set page configuration
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the main function
def main():
    # Add a sidebar
    st.sidebar.title("Navigation")
    
    # Create navigation options
    pages = {
        "Home": show_home_page,
        "Exploratory Data Analysis": show_eda_page,
        "Prediction": show_prediction_page,
        "Model Information": show_model_info_page
    }
    
    # Select the page
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Display the selected page
    pages[selection]()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application predicts whether a client will subscribe to a term deposit "
        "based on various features. It was built using Streamlit and scikit-learn."
    )

# Run the main function
if __name__ == "__main__":
    main()