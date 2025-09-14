import streamlit as st

def show_home_page():
    """
    Display the home page of the application.
    """
    st.title("Bank Term Deposit Prediction")
    
    st.markdown("""
    ## Problem Statement
    
    A Portuguese banking institution has conducted multiple direct marketing campaigns via phone calls to promote term deposit products. 
    The bank aims to improve the efficiency of these campaigns by predicting which clients are likely to subscribe to a term deposit.
    
    This application uses machine learning to predict whether a client will subscribe to a term deposit based on various features.
    
    ## Business Use Cases
    
    ### Targeted Marketing Campaigns
    By predicting which customers are most likely to subscribe, the bank can focus its marketing efforts on high-potential leads. 
    This increases the efficiency of campaigns and reduces wasted resources on uninterested customers.
    
    ### Cost Reduction in Telemarketing
    Telemarketing involves significant human resource and time costs. A predictive model helps minimize unnecessary calls by avoiding 
    low-probability clients, leading to reduced operational expenses.
    
    ### Improved Customer Experience
    By avoiding repeated or irrelevant calls to unlikely customers, the bank can enhance the overall customer experience. 
    Clients feel more valued when interactions are relevant and well-timed.
    
    ### Campaign Performance Optimization
    Historical insights from model predictions allow marketing teams to understand what factors influence successful subscriptions. 
    This helps in designing more effective future campaigns with better strategies and timing.
    
    ### Personalized Financial Product Recommendations
    The model can be used as part of a larger recommendation engine that personalizes financial product offerings for clients based on their profile, 
    increasing cross-selling and upselling opportunities.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## How to Use This Application
    
    This application has the following pages:
    
    1. **Home** - Overview of the problem and application
    2. **Exploratory Data Analysis** - Visualizations and insights from the data
    3. **Prediction** - Make predictions using the trained model
    4. **Model Information** - Details about the model performance
    
    Use the sidebar to navigate between pages.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Dataset Information
    
    The dataset contains information about bank clients, previous marketing campaigns, and whether the client subscribed to a term deposit.
    
    ### Features
    
    #### Bank Client Data
    - **age**: Age of the client (numeric)
    - **job**: Type of job (categorical)
    - **marital**: Marital status (categorical)
    - **education**: Education level (categorical)
    - **default**: Has credit in default? (binary)
    - **balance**: Average yearly balance in euros (numeric)
    - **housing**: Has housing loan? (binary)
    - **loan**: Has personal loan? (binary)
    
    #### Last Contact of Current Campaign
    - **contact**: Contact communication type (categorical)
    - **day**: Last contact day of the month (numeric)
    - **month**: Last contact month of year (categorical)
    - **duration**: Last contact duration in seconds (numeric)
    
    #### Other Attributes
    - **campaign**: Number of contacts performed during this campaign for this client (numeric)
    - **pdays**: Number of days since client was last contacted from a previous campaign (numeric)
    - **previous**: Number of contacts performed before this campaign for this client (numeric)
    - **poutcome**: Outcome of the previous marketing campaign (categorical)
    
    ### Target Variable
    - **y**: Has the client subscribed to a term deposit? (binary: "yes", "no")
    """)