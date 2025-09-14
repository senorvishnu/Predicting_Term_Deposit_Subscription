# Bank Marketing Term Deposit Prediction

## Project Overview

This project aims to predict whether a client will subscribe to a term deposit based on various client attributes and campaign information. The prediction model is built using machine learning techniques and is deployed as a Streamlit web application.

## Business Problem

A Portuguese banking institution is running direct marketing campaigns (phone calls) to promote term deposits to their clients. The bank wants to identify which clients are more likely to subscribe to a term deposit, allowing them to focus their marketing efforts more effectively and increase their success rate.

## Dataset

The dataset contains information about client demographics, previous contacts, and other attributes related to the marketing campaign. The target variable is whether the client subscribed to a term deposit ('yes' or 'no').

Key features include:
- Client demographics (age, job, marital status, education, etc.)
- Financial information (balance, housing loan, personal loan)
- Contact information (contact type, day, month)
- Campaign information (duration, number of contacts, previous campaign outcome)

## Project Structure

```
├── app.py                  # Main Streamlit application entry point
├── bank-full.csv           # Dataset file
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── src/                    # Source code directory
    ├── app/                # Streamlit application pages
    │   ├── eda.py          # Exploratory Data Analysis page
    │   ├── home.py         # Home page
    │   ├── model_info.py   # Model information page
    │   └── prediction.py   # Prediction page
    ├── models/             # Machine learning models
    │   └── model_trainer.py # Model training and evaluation
    ├── preprocessing/      # Data preprocessing
    │   └── data_processor.py # Data loading and preprocessing
    └── visualization/      # Data visualization
        └── visualizer.py   # Visualization utilities
```

## Features

- **Data Preprocessing**: Handles categorical and numerical features, missing values, and feature scaling.
- **Exploratory Data Analysis**: Visualizes data distributions, correlations, and key insights.
- **Machine Learning Models**: Implements and evaluates multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM).
- **Model Evaluation**: Provides comprehensive metrics (accuracy, precision, recall, F1-score, ROC-AUC) and visualizations (ROC curve, confusion matrix).
- **Prediction Interface**: Allows users to input client information and get predictions with probability scores.
- **Model Insights**: Displays feature importance and provides business recommendations.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## Usage

The application consists of four main pages:

1. **Home**: Provides an overview of the project, business problem, and application usage.
2. **EDA**: Displays exploratory data analysis with visualizations and insights.
3. **Model Information**: Shows model performance metrics, feature importance, and interpretation.
4. **Prediction**: Allows users to input client information and get predictions.

## Model Performance

The application evaluates multiple machine learning models and selects the best-performing one based on ROC-AUC score. The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Business Insights

Based on the model and feature importance, the following factors are most influential in predicting whether a client will subscribe to a term deposit:

1. **Duration of Last Contact**: Longer call durations are strongly associated with higher subscription rates.
2. **Previous Campaign Outcome**: Clients with a successful previous marketing campaign outcome have significantly higher subscription rates.
3. **Month of Contact**: Certain months show higher subscription rates.
4. **Age**: Middle-aged and older clients tend to have higher subscription rates.
5. **Balance**: Clients with higher account balances are more likely to subscribe.

## Future Improvements

- Implement more advanced feature engineering techniques
- Add more sophisticated models (e.g., neural networks)
- Enhance the user interface with more interactive visualizations
- Implement A/B testing for different marketing strategies
- Add batch prediction functionality for multiple clients

## Deployment on AWS EC2

### Prerequisites

- AWS account with EC2 access
- Basic knowledge of AWS EC2 and SSH

### Steps to Deploy

1. **Launch an EC2 Instance**:
   - Log in to the AWS Management Console
   - Navigate to EC2 Dashboard
   - Launch a new instance (recommended: t2.micro for testing, t2.small or larger for production)
   - Select Ubuntu Server 20.04 LTS as the AMI
   - Configure security group to allow inbound traffic on ports 22 (SSH) and 8501 (Streamlit)
   - Create or select an existing key pair for SSH access
   - Launch the instance

2. **Connect to Your EC2 Instance**:
   ```
   ssh -i /path/to/your-key.pem ubuntu@your-instance-public-dns
   ```

3. **Deploy the Application**:
   - Upload the deployment script to your instance:
     ```
     scp -i /path/to/your-key.pem deploy_aws.sh ubuntu@your-instance-public-dns:~
     ```
   - Connect to your instance and make the script executable:
     ```
     chmod +x deploy_aws.sh
     ```
   - Run the deployment script:
     ```
     ./deploy_aws.sh
     ```

4. **Access the Application**:
   - Open your web browser and navigate to:
     ```
     http://your-instance-public-dns:8501
     ```

### Running as a Background Service

To keep the application running after you disconnect from SSH:

```
nohup ./deploy_aws.sh > streamlit.log 2>&1 &
```

This will run the application in the background and save logs to streamlit.log.

## License

This project is licensed under the MIT License - see the LICENSE file for details.