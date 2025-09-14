#!/bin/bash

# Update system packages (for Amazon Linux)
sudo yum update -y

# Install Python and pip if not already installed
sudo yum install -y python3 python3-pip

# Install git
sudo yum install -y git

# Clone the repository
git clone https://github.com/senorvishnu/Predicting_Term_Deposit_Subscription.git
cd Predicting_Term_Deposit_Subscription

# Install required packages
pip3 install -r requirements.txt

# Run setup script to prepare data
python3 setup.py

# Run the Streamlit app
echo "Starting Streamlit app..."
python3 -m streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false