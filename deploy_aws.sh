#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip if not already installed
sudo apt-get install -y python3 python3-pip

# Install git
sudo apt-get install -y git

# Clone the repository
git clone https://github.com/senorvishnu/Predicting_Term_Deposit_Subscription.git
cd Predicting_Term_Deposit_Subscription

# Install required packages
pip3 install -r requirements.txt

# Run setup script to prepare data
python3 setup.py

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false