# Amazon Linux Deployment Troubleshooting Guide

This guide addresses common issues when deploying the Bank Term Deposit Prediction application on Amazon Linux EC2 instances.

## Common Issues and Solutions

### 1. Command Not Found Errors

**Issue**: Commands like `apt-get` are not found

**Solution**: Amazon Linux uses `yum` instead of `apt-get`
```bash
# Use these commands instead
sudo yum update -y
sudo yum install -y python3 python3-pip git
```

### 2. Python Package Installation Issues

**Issue**: Problems installing Python packages

**Solution**: Ensure pip is properly installed and updated
```bash
sudo yum install -y python3-pip
pip3 install --upgrade pip
```

### 3. Streamlit Not Running

**Issue**: Streamlit command not found or not running properly

**Solution**: Run Streamlit using the Python module syntax
```bash
python3 -m streamlit run app.py
```

### 4. Port Access Issues

**Issue**: Cannot access the application on port 8501

**Solution**: Check security group settings and firewall
```bash
# Verify the application is running
ps aux | grep streamlit

# Check if port 8501 is listening
netstat -tuln | grep 8501

# Ensure security group allows inbound traffic on port 8501
```

### 5. Git Clone Issues

**Issue**: Cannot clone the repository

**Solution**: Install git and verify the repository URL
```bash
sudo yum install -y git
git clone https://github.com/senorvishnu/Predicting_Term_Deposit_Subscription.git
```

### 6. Running the Application in Background

**Issue**: Application stops when SSH session ends

**Solution**: Use nohup to keep the application running
```bash
nohup python3 -m streamlit run app.py --server.port 8501 --server.enableCORS false > streamlit.log 2>&1 &
```

### 7. Checking Application Status

**Issue**: Need to verify if application is running

**Solution**: Use the check_app_status.sh script
```bash
chmod +x check_app_status.sh
./check_app_status.sh
```

## Additional Resources

- [Amazon Linux Documentation](https://docs.aws.amazon.com/linux/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Project GitHub Repository](https://github.com/senorvishnu/Predicting_Term_Deposit_Subscription)

If you encounter issues not covered in this guide, please open an issue on the GitHub repository.