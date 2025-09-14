# AWS EC2 Setup Guide for Bank Term Deposit Prediction App

This guide provides step-by-step instructions for deploying the Bank Term Deposit Prediction application on an AWS EC2 instance.

## 1. Launch an EC2 Instance

1. **Sign in to AWS Console**:
   - Go to https://aws.amazon.com/console/
   - Sign in with your credentials (email: vishnudavid1999@gmail.com)

2. **Navigate to EC2 Dashboard**:
   - From the AWS Management Console, find and click on "EC2" under Services

3. **Launch a New Instance**:
   - Click the "Launch Instance" button
   - Enter a name for your instance (e.g., "Bank-Term-Deposit-App")

4. **Choose an Amazon Machine Image (AMI)**:
   - Select "Ubuntu Server 20.04 LTS (HVM), SSD Volume Type"

5. **Choose an Instance Type**:
   - For testing: t2.micro (Free tier eligible)
   - For production: t2.small or t2.medium (recommended for better performance)

6. **Configure Key Pair**:
   - Create a new key pair or use an existing one
   - If creating new, download the .pem file and keep it secure
   - **IMPORTANT**: You cannot download the key pair again after creation

7. **Configure Network Settings**:
   - Create a new security group or select an existing one
   - Add the following inbound rules:
     - SSH (Port 22) - Source: Your IP or 0.0.0.0/0 (less secure)
     - Custom TCP (Port 8501) - Source: 0.0.0.0/0 (for Streamlit access)

8. **Configure Storage**:
   - Default 8GB is sufficient for this application
   - Increase if you plan to store more data

9. **Review and Launch**:
   - Review your instance configuration
   - Click "Launch Instance"

## 2. Connect to Your EC2 Instance

### For Windows Users:

1. **Open PowerShell or Command Prompt**

2. **Navigate to the directory containing your .pem file**:
   ```
   cd path\to\key-directory
   ```

3. **Set proper permissions for your key file**:
   ```
   icacls "your-key-file.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"
   ```

4. **Connect to your instance**:
   ```
   ssh -i "your-key-file.pem" ubuntu@your-instance-public-dns
   ```
   Replace `your-instance-public-dns` with your actual EC2 public DNS (found in the EC2 Dashboard)

### For Mac/Linux Users:

1. **Open Terminal**

2. **Navigate to the directory containing your .pem file**:
   ```
   cd /path/to/key-directory
   ```

3. **Set proper permissions for your key file**:
   ```
   chmod 400 your-key-file.pem
   ```

4. **Connect to your instance**:
   ```
   ssh -i your-key-file.pem ubuntu@your-instance-public-dns
   ```

## 3. Deploy the Application

1. **Upload the deployment script**:

   From your local machine, upload the deployment script to your EC2 instance:

   **For Windows**:
   ```
   scp -i "your-key-file.pem" deploy_aws.sh ubuntu@your-instance-public-dns:~
   ```

   **For Mac/Linux**:
   ```
   scp -i your-key-file.pem deploy_aws.sh ubuntu@your-instance-public-dns:~
   ```

2. **Make the script executable**:
   ```
   chmod +x deploy_aws.sh
   ```

3. **Run the deployment script**:
   ```
   ./deploy_aws.sh
   ```

   This script will:
   - Update the system
   - Install Python and required dependencies
   - Clone the repository from GitHub
   - Install project requirements
   - Run the setup script
   - Start the Streamlit application

4. **Run as a background service** (optional):

   To keep the application running after you disconnect from SSH:
   ```
   nohup ./deploy_aws.sh > streamlit.log 2>&1 &
   ```

   To check if the process is running:
   ```
   ps aux | grep streamlit
   ```

## 4. Access the Application

1. **Find your instance's public DNS**:
   - Go to the EC2 Dashboard
   - Select your running instance
   - Copy the "Public IPv4 DNS" value

2. **Access the application**:
   - Open a web browser
   - Navigate to: `http://your-instance-public-dns:8501`

## Troubleshooting

1. **Cannot connect to the instance**:
   - Verify that your instance is running
   - Check that your security group allows inbound traffic on port 22
   - Ensure you're using the correct key file

2. **Cannot access the Streamlit application**:
   - Verify that your security group allows inbound traffic on port 8501
   - Check if the application is running with: `ps aux | grep streamlit`
   - Check the application logs: `cat streamlit.log`

3. **Application crashes or has errors**:
   - Check the logs: `cat streamlit.log`
   - Ensure all dependencies are installed correctly

## Stopping or Terminating Your Instance

- **To stop your instance** (can be restarted later, storage charges still apply):
  - In the EC2 Dashboard, select your instance
  - Click "Instance state" > "Stop instance"

- **To terminate your instance** (permanent deletion, no charges after termination):
  - In the EC2 Dashboard, select your instance
  - Click "Instance state" > "Terminate instance"

**Note**: Remember to terminate your instance when you're done to avoid unnecessary charges.