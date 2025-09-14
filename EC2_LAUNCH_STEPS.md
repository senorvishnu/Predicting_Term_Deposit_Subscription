# EC2 Instance Launch Steps

This guide provides specific steps to launch an EC2 instance for the Bank Term Deposit Prediction application.

## Step 1: Sign in to AWS Console

1. Go to https://aws.amazon.com/console/
2. Sign in with your credentials:
   - Email: vishnudavid1999@gmail.com
   - Password: Shiny@2517

## Step 2: Launch an EC2 Instance

1. In the AWS Management Console, search for "EC2" and select it
2. Click on "Launch instance"

## Step 3: Configure Your Instance

1. **Name and tags**:
   - Name: Bank-Term-Deposit-App

2. **Application and OS Images**:
   - Amazon Machine Image (AMI): Amazon Linux 2023 AMI (Free tier eligible)

3. **Instance type**:
   - Select: t2.micro (Free tier eligible) for testing
   - For production: t2.small or t2.medium

4. **Key pair**:
   - Create a new key pair
   - Key pair name: bank-app-key
   - Key pair type: RSA
   - Private key file format: .pem
   - Click "Create key pair" and save the .pem file securely

5. **Network settings**:
   - Click "Edit" to modify settings
   - VPC: Default VPC
   - Auto-assign public IP: Enable
   - Firewall (security groups): Create security group
   - Security group name: bank-app-sg
   - Description: Security group for Bank Term Deposit App
   - Add security group rules:
     - SSH: Port 22, Source: My IP
     - Custom TCP: Port 8501, Source: Anywhere (0.0.0.0/0)

6. **Configure storage**:
   - 8 GB gp2 (default is sufficient)

7. **Advanced details**:
   - No changes needed

8. **Summary**:
   - Review your configuration
   - Click "Launch instance"

## Step 4: Connect to Your Instance

1. Wait for the instance to initialize (Status checks: 2/2 checks passed)
2. Select your instance and click "Connect"
3. Choose the "SSH client" tab
4. Follow the instructions provided to connect using SSH

## Step 5: Deploy the Application

1. Upload the deployment script: 
   ```
   scp -i "path/to/bank-app-key.pem" deploy_aws.sh ec2-user@your-instance-public-dns:~
   ```

2. Connect to your instance:
   ```
   ssh -i "path/to/bank-app-key.pem" ec2-user@your-instance-public-dns
   ```

3. Make the script executable and run it:
   ```
   chmod +x deploy_aws.sh
   ./deploy_aws.sh
   ```

## Step 6: Access Your Application

1. Find your instance's public DNS in the EC2 Dashboard
2. Open a web browser and navigate to:
   ```
   http://your-instance-public-dns:8501
   ```

## Important Notes

- Keep your .pem key file secure and do not share it
- Remember to stop or terminate your instance when not in use to avoid charges
- The free tier includes 750 hours of t2.micro instances per month