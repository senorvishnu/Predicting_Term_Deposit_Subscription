#!/bin/bash

# Script to check if the Streamlit application is running properly

echo "Checking if Streamlit application is running..."

# Check if streamlit process is running
if pgrep -f "python3 -m streamlit run app.py" > /dev/null
then
    echo "✅ Streamlit application is running."
    
    # Get the process ID
    PID=$(pgrep -f "python3 -m streamlit run app.py")
    echo "Process ID: $PID"
    
    # Check how long it's been running
    echo "Running since:"
    ps -p $PID -o lstart=
    
    # Check if port 8501 is listening
    if netstat -tuln | grep -q ":8501 "
    then
        echo "✅ Port 8501 is open and listening."
    else
        echo "❌ Port 8501 is not listening. There might be an issue with the application."
    fi
    
    # Check memory usage
    echo "Memory usage:"
    ps -p $PID -o %mem=
    
    # Check CPU usage
    echo "CPU usage:"
    ps -p $PID -o %cpu=
else
    echo "❌ Streamlit application is not running."
    echo "Checking logs for errors..."
    
    if [ -f "streamlit.log" ]
    then
        echo "Last 10 lines of streamlit.log:"
        tail -n 10 streamlit.log
    else
        echo "No streamlit.log file found."
    fi
    
    echo ""
    echo "To start the application, run:"
    echo "./deploy_aws.sh"
    echo ""
    echo "Or to run in the background:"
    echo "nohup ./deploy_aws.sh > streamlit.log 2>&1 &"
fi

echo ""
echo "To access the application, open a web browser and navigate to:"
echo "http://$(curl -s http://169.254.169.254/latest/meta-data/public-hostname):8501"