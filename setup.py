import os
import shutil
import pandas as pd

def setup_project():
    """
    Set up the project by copying the dataset to the data directory
    and creating necessary directories if they don't exist.
    """
    print("Setting up project...")
    
    # Create directories if they don't exist
    directories = [
        'src/data',
        'src/models',
        'src/preprocessing',
        'src/visualization',
        'src/app'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Copy dataset to data directory
    source_file = 'bank-full.csv'
    destination_file = 'src/data/bank-full.csv'
    
    if os.path.exists(source_file):
        shutil.copy2(source_file, destination_file)
        print(f"Copied dataset from {source_file} to {destination_file}")
        
        # Create a sample of the dataset for quick testing
        df = pd.read_csv(source_file, sep=';')
        sample_df = df.sample(n=1000, random_state=42)
        sample_df.to_csv('src/data/bank-sample.csv', sep=';', index=False)
        print("Created sample dataset for quick testing")
    else:
        print(f"Warning: Dataset file {source_file} not found")
    
    print("Project setup complete!")

if __name__ == "__main__":
    setup_project()